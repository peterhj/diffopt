use arraydiff::prelude::*;
use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use devicemem_cuda::comm::*;

//use rand::chacha::{ChaChaRng};
//use std::cell::{RefCell};
use std::cmp::{min};
use std::rc::{Rc};

pub trait Schedule {
  fn at_iter(&self, iter_nr: usize) -> f64;
}

pub struct ConstantSchedule(pub f64);

impl Schedule for ConstantSchedule {
  fn at_iter(&self, _iter_nr: usize) -> f64 {
    self.0
  }
}

pub struct PiecewiseStepSchedule {
  pub init:     f64,
  pub pieces:   Vec<(usize, f64)>,
}

impl Schedule for PiecewiseStepSchedule {
  fn at_iter(&self, iter_nr: usize) -> f64 {
    if self.pieces.is_empty() {
      return self.init;
    }
    for k in 0 .. self.pieces.len() {
      if iter_nr < self.pieces[k].0 {
        match k {
          0 => return self.init,
          _ => return self.pieces[k-1].1,
        }
      }
    }
    self.pieces[self.pieces.len()-1].1
  }
}

pub struct SgdConfig {
  pub batch_sz:         usize,
  pub minibatch_sz:     usize,
  pub step_size:        Box<Schedule>,
  pub momentum:         Option<f64>,
}

pub struct Sgd {
  cfg:          SgdConfig,
  //params_grads: VarSet,
  params:       VarSet,
  grads:        VarSet,
  obj:          Rc<AutodiffSink>,
  iter_nr:      usize,
  param:        Vec<f32>,
  grad:         Vec<f32>,
  prev_param:   Array1d<f32>,
  step:         Array1d<f32>,
}

impl Sgd {
  pub fn new(cfg: SgdConfig, obj: Rc<AutodiffSink>, init_txn: TxnId, param_vars: &VarSet) -> Self {
    let mut params = param_vars.filter(|v| v.kind == Val);
    let grads = param_vars.filter(|v| v.kind == Grad);
    //let params_grads = params.clone().union(grads.clone());

    // The function is assumed to be already initialized.
    //obj.persist(init_txn, &mut params);
    let dim = obj.val_size(init_txn, &mut params);
    assert!(dim > 0);

    let mut param = Vec::with_capacity(dim);
    param.resize(dim, 0.0);
    let mut grad = Vec::with_capacity(dim);
    grad.resize(dim, 0.0);
    let mut prev_param = Array1d::zeros(dim);
    let step = Array1d::zeros(dim);

    assert_eq!(dim, obj.store_val(init_txn, &mut params, 0, &mut param));
    prev_param.as_view_mut().copy(param.flatten());

    Sgd{
      cfg:          cfg,
      //params_grads: params_grads,
      params:       params,
      grads:        grads,
      obj:          obj,
      iter_nr:      0,
      param:        param,
      grad:         grad,
      prev_param:   prev_param,
      step:         step,
    }
  }

  pub fn step<BatchFn>(&mut self, mut batch_fn: BatchFn) where BatchFn: FnMut(TxnId, usize, usize, Rc<AutodiffSink>) {
    // Calculate the gradient.
    let num_batches = (self.cfg.minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;
    let mut batch_offset = 0;
    for batch_nr in 0 .. num_batches {
      let batch_txn = txn();
      let batch_size = min(self.cfg.batch_sz, self.cfg.minibatch_sz - batch_nr * self.cfg.batch_sz);
      batch_fn(batch_txn, batch_offset, batch_size, self.obj.clone());
      if batch_nr == 0 {
        self.obj.store_val(batch_txn, &mut self.params, 0, &mut self.param);
      }
      if batch_nr == num_batches - 1 {
        self.obj.store_grad(batch_txn, &mut self.grads, 0, &mut self.grad);
      }
      batch_offset += batch_size;
    }

    self.grad.flatten_mut().scale(1.0 / self.cfg.minibatch_sz as f32);

    // Calculate the update.
    let step_size = self.cfg.step_size.at_iter(self.iter_nr) as f32;
    if let Some(mu) = self.cfg.momentum {
      self.step.as_view_mut().copy(self.param.flatten());
      self.step.as_view_mut().add(-1.0, self.prev_param.as_view());
      self.step.as_view_mut().scale(mu as _);
    } else {
      self.step.as_view_mut().set_constant(0.0);
    }
    self.step.as_view_mut().add(-step_size, self.grad.flatten());
    self.prev_param.as_view_mut().copy(self.param.flatten());
    self.param.flatten_mut().add(1.0, self.step.as_view());

    // Apply the update.
    let load_txn = txn();
    self.obj.load_val(load_txn, &mut self.params, 0, &mut self.param);

    // TODO: Optionally, update the batch norm running statistics.

    self.iter_nr += 1;
  }
}

#[derive(Clone)]
pub struct GPUSyncSgdBuilder {
  num_workers:  usize,
  //cfg:          SgdConfig,
  allreduce_builder:    GPURingAllreduceBuilder<f32>,
}

pub struct GPUSyncSgdWorker {
  worker_rank:  usize,
  num_workers:  usize,
  cfg:          SgdConfig,
  params:       VarSet,
  grads:        VarSet,
  obj:          Rc<AutodiffSink>,
  iter_nr:      usize,
  stream:       Rc<DeviceStream>,
  param:        DeviceMem<f32>,
  local_grad:   DeviceMem<f32>,
  grad_reducer: GPUAllreduceIo<f32>,
  prev_param:   DeviceArray1d<f32>,
  step:         DeviceArray1d<f32>,
}

impl GPUSyncSgdBuilder {
  pub fn new(num_workers: usize, /*cfg: SgdConfig*/) -> Self {
    GPUSyncSgdBuilder{
      num_workers:  num_workers,
      //cfg:          cfg,
      allreduce_builder:    GPURingAllreduceBuilder::new(num_workers),
    }
  }

  pub fn into_worker(self, worker_rank: usize, cfg: SgdConfig, obj: Rc<AutodiffSink>, init_txn: TxnId, param_vars: &VarSet, stream: Rc<DeviceStream>) -> GPUSyncSgdWorker {
    let mut params = param_vars.filter(|v| v.kind == Val);
    let grads = param_vars.filter(|v| v.kind == Grad);

    // The function is assumed to be already initialized.
    let dim = obj.val_size(init_txn, &mut params);
    assert!(dim > 0);

    let mut param = DeviceMem::zeros(dim, stream.conn());
    let local_grad = DeviceMem::zeros(dim, stream.conn());
    let mut prev_param = DeviceArray1d::zeros(dim, stream.conn());
    let step = DeviceArray1d::zeros(dim, stream.conn());

    let allreduce_worker = self.allreduce_builder.into_worker(worker_rank, dim, &*stream);
    let grad_reducer = GPUAllreduceIo::new(allreduce_worker, stream.conn());

    assert_eq!(dim, obj.store_val(init_txn, &mut params, 0, &mut param));
    prev_param.as_view_mut().copy(param.as_ref().flatten(), stream.conn());

    GPUSyncSgdWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      cfg:          cfg,
      params:       params,
      grads:        grads,
      obj:          obj,
      iter_nr:      0,
      stream:       stream,
      param:        param,
      local_grad:   local_grad,
      grad_reducer: grad_reducer,
      prev_param:   prev_param,
      step:         step,
    }
  }
}

impl GPUSyncSgdWorker {
  pub fn step<BatchFn>(&mut self, mut batch_fn: BatchFn) where BatchFn: FnMut(TxnId, usize, usize, Rc<AutodiffSink>) {
    // Calculate the gradient.
    let max_local_minibatch_sz = (self.cfg.minibatch_sz + self.num_workers - 1) / self.num_workers;
    let local_minibatch_sz = min(max_local_minibatch_sz, self.cfg.minibatch_sz - self.worker_rank * max_local_minibatch_sz);
    let num_local_batches = (local_minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;
    let mut batch_offset = 0;
    for batch_nr in 0 .. num_local_batches {
      let batch_txn = txn();
      let batch_size = min(self.cfg.batch_sz, local_minibatch_sz - batch_nr * self.cfg.batch_sz);
      batch_fn(batch_txn, batch_offset, batch_size, self.obj.clone());
      if batch_nr == 0 {
        self.obj.store_val(batch_txn, &mut self.params, 0, &mut self.param);
      }
      if batch_nr == num_local_batches - 1 {
        self.obj.store_grad(batch_txn, &mut self.grads, 0, &mut self.local_grad);
      }
      batch_offset += batch_size;
    }

    self.local_grad.as_mut().flatten_mut().scale(1.0 / self.cfg.minibatch_sz as f32, self.stream.conn());
    self.grad_reducer.write_allreduce_sum(self.local_grad.as_ref(), &*self.stream);

    // Calculate the update.
    let step_size = self.cfg.step_size.at_iter(self.iter_nr) as f32;
    if let Some(mu) = self.cfg.momentum {
      self.step.as_view_mut().copy(self.param.as_ref().flatten(), self.stream.conn());
      self.step.as_view_mut().add(-1.0, self.prev_param.as_view(), self.stream.conn());
      self.step.as_view_mut().scale(mu as _, self.stream.conn());
    } else {
      self.step.as_view_mut().set_constant(0.0, self.stream.conn());
    }
    self.step.as_view_mut().add(-step_size, self.grad_reducer.as_ref().flatten(), self.stream.conn());
    self.prev_param.as_view_mut().copy(self.param.as_ref().flatten(), self.stream.conn());
    self.param.as_mut().flatten_mut().add(1.0, self.step.as_view(), self.stream.conn());

    // Apply the update.
    let load_txn = txn();
    self.obj.load_val(load_txn, &mut self.params, 0, &mut self.param);

    // TODO: Optionally, update the batch norm running statistics.

    self.iter_nr += 1;
  }
}
