use super::*;

use arraydiff::prelude::*;
use arraydiff::ops::*;
use densearray::prelude::*;
use devicemem_cuda::prelude::*;
//use devicemem_cuda::coll::*;
use devicemem_cuda::comm::*;

//use rand::chacha::{ChaChaRng};
//use std::cell::{RefCell};
use std::cmp::{min};
use std::collections::hash_map::{DefaultHasher, RandomState};
use std::hash::{BuildHasher, Hasher};
use std::rc::{Rc};

#[derive(Clone, Debug)]
pub enum Momentum {
  Polyak(f64),
  Nesterov(f64),
}

#[derive(Clone, Debug)]
pub struct SgdConfig {
  pub batch_sz:         usize,
  pub minibatch_sz:     usize,
  pub step_size:        Rc<Schedule>,
  pub momentum:         Option<Momentum>,
  pub l2_reg:           Option<f64>,
  //pub l2_grad_clip:     Option<f64>,
}

pub struct Sgd {
  cfg:          SgdConfig,
  dim:          usize,
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
      dim:          dim,
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

    // TODO: gradient clipping.

    // Normalize the gradient by sample size.
    self.grad.flatten_mut().scale(1.0 / self.cfg.minibatch_sz as f32);

    // Calculate the update.
    let step_size = self.cfg.step_size.at_iter(self.iter_nr) as f32;
    match self.cfg.momentum {
      Some(Momentum::Polyak(mu)) => {
        self.step.as_view_mut().copy(self.param.flatten());
        self.step.as_view_mut().add(-1.0, self.prev_param.as_view());
        self.step.as_view_mut().scale(mu as _);
      }
      Some(Momentum::Nesterov(_)) => {
        // TODO
        unimplemented!();
      }
      None => {
        self.step.as_view_mut().set_constant(0.0);
      }
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
  //allreduce_builder:    DeviceNcclCommBuilder,
  allreduce_builder:    GPURingAllreduceBuilder<f32>,
  rand_state:   RandomState,
}

pub struct GPUSyncSgdWorker {
  worker_rank:  usize,
  num_workers:  usize,
  cfg:          SgdConfig,
  dim:          usize,
  stats_dim:    Option<usize>,
  params:       VarSet,
  grads:        VarSet,
  obj:          Rc<AutodiffSink>,
  iter_nr:      usize,
  stream:       Rc<DeviceStream>,
  param:        DeviceMem<f32>,
  stats:        Option<DeviceMem<f32>>,
  //local_buf:    DeviceMem<f32>,
  //grad_reducer: GPUNcclAllreduceIo<f32>,
  reducer_builder:  GPURingAllreduceBuilder<f32>,
  local_buf:    Option<DeviceMem<f32>>,
  grad_reducer: GPUAllreduceIo<f32>,
  momentum:     DeviceArray1d<f32>,
  //prev_param:   DeviceArray1d<f32>,
  //step:         DeviceArray1d<f32>,
  param_h:      Vec<f32>,
  grad_h:       Vec<f32>,
  stats_h:      Vec<f32>,
  debug_hasher: DefaultHasher,
}

impl GPUSyncSgdBuilder {
  pub fn new(num_workers: usize, /*cfg: SgdConfig*/) -> Self {
    GPUSyncSgdBuilder{
      num_workers:  num_workers,
      //cfg:          cfg,
      //allreduce_builder:    DeviceNcclCommBuilder::new(num_workers),
      allreduce_builder:    GPURingAllreduceBuilder::new(num_workers),
      rand_state:   RandomState::new(),
    }
  }

  pub fn into_worker(self, worker_rank: usize, cfg: SgdConfig, /*mut maybe_stats_ctrl: Option<&mut BatchStatsControl>,*/ obj: Rc<AutodiffSink>, init_txn: TxnId, param_vars: &VarSet, stream: Rc<DeviceStream>) -> GPUSyncSgdWorker {
    let mut params = param_vars.filter(|v| v.kind == Val);
    let grads = param_vars.filter(|v| v.kind == Grad);

    // The function is assumed to be already initialized.
    let dim = obj.val_size(init_txn, &mut params);
    assert!(dim > 0);

    let mut param = DeviceMem::zeros(dim, stream.conn());
    //let local_buf = DeviceMem::zeros(dim, stream.conn());
    let momentum = DeviceArray1d::zeros(dim, stream.conn());
    //let mut prev_param = DeviceArray1d::zeros(dim, stream.conn());
    //let step = DeviceArray1d::zeros(dim, stream.conn());

    /*let allreduce_worker = self.allreduce_builder.into_worker(worker_rank);
    let mut grad_reducer = GPUNcclAllreduceIo::new(allreduce_worker, stream.conn());
    grad_reducer.resize(dim, stream.conn());*/

    let grad_reducer = GPUAllreduceIo::empty(worker_rank);

    assert_eq!(dim, obj.store_val(init_txn, &mut params, 0, &mut param));
    //prev_param.as_view_mut().copy(param.as_ref().flatten(), stream.conn());

    let mut param_h = Vec::with_capacity(dim);
    param_h.resize(dim, 0.0);
    let mut grad_h = Vec::with_capacity(dim);
    grad_h.resize(dim, 0.0);

    let mut debug_hasher = self.rand_state.build_hasher();
    param.as_ref().store_sync(&mut param_h, stream.conn());
    debug_hasher.write(param_h.alias_bytes());
    println!("DEBUG: worker: rank: {} init param hash: {:x}", worker_rank, debug_hasher.finish());

    GPUSyncSgdWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      cfg:          cfg,
      dim:          dim,
      stats_dim:    None,
      params:       params,
      grads:        grads,
      obj:          obj,
      iter_nr:      0,
      stream:       stream,
      param:        param,
      stats:        None,
      //local_buf:    local_buf,
      reducer_builder:  self.allreduce_builder,
      local_buf:    None,
      grad_reducer: grad_reducer,
      momentum:     momentum,
      //prev_param:   prev_param,
      //step:         step,
      param_h:      param_h,
      grad_h:       grad_h,
      stats_h:      vec![],
      debug_hasher: debug_hasher,
    }
  }
}

impl GPUSyncSgdWorker {
  pub fn iteration_count(&self) -> usize {
    self.iter_nr
  }

  pub fn step<BatchFn>(&mut self, mut maybe_stats_ctrl: Option<&mut BatchStatsControl>, mut batch_fn: BatchFn) where BatchFn: FnMut(TxnId, usize, usize, Rc<AutodiffSink>) {
    //self.local_buf.as_mut().set_constant(0.0, self.stream.conn());
    if let Some(ref stats_ctrl) = maybe_stats_ctrl {
      stats_ctrl.reset_accumulators(txn());
    }

    // Calculate the gradient.
    let max_local_minibatch_sz = (self.cfg.minibatch_sz + self.num_workers - 1) / self.num_workers;
    let local_minibatch_sz = min(max_local_minibatch_sz, self.cfg.minibatch_sz - self.worker_rank * max_local_minibatch_sz);
    let num_local_batches = (local_minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;
    let mut batch_offset = 0;
    for batch_nr in 0 .. num_local_batches {
      let batch_txn = txn();
      let batch_size = min(self.cfg.batch_sz, local_minibatch_sz - batch_nr * self.cfg.batch_sz);
      batch_fn(batch_txn, batch_offset, batch_size, self.obj.clone());
      if let Some(ref stats_ctrl) = maybe_stats_ctrl {
        stats_ctrl.accumulate(batch_txn);
      }
      if batch_nr == num_local_batches - 1 {
        if self.grad_reducer.is_empty() {
          self.stats_dim = Some(match maybe_stats_ctrl {
            Some(ref mut stats_ctrl) => stats_ctrl.dim(batch_txn),
            None => 0,
          });
          self.stats = Some(DeviceMem::zeros(self.stats_dim.unwrap(), self.stream.conn()));
          self.local_buf = Some(DeviceMem::zeros(self.dim + self.stats_dim.unwrap(), self.stream.conn()));
          let worker = self.reducer_builder.clone().into_worker(self.worker_rank, self.dim + self.stats_dim.unwrap(), &*self.stream);
          self.grad_reducer.attach(worker, self.stream.conn());
          let mut stats_h = Vec::with_capacity(self.stats_dim.unwrap());
          stats_h.resize(self.stats_dim.unwrap(), 0.0);
          self.stats_h = stats_h;
          if self.worker_rank == 0 {
            println!("DEBUG: worker: rank: {} dim: {} stats dim: {:?} total dim: {}", self.worker_rank, self.dim, self.stats_dim, self.dim + self.stats_dim.unwrap());
          }
        }
        self.obj.store_grad(batch_txn, &mut self.grads, 0, self.local_buf.as_mut().unwrap());
        // TODO: load the batch stats accumulators into the local buffer as well.
        if let Some(ref mut stats_ctrl) = maybe_stats_ctrl {
          stats_ctrl.store_accumulators(batch_txn, self.dim, self.local_buf.as_mut().unwrap());
        }
      }
      batch_offset += batch_size;
    }

    self.local_buf.as_mut().unwrap().as_mut().slice_mut(0, self.dim).flatten_mut()
      .scale(1.0 / self.cfg.minibatch_sz as f32, self.stream.conn());
    if let Some(lambda) = self.cfg.l2_reg {
      self.local_buf.as_mut().unwrap().as_mut().slice_mut(0, self.dim).flatten_mut()
        .add(lambda as _, self.param.as_ref().flatten(), self.stream.conn());
    }
    if maybe_stats_ctrl.is_some() {
      self.local_buf.as_mut().unwrap().as_mut().slice_mut(self.dim, self.dim + self.stats_dim.unwrap()).flatten_mut()
        .scale(1.0 / self.num_workers as f32, self.stream.conn());
    }
    self.grad_reducer.write_allreduce_sum(self.local_buf.as_ref().unwrap().as_ref(), &*self.stream);

    /*let mut grad_norm = None;
    if let Some(clip) = self.cfg.l2_grad_clip {
      self.grad_reducer.as_ref().store_sync(&mut self.grad_h, self.stream.conn());
      grad_norm = Some(self.grad_h.flatten().l2_norm());
      self.grad_reducer.as_mut().flatten_mut().scale(1.0 / 1.0_f32.max(grad_norm.unwrap()), self.stream.conn());
    }*/

    // Calculate the update.
    let step_size = self.cfg.step_size.at_iter(self.iter_nr) as f32;
    match self.cfg.momentum {
      None => {
        self.param.as_mut().flatten_mut().add(-step_size, self.grad_reducer.as_ref().slice(0, self.dim).flatten(), self.stream.conn());
      }
      Some(Momentum::Polyak(mu)) => {
        self.momentum.as_view_mut().scale(mu as _, self.stream.conn());
        self.momentum.as_view_mut().add(-step_size, self.grad_reducer.as_ref().slice(0, self.dim).flatten(), self.stream.conn());
        self.param.as_mut().flatten_mut().add(1.0, self.momentum.as_view(), self.stream.conn());
      }
      Some(Momentum::Nesterov(mu)) => {
        self.param.as_mut().flatten_mut().add(-mu as _, self.momentum.as_view(), self.stream.conn());
        self.momentum.as_view_mut().scale(mu as _, self.stream.conn());
        self.momentum.as_view_mut().add(-step_size, self.grad_reducer.as_ref().slice(0, self.dim).flatten(), self.stream.conn());
        self.param.as_mut().flatten_mut().add((1.0 + mu) as _, self.momentum.as_view(), self.stream.conn());
      }
    }

    /*//if (self.iter_nr + 1) % 100 == 0 {
    {
      self.param.as_ref().store_sync(&mut self.param_h, self.stream.conn());
      self.grad_reducer.as_ref().store_sync(&mut self.grad_h, self.stream.conn());

      /*self.debug_hasher.write(self.grad_h.alias_bytes());
      println!("DEBUG: worker: rank: {} grad hash:  {}", self.worker_rank, self.debug_hasher.finish());
      self.debug_hasher.write(self.param_h.alias_bytes());
      println!("DEBUG: worker: rank: {} param hash: {}", self.worker_rank, self.debug_hasher.finish());*/

      for j in 0 .. self.dim {
        assert!(!self.grad_h[j].is_nan());
      }
      for j in 0 .. self.dim {
        assert!(!self.param_h[j].is_nan());
      }

      /*if self.worker_rank == 0 {
        let mut max_grad = None;
        for j in 0 .. self.dim {
          match max_grad {
            None    => max_grad = Some(self.grad_h[j].abs()),
            Some(g) => max_grad = Some(g.max(self.grad_h[j].abs())),
          }
        }
        let grad_norm = Some(self.grad_h.flatten().l2_norm());
        println!("DEBUG: worker: rank: {} grad norm: {:.6e} max grad: {:.6e}", self.worker_rank, grad_norm.unwrap(), max_grad.unwrap());
      }*/
    }*/

    if (self.iter_nr + 1) % 500 == 0 {
      self.param.as_ref().store_sync(&mut self.param_h, self.stream.conn());
      self.debug_hasher.write(self.param_h.alias_bytes());
      println!("DEBUG: worker: iter: {} rank: {} param hash: {:x}", self.iter_nr + 1, self.worker_rank, self.debug_hasher.finish());

      self.grad_reducer.as_ref().slice(0, self.dim)
        .store_sync(&mut self.grad_h, self.stream.conn());
      self.debug_hasher.write(self.grad_h.alias_bytes());
      println!("DEBUG: worker: iter: {} rank: {} grad hash:  {:x}", self.iter_nr + 1, self.worker_rank, self.debug_hasher.finish());

      self.grad_reducer.as_ref().slice(self.dim, self.dim + self.stats_dim.unwrap())
        .store_sync(&mut self.stats_h, self.stream.conn());
      self.debug_hasher.write(self.stats_h.alias_bytes());
      println!("DEBUG: worker: iter: {} rank: {} stats hash: {:x}", self.iter_nr + 1, self.worker_rank, self.debug_hasher.finish());
    }

    // Apply the update.
    let load_txn = txn();
    self.obj.load_val(load_txn, &mut self.params, 0, &mut self.param);

    // Optionally, update the batch norm running statistics.
    if let Some(ref mut stats_ctrl) = maybe_stats_ctrl {
      self.stats.as_mut().unwrap().as_mut().flatten_mut()
        .average(0.01, self.grad_reducer.buffer().as_ref().slice(self.dim, self.dim + self.stats_dim.unwrap()).flatten(), self.stream.conn());
      stats_ctrl.load_fixed_stats(load_txn, 0, self.stats.as_mut().unwrap());
    }

    self.stream.sync();

    self.iter_nr += 1;
  }
}
