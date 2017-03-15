use arraydiff::prelude::*;
use densearray::prelude::*;

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
  pub momentum:         f64,
}

pub struct Sgd {
  cfg:          SgdConfig,
  consts_params:        VarSet,
  consts_params_grads:  VarSet,
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
  pub fn new(cfg: SgdConfig, obj: Rc<AutodiffSink>, param_vars: VarSet, const_vars: VarSet) -> Self {
    let mut params = param_vars.filter(|v| v.kind == Val);
    let grads = param_vars.filter(|v| v.kind == Grad);
    let consts = const_vars.filter(|v| v.kind == Val);
    let mut consts_params = consts.clone().union(params.clone());
    let consts_params_grads = consts.union(params.clone()).union(grads.clone());

    let new_txn = txn();

    // The function is assumed to be already initialized.
    obj.persist(new_txn, &mut consts_params);
    let dim = obj.val_size(new_txn, &mut params);
    assert!(dim > 0);

    let mut param = Vec::with_capacity(dim);
    param.resize(dim, 0.0);
    let mut grad = Vec::with_capacity(dim);
    grad.resize(dim, 0.0);
    let mut prev_param = Array1d::zeros(dim);
    let step = Array1d::zeros(dim);

    assert_eq!(dim, obj.store_val(new_txn, &mut params, 0, &mut param));
    prev_param.as_view_mut().copy(param.flatten());

    Sgd{
      cfg:          cfg,
      consts_params:        consts_params,
      consts_params_grads:  consts_params_grads,
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
      // TODO: give `batch_fn` the responsibility of persistence.
      /*match batch_nr {
        0 => self.obj.persist(batch_txn, &mut self.consts_params),
        _ => self.obj.persist(batch_txn, &mut self.consts_params_grads),
      }*/
      let batch_size = min(self.cfg.batch_sz, self.cfg.minibatch_sz - batch_nr * self.cfg.batch_sz);
      batch_fn(batch_txn, batch_offset, batch_size, self.obj.clone());
      self.obj.gradient(batch_txn);
      batch_offset += batch_size;
    }

    // Store the gradient.
    let store_txn = txn();
    self.obj.persist(store_txn, &mut self.grads);
    self.obj.store_grad(store_txn, &mut self.grads, 0, &mut self.grad);

    // Calculate the update.
    let step_size = self.cfg.step_size.at_iter(self.iter_nr) / self.cfg.minibatch_sz as f64;
    self.step.as_view_mut().copy(self.param.flatten());
    self.step.as_view_mut().add(-1.0, self.prev_param.as_view());
    self.step.as_view_mut().scale(self.cfg.momentum as _);
    self.step.as_view_mut().add(-step_size as _, self.grad.flatten());
    self.prev_param.as_view_mut().copy(self.param.flatten());
    self.param.flatten_mut().add(1.0, self.step.as_view());

    // Apply the update.
    let load_txn = txn();
    self.obj.load_val(load_txn, &mut self.params, 0, &mut self.param);

    // TODO: Optionally, update the batch norm running statistics.

    self.iter_nr += 1;
  }
}
