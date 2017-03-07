use arraydiff::prelude::*;
use densearray::prelude::*;

use rand::chacha::{ChaChaRng};
use std::cmp::{min};
use std::rc::{Rc};

pub trait Schedule {
  fn at_iter(&self, iter_nr: usize) -> f64;
}

pub struct ConstantSchedule(pub f64);

pub struct PiecewiseSchedule {
  pub init_value:   f64,
  pub pieces:       Vec<(usize, f64)>,
}

pub struct SgdConfig {
  pub batch_sz:         usize,
  pub minibatch_sz:     usize,
  pub step_size:        Box<Schedule>,
  pub momentum:         f64,
}

pub struct Sgd {
  cfg:      SgdConfig,
  consts_params:        VarSet,
  consts_params_grads:  VarSet,
  params:               VarSet,
  grads:                VarSet,
  obj:      Rc<AutodiffSink>,
  iter_nr:  usize,
  param:        Vec<f32>,
  grad:         Vec<f32>,
  prev_param:   Array1d<f32>,
  step:         Array1d<f32>,
}

impl Sgd {
  pub fn new(obj: Rc<AutodiffSink>, const_vars: VarSet, param_vars: VarSet) -> Self {
    let new_txn = txn();
    let consts = const_vars.filter(|v| v.kind == Val);
    let params = param_vars.filter(|v| v.kind == Val);
    let grads = param_vars.filter(|v| v.kind == Grad);
    let consts_params = consts.clone().union(params.clone());
    let consts_params_grads = consts.union(params.clone()).union(grads.clone());
    /*let param_dim = obj.serial_size(new_txn, &mut params);
    let grad_dim = obj.serial_size(new_txn, &mut grads);
    assert_eq!(param_dim, grad_dim);*/
    unimplemented!();
  }

  pub fn reset(&mut self, rng: &mut ChaChaRng) {
    let init_txn = txn();
    // FIXME
    //self.obj.init(init_txn, rng);
    self.obj.store_val(init_txn, &mut self.params, 0, &mut self.param);
    self.prev_param.as_view_mut().copy(self.param.flatten());
    self.iter_nr = 0;
  }

  pub fn step<BatchFn>(&mut self, batch_fn: BatchFn) where BatchFn: Fn(TxnId, usize, Rc<AutodiffSink>) {
    // Calculate the gradient.
    let num_batches = (self.cfg.minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;
    for batch_nr in 0 .. num_batches {
      let batch_txn = txn();
      match batch_nr {
        0 => self.obj.rollover(batch_txn, &mut self.consts_params),
        _ => self.obj.rollover(batch_txn, &mut self.consts_params_grads),
      }
      let batch_size = min(self.cfg.batch_sz, self.cfg.minibatch_sz - batch_nr * self.cfg.batch_sz);
      batch_fn(batch_txn, batch_size, self.obj.clone());
      self.obj.gradient(batch_txn);
    }

    // Store the gradient.
    let store_txn = txn();
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
