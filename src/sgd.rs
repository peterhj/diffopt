use arraydiff::prelude::*;
use densearray::prelude::*;

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
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    Box<Schedule>,
  pub momentum:     f64,
}

pub struct Sgd {
  cfg:      SgdConfig,
  consts_params:        VarSet,
  consts_params_grads:  VarSet,
  params:               VarSet,
  grads:                VarSet,
  obj:      Rc<AutodiffSink>,
  iter_nr:  usize,
  param:    Array1d<f32>,
  grad:     Array1d<f32>,
  momentum: Array1d<f32>,
}

impl Sgd {
  pub fn new(obj: Rc<AutodiffSink>, const_vars: VarSet, param_vars: VarSet) -> Self {
    let new_txn = txn();
    /*let param_dim = obj.serial_size(new_txn, &mut params);
    let grad_dim = obj.serial_size(new_txn, &mut grads);
    assert_eq!(param_dim, grad_dim);*/
    unimplemented!();
  }

  pub fn step<BatchGrad>(&mut self, batch_grad_fn: BatchGrad) where BatchGrad: Fn(TxnId, usize, Rc<AutodiffSink>) {
    // Calculate the gradient.
    let num_batches = (self.cfg.minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;
    for batch_nr in 0 .. num_batches {
      let batch_txn = txn();
      match batch_nr {
        0 => self.obj.rollover(batch_txn, &mut self.consts_params),
        _ => self.obj.rollover(batch_txn, &mut self.consts_params_grads),
      }
      let batch_size = min(self.cfg.batch_sz, self.cfg.minibatch_sz - batch_nr * self.cfg.batch_sz);
      batch_grad_fn(batch_txn, batch_size, self.obj.clone());
    }

    // FIXME: Store the gradient.
    let store_txn = txn();
    //self.obj.store_grad(store_txn, &mut self.grads, /*0,*/ self.grad.as_mut_slice()); // FIXME

    // Calculate the update.
    let step = self.cfg.step_size.at_iter(self.iter_nr) / self.cfg.minibatch_sz as f64;
    self.momentum.as_view_mut().scale(self.cfg.momentum as _);
    self.momentum.as_view_mut().add(-step as _, self.grad.as_view());
    self.param.as_view_mut().add(1.0, self.momentum.as_view());

    // Apply the update.
    let load_txn = txn();
    //self.obj.load_val(load_txn, &mut self.params, /*0,*/ self.param.as_mut_slice()); // FIXME

    // TODO: Optionally, update the batch norm running statistics.

    self.iter_nr += 1;
  }
}
