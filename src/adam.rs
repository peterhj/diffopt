use super::*;
use sgd::*;

use arraydiff::prelude::*;
use densearray::prelude::*;
use devicemem_cuda::prelude::*;

use std::cmp::{min};
use std::fs::{File, create_dir_all};
use std::path::{PathBuf};
use std::rc::{Rc};

/*const EPSILON_32: f32 = 1.0e-10;*/

pub struct AdamChkptConfig {
  pub chkpt_dir:    PathBuf,
  pub iter_nr:      PathBuf,
  pub grad_m1:      PathBuf,
  pub grad_m2:      PathBuf,
}

#[derive(Clone, Copy, Debug)]
pub enum AdamVariant {
  Adam{gamma1: f64, gamma2: f64, epsilon: f64},
  RMSProp{momentum: f64, gamma1: f64, gamma2: f64, epsilon: f64},
}

#[derive(Clone, Debug)]
pub struct AdamConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    Rc<Schedule>,
  //pub l2_reg:       Option<f64>,
  pub grad_l2_clip: Option<f64>,
  pub variant:      AdamVariant,
}

pub struct Adam {
  cfg:          AdamConfig,
  params:       VarSet,
  grads:        VarSet,
  obj:          Rc<AutodiffSink>,
  iter_nr:      usize,
  param:        Vec<f32>,
  grad:         Vec<f32>,
  grad_m1:      Array1d<f32>,
  grad_m2:      Array1d<f32>,
  tmp_buf:      Array1d<f32>,
  direction:    Array1d<f32>,
}

impl Adam {
  pub fn new(cfg: AdamConfig, obj: Rc<AutodiffSink>, init_txn: TxnId, param_vars: &VarSet) -> Self {
    let mut params = param_vars.filter(|v| v.kind == Val);
    let grads = param_vars.filter(|v| v.kind == Grad);

    // The function is assumed to be already initialized.
    let dim = obj.val_size(init_txn, &mut params);
    assert!(dim > 0);

    let mut param = Vec::with_capacity(dim);
    param.resize(dim, 0.0);
    let mut grad = Vec::with_capacity(dim);
    grad.resize(dim, 0.0);
    let grad_m1 = Array1d::zeros(dim);
    let grad_m2 = Array1d::zeros(dim);
    let tmp_buf = Array1d::zeros(dim);
    let direction = Array1d::zeros(dim);

    //assert_eq!(dim, obj.store_val(init_txn, &mut params, 0, &mut param));

    Adam{
      cfg:          cfg,
      params:       params,
      grads:        grads,
      obj:          obj,
      iter_nr:      0,
      param:        param,
      grad:         grad,
      grad_m1:      grad_m1,
      grad_m2:      grad_m2,
      tmp_buf:      tmp_buf,
      direction:    direction,
    }
  }

  pub fn iteration_count(&self) -> usize {
    self.iter_nr
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

    // Normalize the gradient by sample size.
    self.grad.flatten_mut().scale(1.0 / self.cfg.minibatch_sz as f32);

    match self.cfg.variant {
      AdamVariant::Adam{gamma1, gamma2, epsilon} => {
        // Update the adaptive moments.
        self.grad_m1.as_view_mut().average(gamma1 as _, self.grad.flatten());
        self.tmp_buf.as_view_mut().copy(self.grad.flatten());
        self.tmp_buf.as_view_mut().square();
        self.grad_m2.as_view_mut().average(gamma2 as _, self.tmp_buf.as_view());

        // Calculate the update.
        let step_size = self.cfg.step_size.at_iter(self.iter_nr) as f32;
        self.direction.as_view_mut().copy(self.grad_m2.as_view());
        self.direction.as_view_mut().scale(1.0 / (1.0 - (1.0 - gamma2 as f32).powi(self.iter_nr as i32 + 1)));
        self.direction.as_view_mut().add_scalar(epsilon as _);
        self.direction.as_view_mut().sqrt();
        self.direction.as_view_mut().elem_ldiv(self.grad_m1.as_view());
        self.direction.as_view_mut().scale(1.0 / (1.0 - (1.0 - gamma1 as f32).powi(self.iter_nr as i32 + 1)));
        self.param.flatten_mut().add(-step_size, self.direction.as_view());
      }
      AdamVariant::RMSProp{momentum, gamma1, gamma2, epsilon} => {
        // TODO
        unimplemented!();
      }
    }

    // Apply the update.
    let load_txn = txn();
    self.obj.load_val(load_txn, &mut self.params, 0, &mut self.param);

    // TODO: Optionally, update the batch norm running statistics.

    self.iter_nr += 1;
  }
}

pub struct GPUAdam {
  cfg:          AdamConfig,
  params:       VarSet,
  grads:        VarSet,
  obj:          Rc<AutodiffSink>,
  iter_nr:      usize,
  stream:       Rc<DeviceStream>,
  param:        DeviceMem<f32>,
  grad:         DeviceMem<f32>,
  //grad_norm:    DeviceMem<f32>,
  grad_m1:      DeviceArray1d<f32>,
  grad_m2:      DeviceArray1d<f32>,
  momentum:     DeviceArray1d<f32>,
  tmp_buf:      DeviceArray1d<f32>,
  direction:    DeviceArray1d<f32>,
}

impl GPUAdam {
  pub fn new(cfg: AdamConfig, obj: Rc<AutodiffSink>, init_txn: TxnId, param_vars: &VarSet, stream: Rc<DeviceStream>) -> Self {
    let mut params = param_vars.filter(|v| v.kind == Val);
    let grads = param_vars.filter(|v| v.kind == Grad);

    // The function is assumed to be already initialized.
    let dim = obj.val_size(init_txn, &mut params);
    assert!(dim > 0);

    let param = DeviceMem::zeros(dim, stream.conn());
    let grad = DeviceMem::zeros(dim, stream.conn());
    let grad_m1 = DeviceArray1d::zeros(dim, stream.conn());
    let grad_m2 = DeviceArray1d::zeros(dim, stream.conn());
    let momentum = DeviceArray1d::zeros(dim, stream.conn());
    let tmp_buf = DeviceArray1d::zeros(dim, stream.conn());
    let direction = DeviceArray1d::zeros(dim, stream.conn());

    //assert_eq!(dim, obj.store_val(init_txn, &mut params, 0, &mut param));

    GPUAdam{
      cfg:          cfg,
      params:       params,
      grads:        grads,
      obj:          obj,
      iter_nr:      0,
      stream:       stream,
      param:        param,
      grad:         grad,
      grad_m1:      grad_m1,
      grad_m2:      grad_m2,
      momentum:     momentum,
      tmp_buf:      tmp_buf,
      direction:    direction,
    }
  }

  pub fn iteration_count(&self) -> usize {
    self.iter_nr
  }

  pub fn save_checkpoint(&mut self, cfg: AdamChkptConfig) {
    // FIXME: create the directory if necessary.
    create_dir_all(&cfg.chkpt_dir).ok();
    // FIXME: save the iter nr to file.
    let mut file = File::create(&cfg.iter_nr).unwrap();
    // FIXME: save the 1st grad moment to file.
    let mut file = File::create(&cfg.grad_m1).unwrap();
    // FIXME: save the 2nd grad moment to file.
    let mut file = File::create(&cfg.grad_m2).unwrap();
    // FIXME
    unimplemented!();
  }

  pub fn resume_checkpoint(&mut self, cfg: AdamChkptConfig) {
    // FIXME: create the directory if necessary.
    create_dir_all(&cfg.chkpt_dir).ok();
    // FIXME: load the iter nr from file.
    let mut file = File::open(&cfg.iter_nr).unwrap();
    // FIXME: load the 1st grad moment from file.
    let mut file = File::open(&cfg.grad_m1).unwrap();
    // FIXME: load the 2nd grad moment from file.
    let mut file = File::open(&cfg.grad_m2).unwrap();
    // FIXME
    unimplemented!();
  }

  pub fn set_batch_size(&mut self, new_batch_size: usize) {
    self.cfg.batch_sz = new_batch_size;
  }

  pub fn set_minibatch_size(&mut self, new_minibatch_size: usize) {
    self.cfg.minibatch_sz = new_minibatch_size;
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

    // Normalize the gradient by sample size.
    self.grad.as_mut().flatten_mut().scale(1.0 / self.cfg.minibatch_sz as f32, self.stream.conn());

    // Gradient clipping.
    if let Some(clip_norm) = self.cfg.grad_l2_clip {
      assert!(clip_norm > 0.0);
      let grad_norm = self.grad.as_ref().flatten().l2_norm(self.stream.conn());
      if grad_norm.abs() > clip_norm as f32 {
        self.grad.as_mut().flatten_mut().scale(clip_norm as f32 / grad_norm.abs(), self.stream.conn());
      }
    }

    match self.cfg.variant {
      AdamVariant::Adam{gamma1, gamma2, epsilon} => {
        // Update the adaptive moments.
        self.grad_m1.as_view_mut().average(gamma1 as _, self.grad.as_ref().flatten(), self.stream.conn());
        self.tmp_buf.as_view_mut().copy(self.grad.as_ref().flatten(), self.stream.conn());
        self.tmp_buf.as_view_mut().square(self.stream.conn());
        self.grad_m2.as_view_mut().average(gamma2 as _, self.tmp_buf.as_view(), self.stream.conn());

        // Calculate the update.
        let step_size = self.cfg.step_size.at_iter(self.iter_nr) as f32;
        self.direction.as_view_mut().copy(self.grad_m2.as_view(), self.stream.conn());
        self.direction.as_view_mut().scale(1.0 / (1.0 - (1.0 - gamma2 as f32).powi(self.iter_nr as i32 + 1)), self.stream.conn());
        self.direction.as_view_mut().add_constant(epsilon as _, self.stream.conn());
        self.direction.as_view_mut().sqrt(self.stream.conn());
        self.direction.as_view_mut().elem_ldiv(self.grad_m1.as_view(), self.stream.conn());
        self.direction.as_view_mut().scale(1.0 / (1.0 - (1.0 - gamma1 as f32).powi(self.iter_nr as i32 + 1)), self.stream.conn());
        self.param.as_mut().flatten_mut().add(-step_size, self.direction.as_view(), self.stream.conn());
      }
      AdamVariant::RMSProp{momentum, gamma1, gamma2, epsilon} => {
        // Update the adaptive moments.
        self.grad_m1.as_view_mut().average(gamma1 as _, self.grad.as_ref().flatten(), self.stream.conn());
        self.tmp_buf.as_view_mut().copy(self.grad.as_ref().flatten(), self.stream.conn());
        self.tmp_buf.as_view_mut().square(self.stream.conn());
        self.grad_m2.as_view_mut().average(gamma2 as _, self.tmp_buf.as_view(), self.stream.conn());

        // Calculate the update.
        let step_size = self.cfg.step_size.at_iter(self.iter_nr) as f32;
        self.tmp_buf.as_view_mut().copy(self.grad_m1.as_view(), self.stream.conn());
        //self.tmp_buf.as_view_mut().scale(1.0 / (1.0 - (1.0 - gamma as f32).powi(self.iter_nr as i32 + 1)), self.stream.conn());
        self.tmp_buf.as_view_mut().square(self.stream.conn());
        self.direction.as_view_mut().copy(self.grad_m2.as_view(), self.stream.conn());
        //self.direction.as_view_mut().scale(1.0 / (1.0 - (1.0 - gamma as f32).powi(self.iter_nr as i32 + 1)), self.stream.conn());
        self.direction.as_view_mut().add(-1.0, self.tmp_buf.as_view(), self.stream.conn());
        self.direction.as_view_mut().add_constant(epsilon as _, self.stream.conn());
        self.direction.as_view_mut().sqrt(self.stream.conn());
        /*self.direction.as_view_mut().elem_ldiv(self.grad.as_ref().flatten(), self.stream.conn());
        self.momentum.as_view_mut().scale(momentum as _, self.stream.conn());
        self.momentum.as_view_mut().add(-step_size, self.direction.as_view(), self.stream.conn());
        self.param.as_mut().flatten_mut().add(1.0, self.momentum.as_view(), self.stream.conn());*/
        self.direction.as_view_mut().elem_ldiv(self.grad_m1.as_view(), self.stream.conn());
        self.param.as_mut().flatten_mut().add(1.0, self.direction.as_view(), self.stream.conn());
      }
    }

    // Apply the update.
    let load_txn = txn();
    self.obj.load_val(load_txn, &mut self.params, 0, &mut self.param);

    // TODO: Optionally, update the batch norm running statistics.

    self.iter_nr += 1;
  }
}
