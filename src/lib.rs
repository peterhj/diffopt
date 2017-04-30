extern crate arraydiff;
extern crate densearray;
extern crate devicemem_cuda;

extern crate rand;

use std::fmt::{Debug};

pub mod adam;
pub mod prelude;
pub mod sgd;

pub trait Schedule: Debug {
  fn at_iter(&self, iter_nr: usize) -> f64;
}

#[derive(Clone, Debug)]
pub struct ConstantSchedule(pub f64);

impl Schedule for ConstantSchedule {
  fn at_iter(&self, _iter_nr: usize) -> f64 {
    self.0
  }
}

#[derive(Clone, Debug)]
pub struct PiecewiseLinearSchedule {
  pub init:     f64,
  pub points:   Vec<(usize, f64)>,
}

impl Schedule for PiecewiseLinearSchedule {
  fn at_iter(&self, iter_nr: usize) -> f64 {
    if self.points.is_empty() {
      return self.init;
    }
    for k in 0 .. self.points.len() {
      if iter_nr < self.points[k].0 {
        let (a, b) = match k {
          0 => (self.init,          self.points[0].1),
          _ => (self.points[k-1].1, self.points[k].1),
        };
        let base_iter = match k {
          0 => 0,
          _ => self.points[k-1].0,
        };
        assert!(iter_nr >= base_iter);
        let t = (iter_nr - base_iter) as f64 / self.points[k].0 as f64;
        return a + t * (b - a);
      }
    }
    self.points[self.points.len()-1].1
  }
}

#[derive(Clone, Debug)]
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
