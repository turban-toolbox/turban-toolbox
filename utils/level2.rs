use pyo3::prelude::*;
use std::cmp::{max, min};

#[pyfunction]
pub fn replace_spikes(
    x: Vec<f64>,
    spikes: Vec<Vec<usize>>,
    spike_replace_before: usize,
    spike_replace_after: usize,
) -> Vec<f64> {
    // sample to use for replacing spikes
    if spikes.len() == 0 {
        return x;
    }
    // despike
    let mut xd = x.clone();
    // println!("xd: {:?}", xd);
    for spike in spikes.iter() {
        // println!("Spikes: {:?}", spike);
        let start = spike.iter().min().expect("danger zone");
        let stop = spike.iter().max().expect("danger zone");
        let context_mean = mean(&vec![
            mean(
                &x[max(*start, spike_replace_before) - spike_replace_before..*start],
            ),
            mean(&x[*stop + 1..min(x.len(), stop + spike_replace_after + 1)]),
        ]);
        for i in (*start..*stop + 1) {
            xd[i] = context_mean;
        }
    }
    xd
}

use num::{Num, Zero};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::ops::{Add, Div};

fn mean<T>(s: &[T]) -> T
where
    T: Copy + Add<T, Output = T> + Div<Output = T> + Zero + TryFrom<u32>,
    <T as std::convert::TryFrom<u32>>::Error: Debug,
{
    s.iter().fold(T::zero(), |acc, &item| acc + item) / T::try_from(s.len() as u32).expect("zero?")
}
