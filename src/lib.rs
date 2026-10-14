// 物理範囲を指定してCZTを行う

use std::{error::Error, sync::Arc};

use ndarray::{Array1, ArrayView1, s};
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

struct Chirp
{
    gs: Array1<Complex<f64>>,
    hs: Array1<Complex<f64>>,
}

#[derive(Clone, Copy, Debug)]
pub enum GridEndpoint {
    /// 最大 endpoint を含む grid。
    Included,
    /// 最大 endpoint を含まない grid。
    Excluded,
}

struct ChirpCoeffs {
    a: Complex<f64>,
    w: Complex<f64>,
    post_const: Complex<f64>,
    post_step: Complex<f64>,
}

#[derive(Clone, Copy, Debug)]
/// [`zoom_dft`] の input/output grid 仕様。
pub struct ZoomDftSpec {
    pub i_min: f64,
    pub i_max: f64,
    pub o_min: f64,
    pub o_max: f64,
    pub i_endpoint: GridEndpoint,
    pub i_index_shift: f64,
    pub o_endpoint: GridEndpoint,
    pub o_index_shift: f64,
}

impl ZoomDftSpec
{
    /// デフォルトの unit-range、endpoint-excluded transform spec を構築する。
    pub fn new() -> Self {
        Self {
            i_min: 0.0,
            i_max: 1.0,
            o_min: 0.0,
            o_max: 1.0,
            i_endpoint: GridEndpoint::Excluded,
            i_index_shift: 0.0,
            o_endpoint: GridEndpoint::Excluded,
            o_index_shift: 0.0,
        }
    }

    pub fn i_range(mut self, i_min: f64, i_max: f64) -> Self {
        self.i_min = i_min;
        self.i_max = i_max;
        self
    }
    pub fn o_range(mut self, o_min: f64, o_max: f64) -> Self {
        self.o_min = o_min;
        self.o_max = o_max;
        self
    }
    pub fn i_grid(mut self, endpoint: GridEndpoint, index_shift: f64) -> Self {
        self.i_endpoint = endpoint;
        self.i_index_shift = index_shift;
        self
    }
    pub fn o_grid(mut self, endpoint: GridEndpoint, index_shift: f64) -> Self {
        self.o_endpoint = endpoint;
        self.o_index_shift = index_shift;
        self
    }

    pub fn i_delta(&self, points: usize) -> Result<f64, Box<dyn Error>> {
        grid_delta(self.i_min, self.i_max, points, self.i_endpoint)
    }
    pub fn o_delta(&self, points: usize) -> Result<f64, Box<dyn Error>> {
        grid_delta(self.o_min, self.o_max, points, self.o_endpoint)
    }
    fn coefficients(
        &self,
        input_points: usize,
        output_points: usize,
    ) -> Result<ChirpCoeffs, Box<dyn Error>> {
        let imag = Complex::<f64>::i();
        let delta_q = self.i_delta(input_points)?;
        let delta_r = self.o_delta(output_points)?;
        let shift_q = self.i_min + delta_q * self.i_index_shift;
        let shift_r = self.o_min + delta_r * self.o_index_shift;

        let a_phase = delta_q * shift_r;
        let a = (-imag * a_phase).exp();

        let w_phase = delta_q * delta_r;
        let w = (imag * w_phase).exp();

        let post_const = (imag * shift_q * shift_r).exp() * delta_q;
        let post_step = imag * delta_r * shift_q;
        Ok(ChirpCoeffs {
            a,
            w,
            post_const,
            post_step,
        })
    }

    pub fn plan(self, input_len: usize, output_len: usize) -> Result<ZoomDftPlan, Box<dyn Error>> {
        ZoomDftPlan::new(input_len, output_len, self)
    }

    pub fn zdft(
        &self,
        ys: ArrayView1<'_, f64>,
        m: usize,
    ) -> Result<ZoomDftResult, Box<dyn Error>>
    {
        // sin(x) を exp(i*x) の虚部として表し、CZT を実行して、scale 済み虚部を sine transform として返す。
        let imag = Complex::<f64>::i();
        let delta_q = self.i_delta(ys.len())?;
        let delta_r = self.o_delta(m)?;
        let shift_q = self.i_min + delta_q * self.i_index_shift;
        let shift_r = self.o_min + delta_r * self.o_index_shift;

        let a_phase = delta_q * shift_r;
        let a = (-imag * a_phase).exp();

        let w_phase = delta_q * delta_r;
        let w = (imag * w_phase).exp();

        let mut ys = czt(ys.view(), a, w, m)?;

        let const_post = (imag * shift_q * shift_r).exp() * delta_q;
        let step_post = imag * delta_r * shift_q;

        for (k, yk) in ys.iter_mut().enumerate() {
            let kf = k as f64;
            let post = (step_post * kf).exp();
            *yk = *yk * const_post * post;
        }
        Ok(ZoomDftResult(ys))
    }

    pub fn zdst(&self,
        ys: ArrayView1<'_, f64>,
        m: usize
    ) -> Result<Array1<f64>, Box<dyn Error>> {
        Ok(self.zdft(ys, m)?.zdst())
    }

    pub fn zdct(&self,
        ys: ArrayView1<'_, f64>,
        m: usize
    ) -> Result<Array1<f64>, Box<dyn Error>> {
        Ok(self.zdft(ys, m)?.zdct())
    }

    pub fn xs(&self, m: usize) -> Result<Array1<f64>, Box<dyn Error>>
    {
        // transform spec に対応する出力 grid を生成する。
        let delta = self.o_delta(m)?;
        Ok(Array1::from_iter((0..m).map(|i| {
            delta * ((i as f64) + self.o_index_shift) + self.o_min
        })))
    }
}

pub struct ZoomDftPlan {
    input_len: usize,
    output_len: usize,
    convolution_len: usize,
    forward_fft: Arc<dyn Fft<f64>>,
    inverse_fft: Arc<dyn Fft<f64>>,
    fft_hs: Array1<Complex<f64>>,
    chirps: Array1<Complex<f64>>,
    shifts: Array1<Complex<f64>>,
    output_xs: Array1<f64>,
}

pub struct ZoomDftResult(Array1<Complex<f64>>);

impl ZoomDftResult
{
    pub fn zdst(&self) -> Array1<f64> {
        self.0.mapv(|v| v.im)
    }

    pub fn zdct(&self) -> Array1<f64> {
        self.0.mapv(|v| v.re)
    }

    pub fn view(&self) -> ArrayView1<'_, Complex<f64>> {
        self.0.view()
    }
}

impl ZoomDftPlan
{
    pub fn new(
        input_len: usize,
        output_len: usize,
        spec: ZoomDftSpec,
    ) -> Result<Self, Box<dyn Error>> {
        if input_len < 2 {
            return Err("input_len must be >= 2".into());
        }
        if output_len < 2 {
            return Err("output_len must be >= 2".into());
        }
        let convolution_len = smooth_number(input_len + output_len - 1);
        let coeffs = spec.coefficients(input_len, output_len)?;
        let mut hs = Array1::<Complex<f64>>::zeros(convolution_len);
        let mut chirps = Array1::<Complex<f64>>::zeros(input_len);
        let mut shifts = Array1::<Complex<f64>>::zeros(output_len);
        hs[0] = Complex {
            re: 1.0,
            im: 0.0,
        };
        chirps[0] = Complex {
            re: 1.0,
            im: 0.0,
        };
        shifts[0] = coeffs.post_const;
        for i in 1..input_len {
            let nf = i as f64;
            let wn = coeffs.w.powf(nf * nf * 0.5);
            chirps[i] = coeffs.a.powf(-nf) * wn;
            hs[convolution_len - i] = wn.inv();
        }
        for i in 1..output_len {
            let nf = i as f64;
            let wn = coeffs.w.powf(nf * nf * 0.5);
            shifts[i] = wn * (coeffs.post_step * nf).exp() * coeffs.post_const;
            hs[i] = wn.inv();
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(convolution_len);
        let ifft = planner.plan_fft_inverse(convolution_len);
        if let Some(buf) = hs.as_slice_memory_order_mut() {
            fft.process(buf);
        } else {
            let mut tmp = hs.to_vec();
            fft.process(&mut tmp);
            hs.assign(&Array1::from(tmp));
        }
        let output_xs = spec.xs(output_len)?;
        Ok(Self {
            input_len,
            output_len,
            convolution_len,
            forward_fft: fft,
            inverse_fft: ifft,
            fft_hs: hs,
            chirps,
            shifts,
            output_xs,
        })
    }

    pub fn zdft(&self, ys: ArrayView1<'_, f64>) -> Result<ZoomDftResult, Box<dyn Error>> {
        if self.input_len != ys.len() {
            return Err(format!("input_len must be == {}", self.input_len).into());
        }
        let mut gs = Array1::<Complex<f64>>::zeros(self.convolution_len);
        gs[0] = Complex {
            re: ys[0],
            im: 0.0,
        };
        for i in 1..self.input_len {
            gs[i] = self.chirps[i] * ys[i];
        }
        if let Some(buf) = gs.as_slice_memory_order_mut() {
            self.forward_fft.process(buf);
        } else {
            let mut tmp = gs.to_vec();
            self.forward_fft.process(&mut tmp);
            gs.assign(&Array1::from(tmp));
        }
        let mut ghs = gs * &self.fft_hs;
        if let Some(buf) = ghs.as_slice_memory_order_mut() {
            self.inverse_fft.process(buf);
        } else {
            let mut tmp = ghs.to_vec();
            self.inverse_fft.process(&mut tmp);
            ghs.assign(&Array1::from(tmp));
        }
        ghs = &ghs / (ghs.len() as f64);

        let res = ghs.slice(s![..self.output_len]).to_owned() * &self.shifts;
        Ok(ZoomDftResult(res))
    }

    pub fn zdst(&self, ys: ArrayView1<'_, f64>) -> Result<Array1<f64>, Box<dyn Error>> {
        Ok(self.zdft(ys)?.zdst())
    }

    pub fn zdct(&self, ys: ArrayView1<'_, f64>) -> Result<Array1<f64>, Box<dyn Error>> {
        Ok(self.zdft(ys)?.zdct())
    }

    pub fn xs(&self) -> ArrayView1<'_, f64> {
        self.output_xs.view()
    }
}

fn grid_delta(min: f64, max: f64, points: usize, endpoint: GridEndpoint) -> Result<f64, Box<dyn Error>>
{
    let denom = match endpoint {
        GridEndpoint::Included => points
            .checked_sub(1)
            .ok_or("grid with included endpoint must have at least 2 points")?,
        GridEndpoint::Excluded => points,
    };
    if denom == 0 {
        return Err("grid denominator must be non-zero".into());
    }
    Ok((max - min) / (denom as f64))
}

fn smooth_number(n: usize) -> usize {
    // 2 と 3 のべき乗のみを使い、次の FFT-friendly な長さを求める。
    if n <= 1 {
        return 1;
    }
    let pow2: Array1<usize> =
        Array1::from_iter(0..(n - 1).ilog2() + 2).mapv(|x| (2 as usize).pow(x));
    let pow3: Array1<usize> =
        Array1::from_iter(0..(n - 1).ilog(3) + 2).mapv(|x| (3 as usize).pow(x));
    let mut num: usize = n;
    let mut rem: usize = std::usize::MAX;
    for i in 0..pow2.len() {
        if n == pow2[i] {
            return n;
        }
        let thr = n.div_ceil(pow2[i]);
        let j = pow3.to_vec().partition_point(|&y| y < thr);
        if j >= pow3.len() {
            continue;
        }
        let p = pow2[i] * pow3[j];
        if p - n < rem {
            rem = p - n;
            num = p;
        } else if p == n {
            return p;
        }
    }
    num
}

fn apply_input_chirp(xs: ArrayView1<'_, f64>, m: usize, a: Complex<f64>, w: Complex<f64>) -> Chirp
{
    let n = xs.len();
    let l = smooth_number(n + m - 1);
    let mut gs = Array1::<Complex<f64>>::zeros(l);
    let mut hs = Array1::<Complex<f64>>::zeros(l);
    gs[0] = Complex {
        re: xs[0],
        im: 0.0,
    };
    hs[0] = Complex {
        re: 1.0,
        im: 0.0,
    };
    for i in 1..n {
        let nf = i as f64;
        let wn = w.powf(nf * nf * 0.5);
        let chirp = a.powf(-nf) * wn;
        gs[i] = chirp * xs[i];
        hs[l - i] = wn.inv();
    }
    for i in 1..m {
        let nf = i as f64;
        let wn = w.powf(nf * nf * 0.5);
        hs[i] = wn.inv();
    }
    Chirp { gs, hs }
}

fn czt(
    xs: ArrayView1<'_, f64>,
    a: Complex<f64>,
    w: Complex<f64>,
    m: usize,
) -> Result<Array1<Complex<f64>>, Box<dyn Error>>
{
    // convolution による Chirp-z transform。FFT 処理は `rustfft` が行い、
    // pre/post chirp factor で sample を指定 contour 上へ移す。
    if m < 2 {
        return Err("m must be >= 2".into());
    }
    if xs.len() < 2 {
        return Err("input len must be >= 2".into());
    }
    let Chirp { mut gs, mut hs } = apply_input_chirp(xs, m, a, w);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(gs.len());
    let ifft = planner.plan_fft_inverse(gs.len());
    if let Some(buf) = gs.as_slice_memory_order_mut() {
        fft.process(buf);
    } else {
        let mut tmp = gs.to_vec();
        fft.process(&mut tmp);
        gs.assign(&Array1::from(tmp));
    }
    if let Some(buf) = hs.as_slice_memory_order_mut() {
        fft.process(buf);
    } else {
        let mut tmp = hs.to_vec();
        fft.process(&mut tmp);
        hs.assign(&Array1::from(tmp));
    }
    let mut ghs = gs * hs;
    if let Some(buf) = ghs.as_slice_memory_order_mut() {
        ifft.process(buf);
    } else {
        let mut tmp = ghs.to_vec();
        ifft.process(&mut tmp);
        ghs.assign(&Array1::from(tmp));
    }
    ghs = &ghs / (ghs.len() as f64);

    let mut ys = ghs.slice(s![..m]).to_owned();
    for (k, yk) in ys.iter_mut().enumerate() {
        let kf = k as f64;
        let post = w.powf(kf * kf * 0.5);
        *yk = *yk * post;
    }
    Ok(ys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn assert_all_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.len(), expected.len());

        for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1.0e-11 + 1.0e-10 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {i}: actual={actual:.16e}, expected={expected:.16e}, \
                 abs_diff={:.3e}, tolerance={tolerance:.3e}",
                (actual - expected).abs(),
            );
        }
    }

    fn direct_zoom_dst(
        ys: ArrayView1<'_, f64>,
        output_len: usize,
        spec: &ZoomDftSpec,
    ) -> Array1<f64> {
        let delta_q = spec.i_delta(ys.len()).unwrap();
        let delta_r = spec.o_delta(output_len).unwrap();

        Array1::from_iter((0..output_len).map(|k| {
            let r = spec.o_min + delta_r * (k as f64 + spec.o_index_shift);

            ys.iter()
                .enumerate()
                .map(|(j, &y)| {
                    let q = spec.i_min + delta_q * (j as f64 + spec.i_index_shift);
                    y * (q * r).sin()
                })
                .sum::<f64>()
                * delta_q
        }))
    }

    #[test]
    fn plan_matches_existing_and_direct_transform_on_shifted_grids() {
        // Non-unit delta and non-zero ranges/shifts make a missing post_const
        // change both the scale and phase, so this catches that regression.
        let ys = array![0.3, -1.2, 2.4, 0.7, -0.9, 1.8, 0.2];
        let output_len = 11;
        let spec = ZoomDftSpec::new()
            .i_range(0.35, 4.85)
            .o_range(0.2, 3.7)
            .i_grid(GridEndpoint::Included, 0.25)
            .o_grid(GridEndpoint::Excluded, -0.4);

        let expected_direct = direct_zoom_dst(ys.view(), output_len, &spec);
        let expected_existing = spec.zdst(ys.view(), output_len).unwrap();
        let plan = ZoomDftPlan::new(ys.len(), output_len, spec).unwrap();
        let actual = plan.zdst(ys.view()).unwrap();

        assert_all_close(&expected_existing, &expected_direct);
        assert_all_close(&actual, &expected_existing);
    }

    #[test]
    fn plan_is_reusable_with_different_inputs() {
        let first = array![1.0, 0.5, -0.25, 2.0, -1.0, 0.75, 0.1, -0.3];
        let second = array![-0.4, 1.1, 0.8, -1.5, 0.2, 0.6, -0.9, 2.2];
        let output_len = 5;
        let spec = ZoomDftSpec::new()
            .i_range(-0.7, 2.9)
            .o_range(0.15, 1.65)
            .i_grid(GridEndpoint::Excluded, -0.5)
            .o_grid(GridEndpoint::Included, 0.75);
        let plan = ZoomDftPlan::new(first.len(), output_len, spec).unwrap();

        for ys in [&first, &second] {
            let expected = spec.zdst(ys.view(), output_len).unwrap();
            let actual = plan.zdst(ys.view()).unwrap();
            assert_all_close(&actual, &expected);
        }
    }
}
