use ta_lib::{
    abstract_api::{FuncId, OptInputType},
    Core, FuncUnstId, MAType, RetCode,
};

const CANARY: f64 = f64::from_bits(0x7ff8_0000_0000_55aa);
const NAN_SEED: f64 = f64::from_bits(0x7ff8_0000_0000_aa55);

fn kinds() -> Vec<MAType> {
    let option = FuncId::MAVP
        .info()
        .opt_inputs
        .iter()
        .find(|p| p.param_name == "optInMAType")
        .unwrap();
    let OptInputType::IntegerList { values, .. } = option.kind else {
        panic!("MA metadata");
    };
    assert_eq!(values.len(), 16);
    values
        .iter()
        .map(|&(code, _)| MAType::try_from(i32::try_from(code).unwrap()).unwrap())
        .collect()
}

fn prices(pattern: usize) -> Vec<f64> {
    let n = 512;
    let mut price: Vec<_> = (0..n)
        .map(|i| match pattern {
            0 => 100. + ((i * 17) % 97) as f64 / 100.,
            1 => 2e307,
            2 => {
                if i % 2 == 0 {
                    0.
                } else {
                    -0.
                }
            }
            3 => {
                if i % 2 == 0 {
                    2e307
                } else {
                    -2e307
                }
            }
            _ => 100. + (i % 17) as f64 / 100.,
        })
        .collect();
    if pattern >= 4 {
        let value = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][(pattern - 4) % 3];
        let index = [0, 19, 128, 400][(pattern - 4) / 3];
        price[index] = value;
    }
    price
}

fn assert_bits(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{context}: row {i}, {a:?} != {b:?}"
        );
    }
}

fn seeded(n: usize, value: f64) -> Vec<f64> {
    let mut out = vec![value; n + 8];
    out[n..].fill(CANARY);
    out
}

fn tail(out: &[f64], n: usize, count: usize, seed: f64) {
    assert!(out[count..n].iter().all(|v| v.to_bits() == seed.to_bits()));
    assert!(out[n..].iter().all(|v| v.to_bits() == CANARY.to_bits()));
}

#[test]
fn every_ma_overwrites_its_prefix_independent_of_zero_or_nan_initialization() {
    let mut calls = 0;
    for core in [
        Core::new(),
        Core::new()
            .to_builder()
            .unstable_period(FuncUnstId::ALL, 7)
            .build()
            .unwrap(),
    ] {
        for kind in kinds() {
            for period in [1, 2, 3, 14, 30] {
                for pattern in 0..16 {
                    let price = prices(pattern);
                    let n = price.len();
                    for (start, end) in [(0, n - 1), (42, n - 1), (200, 400), (0, 0)] {
                        let mut zero = seeded(n, 0.);
                        let mut nan = seeded(n, NAN_SEED);
                        let a = core
                            .ma(start, end, &price, period, kind, &mut zero)
                            .unwrap();
                        let b = core.ma(start, end, &price, period, kind, &mut nan).unwrap();
                        assert_eq!((a.beg_idx, a.count), (b.beg_idx, b.count));
                        assert_bits(
                            &zero[..a.count],
                            &nan[..b.count],
                            &format!("{kind:?}/p{period}/pattern{pattern}/{start}/{end}"),
                        );
                        tail(&zero, n, a.count, 0.);
                        tail(&nan, n, b.count, NAN_SEED);
                        calls += 1;
                    }
                }
            }
        }
    }
    assert_eq!(calls, 10240);
}

fn normalized(value: f64, min: i32, max: i32) -> i32 {
    if value.is_nan() || value < f64::from(min) {
        min
    } else if value > f64::from(max) {
        max
    } else {
        (value as i32).max(1)
    }
}

#[test]
fn every_mavp_mode_matches_initialized_ma_prefixes_for_ieee_prices_and_partial_ranges() {
    let mut calls = 0;
    for core in [
        Core::new(),
        Core::new()
            .to_builder()
            .unstable_period(FuncUnstId::ALL, 7)
            .build()
            .unwrap(),
    ] {
        for kind in kinds() {
            for pattern in 0..16 {
                let price = prices(pattern);
                let n = price.len();
                for (min, max) in [(1, 14), (2, 30)] {
                    let periods: Vec<_> = (0..n)
                        .map(|i| {
                            [
                                1.,
                                3.75,
                                5.,
                                9.,
                                14.,
                                f64::NAN,
                                f64::INFINITY,
                                f64::NEG_INFINITY,
                            ][i % 8]
                        })
                        .collect();
                    for (start, end) in [(0, n - 1), (42, n - 1), (200, 400), (0, 0)] {
                        let mut actual = seeded(n, CANARY);
                        let range = core
                            .mavp(start, end, &price, &periods, min, max, kind, &mut actual)
                            .unwrap();
                        let anchor = start.max(core.mavp_lookback(min, max, kind).unwrap());
                        let count = if anchor > end { 0 } else { end - anchor + 1 };
                        assert_eq!(
                            (range.beg_idx, range.count),
                            (if count == 0 { 0 } else { anchor }, count)
                        );
                        tail(&actual, n, count, CANARY);
                        if count > 0 {
                            for period in min..=max {
                                if !periods[anchor..=end]
                                    .iter()
                                    .any(|&p| normalized(p, min, max) == period)
                                {
                                    continue;
                                }
                                let mut expected = seeded(n, NAN_SEED);
                                let want = core
                                    .ma(anchor, end, &price, period, kind, &mut expected)
                                    .unwrap();
                                assert_eq!((want.beg_idx, want.count), (anchor, count));
                                tail(&expected, n, count, NAN_SEED);
                                for i in 0..count {
                                    if normalized(periods[anchor + i], min, max) == period {
                                        assert_bits(&actual[i..=i], &expected[i..=i],
                                        &format!("{kind:?}/pattern{pattern}/{start}/{end}/p{period}/row{i}"));
                                    }
                                }
                            }
                        }
                        calls += 1;
                    }
                }
            }
        }
    }
    assert_eq!(calls, 4096);
}

#[test]
fn every_mavp_mode_keeps_rejected_buffers_unwritten() {
    let core = Core::new();
    let price = prices(0);
    let n = price.len();
    let periods: Vec<_> = (0..n).map(|i| [3., 5., 9., 14.][i % 4]).collect();
    for kind in kinds() {
        let mut out = seeded(n, CANARY);
        for (start, end, price_len, period_len, out_len, min, max) in [
            (0, n - 1, n, n, n, 30, 2),
            (0, n - 1, n - 1, n, n, 2, 30),
            (0, n - 1, n, n - 1, n, 2, 30),
            (0, n - 1, n, n, 1, 2, 30),
            (n, n - 1, n, n, n, 2, 30),
        ] {
            assert!(core
                .mavp(
                    start,
                    end,
                    &price[..price_len],
                    &periods[..period_len],
                    min,
                    max,
                    kind,
                    &mut out[..out_len]
                )
                .is_err());
            assert!(out.iter().all(|v| v.to_bits() == CANARY.to_bits()));
        }
        assert_eq!(
            core.mavp(0, n - 1, &price, &periods, 2, 30, kind, &mut [])
                .unwrap_err(),
            RetCode::BadParam
        );
    }
}
