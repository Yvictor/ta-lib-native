use ta_lib::{
    abstract_api::{FuncId, OptInputType},
    Core, MAType, RetCode,
};

const POISON: f64 = f64::from_bits(0x7ff8_0000_0000_55aa);

fn same(actual: f64, expected: f64) -> bool {
    actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan())
}

fn clamp(value: f64, min: i32, max: i32) -> i32 {
    if value.is_nan() || value < f64::from(min) {
        min
    } else if value > f64::from(max) {
        max
    } else {
        value as i32
    }
}

fn oracle(
    core: &Core,
    price: &[f64],
    periods: &[f64],
    band: (i32, i32),
    kind: MAType,
    range: (usize, usize),
) {
    let (min, max) = band;
    let (start, end) = range;
    let mut actual = vec![POISON; price.len()];
    let range = core
        .mavp(start, end, price, periods, min, max, kind, &mut actual)
        .unwrap();
    let anchor = start.max(core.mavp_lookback(min, max, kind).unwrap());
    let count = if anchor > end { 0 } else { end - anchor + 1 };
    assert_eq!(
        (range.beg_idx, range.count),
        (if count == 0 { 0 } else { anchor }, count)
    );
    assert!(actual[count..]
        .iter()
        .all(|v| v.to_bits() == POISON.to_bits()));
    if count == 0 {
        return;
    }
    for period in min..=max {
        if !periods[anchor..=end]
            .iter()
            .any(|&v| clamp(v, min, max) == period)
        {
            continue;
        }
        let mut expected = vec![POISON; price.len()];
        let want = core
            .ma(anchor, end, price, period, kind, &mut expected)
            .unwrap();
        assert_eq!((want.beg_idx, want.count), (anchor, count));
        for i in 0..count {
            if clamp(periods[anchor + i], min, max) == period {
                assert!(
                    same(actual[i], expected[i]),
                    "{kind:?}/{min}/{max}/{start}/{end}/period{period}/row{i}: {} != {}",
                    actual[i],
                    expected[i]
                );
            }
        }
    }
}

#[test]
fn period_selection_preserves_all_metadata_ma_types_and_partial_ranges() {
    let option = FuncId::MAVP
        .info()
        .opt_inputs
        .iter()
        .find(|p| p.param_name == "optInMAType")
        .unwrap();
    let OptInputType::IntegerList { values, .. } = option.kind else {
        panic!("MA type metadata");
    };
    assert!(values.len() >= 16);
    let mut cases = 0;
    for &(code, _) in values {
        let kind = MAType::try_from(i32::try_from(code).unwrap()).unwrap();
        for n in [1usize, 2, 31, 64, 257, 512] {
            let price: Vec<_> = (0..n)
                .map(|i| 100. + ((i * 17) % 97) as f64 / 100.)
                .collect();
            for (min, max) in [(1, 14), (2, 30), (1, 65)] {
                for shape in 0..3 {
                    let periods: Vec<_> = (0..n)
                        .map(|i| match shape {
                            0 => [3., 5., 9., 14.][i % 4],
                            1 => f64::from(max) - (i / (n / 4 + 1)) as f64,
                            _ => [f64::NAN, f64::NEG_INFINITY, 1., 3.75, 9., f64::INFINITY][i % 6],
                        })
                        .collect();
                    for (start, end) in [(0, n - 1), (n / 3, n - 1), (n - 1, n - 1), (0, n / 2)] {
                        oracle(
                            &Core::new(),
                            &price,
                            &periods,
                            (min, max),
                            kind,
                            (start, end),
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(cases >= 3000);
}

#[test]
fn sma_nonfinite_overflow_and_signed_zero_follow_the_ma_oracle() {
    let core = Core::new();
    let n = 257;
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
    let mut cases = 0;
    for pattern in 0..7 {
        let mut price: Vec<_> = (0..n)
            .map(|i| match pattern {
                0 => 1e9 + (i % 11) as f64 / 100.,
                1 => 2e307,
                2 => {
                    if i % 2 == 0 {
                        0.
                    } else {
                        -0.
                    }
                }
                _ => 100. + (i % 17) as f64 / 100.,
            })
            .collect();
        if (3..=5).contains(&pattern) {
            price[n / 2] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][pattern - 3];
        }
        for (start, end) in [(0, n - 1), (42, n - 1), (200, n - 1), (0, 126), (200, 220)] {
            oracle(&core, &price, &periods, (1, 30), MAType::SMA, (start, end));
            cases += 1;
        }
    }
    assert_eq!(cases, 35);
}

#[test]
fn sma_keeps_rejected_ranges_and_buffers_unwritten() {
    let core = Core::new();
    let price = vec![100.; 64];
    let periods: Vec<_> = (0..64).map(|i| [3., 5., 9., 14.][i % 4]).collect();
    let mut output = vec![POISON; 64];
    assert_eq!(
        core.mavp(0, 63, &price, &periods, 30, 2, MAType::SMA, &mut output)
            .unwrap_err(),
        RetCode::BadParam
    );
    assert_eq!(
        core.mavp(
            0,
            63,
            &price[..63],
            &periods,
            2,
            30,
            MAType::SMA,
            &mut output
        )
        .unwrap_err(),
        RetCode::BadParam
    );
    assert_eq!(
        core.mavp(
            0,
            63,
            &price,
            &periods[..63],
            2,
            30,
            MAType::SMA,
            &mut output
        )
        .unwrap_err(),
        RetCode::BadParam
    );
    assert_eq!(
        core.mavp(
            0,
            63,
            &price,
            &periods,
            2,
            30,
            MAType::SMA,
            &mut output[..34]
        )
        .unwrap_err(),
        RetCode::BadParam
    );
    assert!(output.iter().all(|v| v.to_bits() == POISON.to_bits()));
}
