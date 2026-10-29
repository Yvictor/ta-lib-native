use ta_lib::{
    abstract_api::{FuncId, OptInputType},
    Core, MAType, RetCode,
};

const CANARY: f64 = f64::from_bits(0x7ff8_0000_0000_55aa);

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

fn assert_untouched(out: &[f64]) {
    assert!(out.iter().all(|v| v.to_bits() == CANARY.to_bits()));
}

#[test]
fn invalid_bound_and_range_rejections_never_write_output() {
    let core = Core::new();
    let price = vec![100.; 512];
    let periods = vec![f64::NAN; 512];
    for kind in kinds() {
        for (min, max) in [
            (0, 30),
            (-1, 30),
            (10001, 10001),
            (i32::MAX, i32::MAX),
            (2, 0),
            (2, -1),
            (2, 10001),
            (2, i32::MAX),
            (30, 2),
            (i32::MIN, 1),
            (31, i32::MIN),
        ] {
            let mut out = vec![CANARY; 520];
            assert_eq!(
                core.mavp(0, 511, &price, &periods, min, max, kind, &mut out[1..519]),
                Err(RetCode::BadParam)
            );
            assert_eq!(core.mavp_lookback(min, max, kind), Err(RetCode::BadParam));
            assert_untouched(&out);
        }
        for (start, end) in [
            (10, 9),
            (Core::INDEX_MAX + 1, Core::INDEX_MAX + 1),
            (0, Core::INDEX_MAX + 1),
        ] {
            let mut out = vec![CANARY; 520];
            assert!(core
                .mavp(start, end, &price, &periods, 1, 30, kind, &mut out[1..519])
                .is_err());
            assert_untouched(&out);
        }
        let mut out = vec![CANARY; 520];
        assert!(core
            .mavp(
                0,
                511,
                &price[..511],
                &periods,
                1,
                30,
                kind,
                &mut out[1..519]
            )
            .is_err());
        assert!(core
            .mavp(
                0,
                511,
                &price,
                &periods[..511],
                1,
                30,
                kind,
                &mut out[1..519]
            )
            .is_err());
        assert!(core
            .mavp(0, 511, &price, &periods, 1, 30, kind, &mut out[1..2])
            .is_err());
        assert_untouched(&out);
    }
}

#[test]
fn default_sentinels_equal_explicit_bounds_without_changing_padding() {
    let core = Core::new();
    let price: Vec<_> = (0..512).map(|i| 100. + f64::from(i % 13) / 100.).collect();
    let periods: Vec<_> = (0..512)
        .map(|i| [f64::NAN, f64::NEG_INFINITY, -0., 1., 3.75, f64::INFINITY][i % 6])
        .collect();
    for kind in kinds() {
        for (min, max, want_min, want_max) in [
            (i32::MIN, i32::MIN, 2, 30),
            (i32::MIN, 30, 2, 30),
            (1, i32::MIN, 1, 30),
        ] {
            let mut actual = vec![CANARY; 520];
            let mut expected = actual.clone();
            let a = core
                .mavp(
                    0,
                    511,
                    &price,
                    &periods,
                    min,
                    max,
                    kind,
                    &mut actual[1..519],
                )
                .unwrap();
            let b = core
                .mavp(
                    0,
                    511,
                    &price,
                    &periods,
                    want_min,
                    want_max,
                    kind,
                    &mut expected[1..519],
                )
                .unwrap();
            assert_eq!((a.beg_idx, a.count), (b.beg_idx, b.count));
            assert!(actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}

#[test]
fn all_ma_types_match_the_period_oracle_for_ieee_inputs_and_partial_ranges() {
    let core = Core::new();
    let n = 512;
    let raw = [
        f64::NAN,
        f64::NEG_INFINITY,
        -1e300,
        -0.,
        0.,
        f64::from_bits(1),
        0.9999,
        1.,
        1.9999,
        2.,
        3.75,
        29.9999,
        30.,
        30.0001,
        1e300,
        f64::INFINITY,
    ];
    let periods: Vec<_> = (0..n).map(|i| raw[i % raw.len()]).collect();
    let mut cases = 0;
    for kind in kinds() {
        for pattern in 0..8 {
            let mut price: Vec<_> = (0..n)
                .map(|i| match pattern {
                    0 => 0.,
                    1 => -0.,
                    2 => 2e307,
                    3 => {
                        if i % 2 == 0 {
                            2e307
                        } else {
                            -2e307
                        }
                    }
                    _ => 100. + f64::from(i32::try_from(i % 17).unwrap()) / 100.,
                })
                .collect();
            if pattern >= 4 {
                price[n / 2] = [
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::from_bits(0x7ff8_0000_0000_aa55),
                ][pattern - 4];
            }
            for (min, max) in [(1, 1), (1, 2), (1, 30), (2, 30)] {
                for (start, end) in [(0, n - 1), (42, n - 1), (200, 400), (0, 0)] {
                    let mut actual = vec![CANARY; n + 8];
                    let got = core
                        .mavp(
                            start,
                            end,
                            &price,
                            &periods,
                            min,
                            max,
                            kind,
                            &mut actual[1..n + 1],
                        )
                        .unwrap();
                    let anchor = start.max(core.mavp_lookback(min, max, kind).unwrap());
                    let count = if anchor > end { 0 } else { end - anchor + 1 };
                    assert_eq!(
                        (got.beg_idx, got.count),
                        (if count == 0 { 0 } else { anchor }, count)
                    );
                    assert_untouched(&actual[..1]);
                    assert_untouched(&actual[1 + count..]);
                    if count > 0 {
                        let normalized: Vec<_> = periods[anchor..=end]
                            .iter()
                            .map(|&v| {
                                if v.is_nan() || v < f64::from(min) {
                                    min
                                } else if v > f64::from(max) {
                                    max
                                } else {
                                    v as i32
                                }
                            })
                            .collect();
                        assert!(normalized.iter().all(|&p| p >= 1 && p <= max));
                        for period in min..=max {
                            if !normalized.contains(&period) {
                                continue;
                            }
                            // The baseline MAVP scratch starts at zero. Preserve that policy here.
                            let mut expected = vec![0.; n];
                            let want = core
                                .ma(anchor, end, &price, period, kind, &mut expected)
                                .unwrap();
                            assert_eq!((want.beg_idx, want.count), (anchor, count));
                            for i in 0..count {
                                if normalized[i] == period {
                                    assert_eq!(actual[1+i].to_bits(),expected[i].to_bits(),"{kind:?}/pattern{pattern}/{min}/{max}/{start}/{end}/row{i}");
                                }
                            }
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 2048);
}

#[test]
fn maximum_validated_bound_preserves_all_ma_ranges_and_bits() {
    let core = Core::new();
    let n = 60_064;
    let price: Vec<_> = (0..n).map(|i| 100. + f64::from(i % 17) / 100.).collect();
    let periods: Vec<_> = (0..n)
        .map(|i| {
            [
                f64::NAN,
                f64::NEG_INFINITY,
                1.,
                9999.9,
                10000.,
                f64::INFINITY,
            ][usize::try_from(i % 6).unwrap()]
        })
        .collect();
    for kind in kinds() {
        for start in [0, 60_000] {
            let mut actual = vec![CANARY; usize::try_from(n).unwrap() + 8];
            let mut expected = vec![0.; usize::try_from(n).unwrap()];
            let got = core
                .mavp(
                    start,
                    usize::try_from(n - 1).unwrap(),
                    &price,
                    &periods,
                    10000,
                    10000,
                    kind,
                    &mut actual[1..],
                )
                .unwrap();
            let anchor = start.max(core.mavp_lookback(10000, 10000, kind).unwrap());
            let want = core
                .ma(
                    anchor,
                    usize::try_from(n - 1).unwrap(),
                    &price,
                    10000,
                    kind,
                    &mut expected,
                )
                .unwrap();
            assert_eq!((got.beg_idx, got.count), (want.beg_idx, want.count));
            for i in 0..got.count {
                assert_eq!(
                    actual[1 + i].to_bits(),
                    expected[i].to_bits(),
                    "{kind:?}/start{start}/row{i}"
                );
            }
            assert_untouched(&actual[..1]);
            assert_untouched(&actual[1 + got.count..]);
        }
    }
}
