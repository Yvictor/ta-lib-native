//! Public contracts for paired real/integer outputs and the guarded zero-trip loop.
use ta_lib::{Core, FuncUnstId, OutRange, RetCode};

const POISON: f64 = 123456.75;
const IPOISON: i32 = -1234567;

fn bars(n: usize, flat: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let close: Vec<f64> = (0..n)
        .map(|i| if flat { if i % 2 == 0 { -0.0 } else { 0.0 } }
             else { 100.0 + ((i * 17 % 53) as f64 - 26.0) / 8.0 })
        .collect();
    let high = close.iter().enumerate().map(|(i, &v)| if flat { v } else { v + (i % 3) as f64 / 4.0 }).collect();
    let low = close.iter().enumerate().map(|(i, &v)| if flat { v } else { v - (i % 5) as f64 / 4.0 }).collect();
    (high, low, close)
}

#[test]
fn zero_trip_partial_ranges_and_minimal_output_slices() {
    for unstable in [0, 3] {
        let core = Core::builder().unstable_period(FuncUnstId::ATR, unstable).build().unwrap();
        for period in [2, 10] {
            let lb = core.supertrend_lookback(period, 3.0).unwrap();
            for n in [1, lb, lb + 1, lb + 2, 100] {
                let (high, low, close) = bars(n, false);
                for start in [0, n / 2, n - 1] {
                    let first = start.max(lb);
                    let count = n.saturating_sub(first);
                    let mut values = vec![POISON; count.max(1) + 2];
                    let mut trends = vec![IPOISON; count.max(1) + 2];
                    let out = core.supertrend(start, n - 1, &high, &low, &close, period, 3.0,
                                             &mut values[1..1 + count.max(1)],
                                             &mut trends[1..1 + count.max(1)]).unwrap();
                    assert_eq!(out, if count == 0 { OutRange { beg_idx: 0, count: 0 } }
                                           else { OutRange { beg_idx: first, count } });
                    assert_eq!(values[0].to_bits(), POISON.to_bits());
                    assert_eq!(trends[0], IPOISON);
                    assert!(values[count + 1..].iter().all(|v| v.to_bits() == POISON.to_bits()));
                    assert!(trends[count + 1..].iter().all(|&v| v == IPOISON));
                    if count > 0 {
                        for shorten_real in [false, true] {
                            let mut short_values = vec![POISON; count - usize::from(shorten_real)];
                            let mut short_trends = vec![IPOISON; count - usize::from(!shorten_real)];
                            assert_eq!(core.supertrend(start, n - 1, &high, &low, &close, period, 3.0,
                                                      &mut short_values, &mut short_trends), Err(RetCode::BadParam));
                            assert!(short_values.iter().all(|v| v.to_bits() == POISON.to_bits()));
                            assert!(short_trends.iter().all(|&v| v == IPOISON));
                        }
                    }
                }
            }
        }
    }
    let mut values = [POISON]; let mut trends = [IPOISON];
    assert_eq!(Core::new().supertrend(0, 0, &[], &[], &[], 10, 3.0, &mut values, &mut trends), Err(RetCode::BadParam));
    let (high, low, close) = bars(1, false);
    assert_eq!(Core::new().supertrend(0, 0, &high, &low, &close, 10, 3.0, &mut [], &mut trends), Err(RetCode::BadParam));
    assert_eq!(Core::new().supertrend(0, 0, &high, &low, &close, 10, 3.0, &mut values, &mut []), Err(RetCode::BadParam));
    assert_eq!(values[0].to_bits(), POISON.to_bits()); assert_eq!(trends[0], IPOISON);
}

#[test]
fn paired_batch_outputs_match_stream_fill_peek_and_update_bit_for_bit() {
    for unstable in [0, 3] {
        let core = Core::builder().unstable_period(FuncUnstId::ATR, unstable).build().unwrap();
        for period in [2, 10] {
            for multiplier in [0.0, 0.5, 3.0] {
                for flat in [false, true] {
                    let (high, low, close) = bars(192, flat);
                    let lb = core.supertrend_lookback(period, multiplier).unwrap();
                    let mut expected = vec![POISON; close.len()];
                    let mut directions = vec![IPOISON; close.len()];
                    let batch = core.supertrend(0, close.len() - 1, &high, &low, &close, period, multiplier,
                                                &mut expected, &mut directions).unwrap();
                    let seed = lb + 2;
                    let mut initial = vec![POISON; seed - lb];
                    let mut initial_directions = vec![IPOISON; seed - lb];
                    let (mut stream, filled) = core.supertrend_open_and_fill(&high[..seed], &low[..seed], &close[..seed],
                        period, multiplier, &mut initial, &mut initial_directions).unwrap();
                    assert_eq!(filled.beg_idx, batch.beg_idx);
                    assert_eq!(initial_directions, directions[..filled.count]);
                    for (a, b) in initial.iter().zip(&expected) { assert_eq!(a.to_bits(), b.to_bits()); }
                    for i in seed..close.len() {
                        let before = stream.out_range();
                        let peek = stream.peek(high[i], low[i], close[i]).unwrap();
                        assert_eq!(stream.out_range(), before);
                        let update = stream.update(high[i], low[i], close[i]).unwrap();
                        assert_eq!(peek.0.to_bits(), update.0.to_bits()); assert_eq!(peek.1, update.1);
                        assert_eq!(update.0.to_bits(), expected[i - lb].to_bits());
                        assert_eq!(update.1, directions[i - lb]);
                    }
                    assert_eq!(stream.out_range(), batch);
                }
            }
        }
    }
}
