//! Frozen C0.8.3 values around variance rebuilds and the public slice contract.
use ta_lib::{Core, OutRange, RetCode};
const POISON: u64 = 0x7ff8_0000_0000_55aa;
fn prices(n: usize, shape: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let wave = 64.0 + ((i % 37) as i32 - 18) as f64 * 0.125;
            match shape {
                0 => wave,
                1 => {
                    if i == n / 3 {
                        1e9
                    } else {
                        wave
                    }
                }
                2 => {
                    if (i / 41) % 2 == 0 {
                        3.3
                    } else {
                        wave
                    }
                }
                3 => 1e12 + ((i % 37) as i32 - 18) as f64 * 0.125,
                4 => 42.0,
                5 => 0.0,
                6 => -0.0,
                7 => {
                    if i % 3 == 0 {
                        -0.0
                    } else {
                        0.0
                    }
                }
                8 => f64::from_bits((i % 7 + 1) as u64),
                9 => {
                    if i == n / 3 {
                        f64::from_bits(0x7ff8_0000_0000_1234)
                    } else {
                        wave
                    }
                }
                10 => {
                    if i == n / 3 {
                        f64::INFINITY
                    } else {
                        wave
                    }
                }
                11 => {
                    if i == n / 3 {
                        f64::NEG_INFINITY
                    } else {
                        wave
                    }
                }
                12 => {
                    if i % 2 == 0 {
                        1e200
                    } else {
                        -1e200
                    }
                }
                13 => 3.3,
                _ => unreachable!(),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn invoke(
    core: &Core,
    function: &str,
    abstract_api: bool,
    input: &[f64],
    period: i32,
    dev: f64,
    start: usize,
    end: usize,
    output: &mut [f64],
) -> Result<OutRange, RetCode> {
    if abstract_api {
        let mut holder = ta_lib::abstract_api::get_func_handle(function)
            .unwrap()
            .new_call(core);
        holder.set_input(0, input)?;
        holder.set_opt_input(0, period)?;
        holder.set_opt_input(1, dev)?;
        holder.set_output(0, output)?;
        holder.call(start, end)
    } else if function == "var" {
        core.var(start, end, input, period, dev, output)
    } else {
        core.stddev(start, end, input, period, dev, output)
    }
}
#[test]
fn checked_recurrence_matches_frozen_c_for_typed_and_abstract_calls() {
    let core = Core::new();
    let golden = include_bytes!("data/var-checked-windows.bin");
    let mut consumed = 0;
    let mut cases = 0;
    for line in include_str!("data/var-checked-windows.tsv").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 12);
        let number = |n: usize| fields[n].parse::<usize>().unwrap();
        assert_eq!(number(0), cases);
        assert_eq!(number(11), consumed);
        let input = prices(number(2), number(3));
        let start = number(4);
        let end = number(5);
        let period = fields[6].parse::<i32>().unwrap();
        let dev = f64::from_bits(u64::from_str_radix(fields[7], 16).unwrap());
        let begin = number(8);
        let count = number(9);
        let capacity = number(10);
        let expected = &golden[8 * consumed..8 * (consumed + capacity)];
        for abstract_api in [false, true] {
            let capacities = if count == 0 {
                vec![capacity]
            } else {
                vec![capacity, count]
            };
            for cap in capacities {
                let mut out = vec![f64::from_bits(POISON); cap];
                let range = invoke(
                    &core,
                    fields[1],
                    abstract_api,
                    &input,
                    period,
                    dev,
                    start,
                    end,
                    &mut out,
                )
                .unwrap();
                assert_eq!(
                    range,
                    OutRange {
                        beg_idx: begin,
                        count
                    },
                    "{}",
                    fields[0]
                );
                for (i, actual) in out.iter().enumerate() {
                    let want = u64::from_le_bytes(expected[8 * i..8 * i + 8].try_into().unwrap());
                    if i < count && f64::from_bits(want).is_nan() {
                        assert!(actual.is_nan(), "{}[{i}]", fields[0]);
                    } else {
                        assert_eq!(actual.to_bits(), want, "{}[{i}]", fields[0]);
                    }
                }
            }
        }
        consumed += capacity;
        cases += 1;
    }
    assert_eq!(cases, 605);
    assert_eq!(consumed * 8, golden.len());
}
#[test]
fn rejected_parameters_ranges_and_short_slices_leave_outputs_untouched() {
    let core = Core::new();
    let input = prices(64, 0);
    for function in ["var", "stddev"] {
        for period in [0, -1, 100001] {
            let mut out = vec![f64::from_bits(POISON); 71];
            assert_eq!(
                invoke(&core, function, false, &input, period, 1.0, 0, 63, &mut out),
                Err(RetCode::BadParam)
            );
            assert!(out.iter().all(|v| v.to_bits() == POISON));
        }
        for dev in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 4e37] {
            let mut out = vec![f64::from_bits(POISON); 71];
            assert_eq!(
                invoke(&core, function, false, &input, 14, dev, 0, 63, &mut out),
                Err(RetCode::BadParam)
            );
            assert!(out.iter().all(|v| v.to_bits() == POISON));
        }
        for abstract_api in [false, true] {
            for input_len in [0, 1, 63] {
                let mut out = vec![f64::from_bits(POISON); 71];
                assert_eq!(
                    invoke(
                        &core,
                        function,
                        abstract_api,
                        &input[..input_len],
                        14,
                        1.0,
                        0,
                        63,
                        &mut out
                    ),
                    Err(RetCode::BadParam)
                );
                assert!(out.iter().all(|v| v.to_bits() == POISON));
            }
            for cap in [0, 50] {
                let mut out = vec![f64::from_bits(POISON); cap];
                assert_eq!(
                    invoke(
                        &core,
                        function,
                        abstract_api,
                        &input,
                        14,
                        1.0,
                        0,
                        63,
                        &mut out
                    ),
                    Err(RetCode::BadParam)
                );
                assert!(out.iter().all(|v| v.to_bits() == POISON));
            }
            let mut out = vec![f64::from_bits(POISON); 7];
            assert_eq!(
                invoke(
                    &core,
                    function,
                    abstract_api,
                    &input[..2],
                    14,
                    1.0,
                    0,
                    1,
                    &mut out
                ),
                Ok(OutRange::EMPTY)
            );
            assert!(out.iter().all(|v| v.to_bits() == POISON));
        }
        for (start, end, expected) in [
            (
                Core::INDEX_MAX + 1,
                Core::INDEX_MAX + 1,
                RetCode::OutOfRangeStartIndex,
            ),
            (0, Core::INDEX_MAX + 1, RetCode::OutOfRangeEndIndex),
            (8, 7, RetCode::OutOfRangeEndIndex),
        ] {
            let mut out = vec![f64::from_bits(POISON); 71];
            assert_eq!(
                invoke(
                    &core,
                    function,
                    false,
                    &input,
                    0,
                    f64::NAN,
                    start,
                    end,
                    &mut out
                ),
                Err(expected)
            );
            assert!(out.iter().all(|v| v.to_bits() == POISON));
        }
    }
}
