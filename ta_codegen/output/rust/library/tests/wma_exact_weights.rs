//! Frozen C0.8.3 values on both sides of reseeds, including public tape callers.
use ta_lib::{Core, MAType};

const POISON: u64 = 0x7ff8_0000_0000_55aa;
fn prices(pattern: &str, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let wave = 64.0 + ((i % 37) as i32 - 18) as f64 * 0.125;
            match pattern {
                "wave" => wave,
                "spike" => {
                    if i == n / 3 {
                        1e9
                    } else {
                        wave
                    }
                }
                "highoffset" => 1e12 + ((i % 37) as i32 - 18) as f64 * 0.125,
                "mixedzero" => {
                    if i % 3 == 0 {
                        -0.0
                    } else {
                        0.0
                    }
                }
                _ => panic!("unknown input pattern"),
            }
        })
        .collect()
}

#[test]
fn bounded_weights_match_frozen_c_across_batch_and_stream_reseeds() {
    let core = Core::new();
    let golden = include_bytes!("data/wma-exact-weights.bin");
    let mut cases = 0;
    let mut consumed_words = 0;
    for line in include_str!("data/wma-exact-weights.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = line.split('\t').collect();
        assert_eq!(f.len(), 13);
        let id = f[0];
        let mode = f[1];
        let number = |i: usize| f[i].parse::<usize>().unwrap();
        let n = number(3);
        let start = number(4);
        let end = number(5);
        let period = f[6].parse::<i32>().unwrap();
        let history = number(7);
        let expected_beg = number(8);
        let expected_count = number(9);
        let offset = number(10);
        let words = number(11);
        assert_eq!(offset, consumed_words);
        consumed_words += words;
        let input = prices(f[2], n);
        let periods: Vec<_> = (0..n).map(|i| [2.0, 7.0, 14.0, 30.0][i % 4]).collect();
        let mut output = vec![f64::from_bits(POISON); words];
        let range = match mode {
            "wma" => core.wma(start, end, &input, period, &mut output).unwrap(),
            "ma" => core
                .ma(start, end, &input, period, MAType::WMA, &mut output)
                .unwrap(),
            "mavp" => core
                .mavp(
                    start,
                    end,
                    &input,
                    &periods,
                    2,
                    30,
                    MAType::WMA,
                    &mut output,
                )
                .unwrap(),
            "stream" => {
                let (mut s, last) = core.wma_open(&input[..history], period).unwrap();
                output.clear();
                output.push(last);
                for &next in &input[history..] {
                    let before = s.out_range();
                    let value = s.value().to_bits();
                    let peek = s.peek(next).unwrap();
                    assert_eq!(s.out_range(), before, "{id}: peek range");
                    assert_eq!(s.value().to_bits(), value, "{id}: peek state");
                    let mut fork = s.clone();
                    let forked = fork.update(next).unwrap();
                    let updated = s.update(next).unwrap();
                    assert_eq!(forked.to_bits(), updated.to_bits(), "{id}: clone");
                    output.extend([peek, updated, s.value()]);
                }
                s.out_range()
            }
            "bank" => {
                let (mut s, last) = core
                    .mavp_open(&input[..history], &periods[..history], 2, 30, MAType::WMA)
                    .unwrap();
                output.clear();
                output.push(last);
                for i in history..n {
                    let before = s.out_range();
                    let value = s.value().to_bits();
                    let peek = s.peek(input[i], periods[i]).unwrap();
                    assert_eq!(s.out_range(), before, "{id}: tape peek range");
                    assert_eq!(s.value().to_bits(), value, "{id}: tape peek state");
                    let mut fork = s.clone();
                    let forked = fork.update(input[i], periods[i]).unwrap();
                    let updated = s.update(input[i], periods[i]).unwrap();
                    assert_eq!(forked.to_bits(), updated.to_bits(), "{id}: tape clone");
                    output.extend([peek, updated, s.value()]);
                }
                s.out_range()
            }
            _ => panic!("unknown route"),
        };
        assert_eq!(
            (range.beg_idx, range.count),
            (expected_beg, expected_count),
            "{id}: output range"
        );
        assert_eq!(output.len(), words, "{id}: output length");
        for (i, (actual, expected)) in output
            .iter()
            .zip(golden[offset * 8..(offset + words) * 8].chunks_exact(8))
            .enumerate()
        {
            let expected = u64::from_le_bytes(expected.try_into().unwrap());
            assert_eq!(
                actual.to_bits(),
                expected,
                "{id}/{mode}: word{i}, finite/zero/canary bits"
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 48);
    assert_eq!(consumed_words * 8, golden.len());
}
