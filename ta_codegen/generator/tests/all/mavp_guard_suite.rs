use crate::common;
use ta_codegen_lib::backends;

fn compact(text: &str) -> String {
    let comments = regex::Regex::new(r"(?s)/\*.*?\*/|//[^\n]*").unwrap();
    comments
        .replace_all(text, "")
        .replace("(int)", "")
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '(' && *c != ')')
        .collect::<String>()
        .replace("asi32", "")
}

fn bodies<'a>(text: &'a str, signature: &str) -> Vec<&'a str> {
    text.match_indices(signature)
        .map(|(at, _)| {
            let tail = &text[at..];
            let open = tail.find('{').unwrap();
            let mut depth = 0;
            for (i, ch) in tail[open..].char_indices() {
                if ch == '{' {
                    depth += 1;
                }
                if ch == '}' {
                    depth -= 1;
                    if depth == 0 {
                        return &tail[open..=open + i];
                    }
                }
            }
            panic!("unclosed batch body")
        })
        .collect()
}

fn top_level(prefix: &str, clause: &str) -> bool {
    let Some(at) = prefix.find(clause) else {
        return false;
    };
    prefix[..at].chars().fold(0_i32, |depth, ch| {
        depth + i32::from(ch == '{') - i32::from(ch == '}')
    }) == 1
}

fn guarded(body: &str, default: &str, reject: &str, braces: bool) -> bool {
    let text = compact(body);
    let Some(prep) = text.find("minUsed=optInMaxPeriod") else {
        return false;
    };
    let prefix = &text[..prep];
    let (open, close) = if braces { ("{", "}") } else { ("", "") };
    for (param, value) in [("optInMinPeriod", 2), ("optInMaxPeriod", 30)] {
        let default = format!("if{param}=={default}{open}{param}={value};{close}");
        let reject = format!("elseif{param}<1||{param}>10000{open}return{reject};{close}");
        if !top_level(prefix, &(default + &reject)) {
            return false;
        }
    }
    top_level(
        prefix,
        &format!("ifoptInMinPeriod>optInMaxPeriod{{return{reject};}}"),
    ) && text.contains("if!tempPeriod>=minPeriodReal")
        && !text.contains("iftempInt<1")
        && !text.contains("ifminUsed<1")
}

#[test]
fn every_mavp_batch_overload_validates_the_floor_invariant_before_preparation() {
    let (func, enums) = common::load_indicator("mavp");
    let registry = common::make_registry();
    let helpers = common::make_helpers();
    let sources = [
        (
            backends::c::generate(&func, &enums, registry, helpers),
            "TA_RetCode TA_MAVP(",
            "TA_INTEGER_DEFAULT",
            "TA_BAD_PARAM",
            false,
            1,
        ),
        (
            backends::c::generate(&func, &enums, registry, helpers),
            "TA_RetCode TA_S_MAVP(",
            "TA_INTEGER_DEFAULT",
            "TA_BAD_PARAM",
            false,
            1,
        ),
        (
            backends::rust_lang::generate(&func, &enums, registry, helpers),
            "pub(crate) fn mavp_impl(",
            "i32::MIN",
            "RetCode::BadParam",
            true,
            1,
        ),
        (
            backends::java::generate(&func, &enums, registry, helpers),
            "RetCode mavpImpl(",
            "Integer.MIN_VALUE",
            "RetCode.BAD_PARAM",
            true,
            2,
        ),
        (
            backends::csharp::generate(&func, &enums, registry, helpers),
            "RetCode MavpImpl(",
            "int.MinValue",
            "RetCode.BadParam",
            true,
            2,
        ),
    ];
    for (source, signature, default, reject, braces, count) in sources {
        let batches = bodies(&source, signature);
        assert_eq!(batches.len(), count, "{signature}");
        for body in batches {
            assert!(guarded(body, default, reject, braces), "{signature}");
            // A removed, weakened, moved or non-returning guard must fail this gate.
            let text = compact(body);
            let (open, close) = if braces { ("{", "}") } else { ("", "") };
            let clause = format!("ifoptInMinPeriod=={default}{open}optInMinPeriod=2;{close}elseifoptInMinPeriod<1||optInMinPeriod>10000{open}return{reject};{close}");
            for bad in [
                text.replacen("optInMinPeriod<1", "optInMinPeriod<0", 1),
                text.replacen("optInMaxPeriod>10000", "optInMaxPeriod>10001", 1),
                text.replacen(&format!("return{reject}"), &format!("ignored{reject}"), 1),
                text.replacen(&clause, &format!("ifdisabled{{{clause}}}"), 1),
                text.replacen(&clause, "", 1) + &clause,
                text.replacen(&clause, &format!("/*{clause}*/"), 1),
            ] {
                assert!(
                    !guarded(&bad, default, reject, braces),
                    "gate must reject changed guard: {signature}"
                );
            }
        }
    }
}
