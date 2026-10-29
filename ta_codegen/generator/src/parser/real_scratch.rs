//! Private real scratch has no initial-value contract: write each slot before
//! reading it. C may use malloc, managed backends their usual arrays/pools, and
//! Rust fully initialized quiet NaNs. This is not a public library API.

use std::collections::{HashMap, HashSet};

use crate::backends::builtins::{real_scratch_count, REAL_SCRATCH_ALLOC};
use crate::ir::{Expr, Statement, VarType};
use crate::streaming::{nested_bodies, walk_expr, walk_stmt_own_exprs};

fn statements<'a>(body: &'a [Statement], out: &mut Vec<&'a Statement>) {
    for statement in body {
        out.push(statement);
        for nested in nested_bodies(statement).0 {
            statements(nested, out);
        }
    }
}

fn calls(statement: &Statement) -> Vec<&Expr> {
    let mut count = 0;
    // Count through the exhaustive expression walker; statement-local roots
    // keep an allocation in a nested body owned by its own statement.
    walk_stmt_own_exprs(statement, &mut |root| {
        walk_expr(root, &mut |e| {
            if matches!(e, Expr::FuncCall(name, _) if name == REAL_SCRATCH_ALLOC) {
                count += 1;
            }
        });
    });
    if count == 0 {
        return Vec::new();
    }
    match statement {
        Statement::Assign {
            value: e @ Expr::FuncCall(name, _),
            compound: false,
            ..
        } if name == REAL_SCRATCH_ALLOC && count == 1 => vec![e],
        _ => panic!("{REAL_SCRATCH_ALLOC}: requires a direct plain assignment"),
    }
}

pub(super) fn validate(body: &[Statement], parameters: &HashSet<String>) {
    let mut all = Vec::new();
    statements(body, &mut all);
    if !all.iter().any(|statement| !calls(statement).is_empty()) {
        return;
    }
    assert!(
        !all.iter().any(|s| matches!(s, Statement::CircBuf(_))),
        "{REAL_SCRATCH_ALLOC}: circular-buffer bodies are outside this private intrinsic"
    );
    let mut declarations = HashMap::new();
    let mut ambiguous = parameters.clone();
    for statement in &all {
        if let Statement::VarDecl { name, var_type, .. } = statement {
            if declarations.insert(name.as_str(), var_type).is_some() {
                ambiguous.insert(name.clone());
            }
        }
    }
    for statement in all {
        for call in calls(statement) {
            let Statement::Assign {
                target: Expr::Var(target),
                ..
            } = statement
            else {
                panic!("{REAL_SCRATCH_ALLOC}: destination must be a private real pointer");
            };
            assert!(
                !ambiguous.contains(target)
                    && declarations.get(target.as_str()) == Some(&&VarType::RealPointer),
                "{REAL_SCRATCH_ALLOC}: destination must be a unique local double pointer"
            );
            let Expr::FuncCall(_, args) = call else {
                unreachable!()
            };
            let Expr::Var(count) = real_scratch_count(args) else {
                unreachable!()
            };
            assert!(
                !ambiguous.contains(count)
                    && matches!(
                        declarations.get(count.as_str()),
                        Some(VarType::Integer | VarType::Index)
                    ),
                "{REAL_SCRATCH_ALLOC}: count must be a unique local integer scalar"
            );
        }
    }
}

pub(super) fn reject(body: &[Statement]) {
    let mut all = Vec::new();
    statements(body, &mut all);
    assert!(
        !all.iter().any(|s| !calls(s).is_empty()),
        "{REAL_SCRATCH_ALLOC}: only batch function bodies support private real scratch"
    );
}

#[cfg(test)]
mod tests {
    use crate::parser::c_source::parse_c_source_str;

    fn source(declarations: &str, use_: &str) -> String {
        format!("TA_RetCode test(int startIdx, int endIdx, const double inReal[], double outReal[]) {{ {declarations} {use_} return TA_SUCCESS; }}")
    }

    #[test]
    fn only_private_one_dimensional_real_scratch_with_scalar_count_is_admitted() {
        parse_c_source_str(&source(
            "double *scratch; int count;",
            "scratch = real_scratch_alloc(count);",
        ));
        parse_c_source_str(&source(
            "double *scratch; size_t count;",
            "scratch = real_scratch_alloc(count);",
        ));
        for (declarations, use_) in [
            (
                "int *scratch; int count;",
                "scratch = real_scratch_alloc(count);",
            ),
            (
                "double scratch[4]; int count;",
                "scratch = real_scratch_alloc(count);",
            ),
            (
                "double *scratch; double count;",
                "scratch = real_scratch_alloc(count);",
            ),
            (
                "double *scratch; int count[4];",
                "scratch = real_scratch_alloc(count);",
            ),
            (
                "double *scratch;",
                "scratch = real_scratch_alloc(startIdx);",
            ),
            (
                "double *scratch; int count;",
                "outReal = real_scratch_alloc(count);",
            ),
            (
                "double *scratch; int count;",
                "scratch = real_scratch_alloc(count, count);",
            ),
            (
                "double *scratch; int count;",
                "scratch = real_scratch_alloc(count + 1);",
            ),
            (
                "double *scratch; int count;",
                "scratch = real_scratch_alloc(count++);",
            ),
            (
                "double *scratch; int count;",
                "scratch = real_scratch_alloc((int)count);",
            ),
            (
                "double *scratch; int count;",
                "scratch = real_scratch_alloc(inReal[0]);",
            ),
            (
                "double *scratch; int count;",
                "scratch = real_scratch_alloc(4);",
            ),
            (
                "double *scratch; int count;",
                "unknown(real_scratch_alloc(count));",
            ),
            (
                "double *scratch; int count;",
                "scratch = unknown(real_scratch_alloc(count));",
            ),
            (
                "double *scratch; int count;",
                "{ double *scratch; scratch = real_scratch_alloc(count); }",
            ),
            (
                "double *scratch; int count;",
                "{ int count; scratch = real_scratch_alloc(count); }",
            ),
            (
                "double *scratch = real_scratch_alloc(count); int count;",
                "",
            ),
            (
                "double *scratch; int count;",
                "scratch[0] = real_scratch_alloc(count);",
            ),
        ] {
            assert!(
                std::panic::catch_unwind(|| parse_c_source_str(&source(declarations, use_)))
                    .is_err(),
                "accepted {declarations} {use_}"
            );
        }
    }
    #[test]
    fn helpers_lookbacks_and_circular_storage_do_not_acquire_the_policy() {
        for source in [
            "int test_lookback(int period) { double *scratch; int count; scratch = real_scratch_alloc(count); return 0; }",
            "TA_RetCode test(int startIdx, int endIdx, const double inReal[], double outReal[]) { double *scratch; int count; CIRCBUF_PROLOG(ring,double,4); scratch = real_scratch_alloc(count); return TA_SUCCESS; }",
        ] {
            assert!(std::panic::catch_unwind(|| parse_c_source_str(source)).is_err());
        }
        assert!(std::panic::catch_unwind(|| crate::parser::c_source::parse_helper_file_str(
            "double helper(int count) { double *scratch; scratch = real_scratch_alloc(count); return 0.0; }"
        )).is_err());
    }
}
