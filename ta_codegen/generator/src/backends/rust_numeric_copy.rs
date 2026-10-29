//! Preserve signed values only where the old renderer already observes them as i32.

use std::collections::{HashMap, HashSet};

use super::{expr_is_i32_typed, expr_is_i32_typed_ctx, RustRenderCtx};
use crate::ir::{BinOp, CircBuf, Expr, Statement, VarType};
use crate::streaming::walk_expr;

pub(super) fn prove(
    body: &[Statement],
    old: &RustRenderCtx,
    signature: &HashSet<String>,
) -> HashSet<String> {
    let mut proof = Proof {
        old,
        declarations: HashMap::new(),
        duplicates: HashSet::new(),
        candidates: HashSet::new(),
        rejected: HashSet::new(),
        writes: HashMap::new(),
    };
    proof.declarations(body);
    proof.duplicates.extend(
        signature
            .iter()
            .chain(&old.int_params)
            .filter(|name| proof.declarations.contains_key(*name))
            .cloned(),
    );
    proof.candidates = proof
        .declarations
        .iter()
        .filter(|(name, ty)| {
            **ty == VarType::Integer
                && old.index_vars.contains(*name)
                && !old.enum_vars.contains_key(*name)
                && !old.int_params.contains(*name)
                && !proof.duplicates.contains(*name)
        })
        .map(|(name, _)| name.clone())
        .collect();
    proof.scan(body);
    let mut valid: HashSet<_> = proof
        .candidates
        .difference(&proof.rejected)
        .cloned()
        .collect();
    loop {
        let next: HashSet<_> = valid
            .iter()
            .filter(|name| {
                proof.writes.get(*name).is_some_and(|writes| {
                    writes.iter().all(|value| match value {
                        Expr::IntLiteral(n) => i32::try_from(*n).is_ok(),
                        Expr::Var(source) => {
                            !proof.duplicates.contains(source)
                                && (proof.old_signed_local(source)
                                    || old.int_params.contains(source)
                                    || valid.contains(source))
                        }
                        _ => false,
                    })
                })
            })
            .cloned()
            .collect();
        if next == valid {
            break;
        }
        valid = next;
    }
    // Literal/parameter-only components are not signed-local copy propagation.
    let mut witnessed = HashSet::new();
    loop {
        let next: HashSet<_> = valid
            .iter()
            .filter(|name| {
                proof.writes[*name].iter().any(|value| {
                    matches!(value, Expr::Var(source)
                if proof.old_signed_local(source) || witnessed.contains(source))
                })
            })
            .cloned()
            .collect();
        if next == witnessed {
            return next;
        }
        witnessed = next;
    }
}

struct Proof<'a> {
    old: &'a RustRenderCtx,
    declarations: HashMap<String, VarType>,
    duplicates: HashSet<String>,
    candidates: HashSet<String>,
    rejected: HashSet<String>,
    writes: HashMap<String, Vec<Expr>>,
}

impl Proof<'_> {
    fn declarations(&mut self, body: &[Statement]) {
        for statement in body {
            match statement {
                Statement::VarDecl { name, var_type, .. } => {
                    if self
                        .declarations
                        .insert(name.clone(), var_type.clone())
                        .is_some()
                    {
                        self.duplicates.insert(name.clone());
                    }
                }
                Statement::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    self.declarations(then_body);
                    self.declarations(else_body);
                }
                Statement::While { body, .. }
                | Statement::DoWhile { body, .. }
                | Statement::For { body, .. }
                | Statement::Block { body } => self.declarations(body),
                Statement::ForC {
                    init, update, body, ..
                } => {
                    self.declarations(std::slice::from_ref(init));
                    self.declarations(std::slice::from_ref(update));
                    self.declarations(body);
                }
                Statement::CircBuf(operation) => {
                    let (id, layout) = match operation {
                        CircBuf::Prolog { id, layout, .. }
                        | CircBuf::Init { id, layout, .. }
                        | CircBuf::InitLocalOnly { id, layout }
                        | CircBuf::Destroy { id, layout } => (id, Some(layout)),
                        CircBuf::Next { id } => (id, None),
                    };
                    self.duplicates.extend([
                        id.clone(),
                        format!("{id}_Idx"),
                        format!("maxIdx_{id}"),
                    ]);
                    if let Some(layout) = layout {
                        for (storage, _) in super::circbuf_storage(id, layout) {
                            self.duplicates.extend([
                                storage.clone(),
                                format!("local_{storage}"),
                                format!("heap_{storage}"),
                            ]);
                        }
                    }
                }
                Statement::Switch { cases, default, .. } => {
                    for (_, body) in cases {
                        self.declarations(body);
                    }
                    self.declarations(default);
                }
                _ => {}
            }
        }
    }

    fn deny(&mut self, expression: &Expr) {
        walk_expr(expression, &mut |e| {
            if let Expr::Var(name)
            | Expr::ArrayAccess(name, _)
            | Expr::PointerDeref(name)
            | Expr::FuncCall(name, _) = e
            {
                if self.candidates.contains(name) {
                    self.rejected.insert(name.clone());
                }
            }
        });
    }

    fn old_signed_local(&self, name: &str) -> bool {
        matches!(
            self.declarations.get(name),
            Some(VarType::Integer | VarType::Index)
        ) && !self.duplicates.contains(name)
            && !self.old.enum_vars.contains_key(name)
            && self.old.sentinel_vars.contains(name)
            && expr_is_i32_typed_ctx(&Expr::Var(name.to_string()), self.old)
    }

    fn old_downcast(&self, candidate: &Expr, partner: &Expr) -> bool {
        matches!((candidate, partner), (Expr::Var(name), Expr::Var(source))
            if self.candidates.contains(name) && self.old_signed_local(source)
                && expr_is_i32_typed_ctx(partner, self.old) && !expr_is_i32_typed(partner)
                && !expr_is_i32_typed(candidate) && self.old.index_vars.contains(name))
    }

    fn read(&mut self, expression: &Expr, copy_destination: bool) {
        match expression {
            Expr::Var(_) if copy_destination => {}
            Expr::BinOp(
                left,
                BinOp::Less
                | BinOp::LessEq
                | BinOp::Greater
                | BinOp::GreaterEq
                | BinOp::Eq
                | BinOp::NotEq,
                right,
            ) if self.old_downcast(left, right) || self.old_downcast(right, left) => {}
            Expr::BinOp(left, BinOp::And | BinOp::Or, right) => {
                self.read(left, false);
                self.read(right, false);
            }
            Expr::Not(inner) => self.read(inner, false),
            _ => self.deny(expression),
        }
    }

    fn write(&mut self, name: &str, value: &Expr, compound: bool) {
        if self.candidates.contains(name) {
            self.writes
                .entry(name.to_string())
                .or_default()
                .push(value.clone());
            if compound {
                self.rejected.insert(name.to_string());
            }
        }
        // This destination keeps its old type unless it separately passes the proof.
        let copy_destination = !compound
            && !self.duplicates.contains(name)
            && self.old.index_vars.contains(name)
            && matches!(
                self.declarations.get(name),
                Some(VarType::Integer | VarType::Index)
            )
            && !self.old.enum_vars.contains_key(name);
        self.read(value, copy_destination);
    }

    fn loop_header(&mut self, statement: &Statement) {
        crate::streaming::walk_stmt_exprs(statement, &mut |e| self.deny(e));
        if let Statement::VarDecl { name, .. } = statement {
            self.rejected.insert(name.clone());
        }
    }

    fn scan(&mut self, body: &[Statement]) {
        for statement in body {
            match statement {
                Statement::VarDecl {
                    name,
                    init,
                    var_type,
                } => {
                    // Rust's generated zero initializer also fits i32.
                    self.write(name, init.as_ref().unwrap_or(&Expr::IntLiteral(0)), false);
                    if let VarType::IntArray(size) | VarType::RealArray(size) = var_type {
                        for word in size.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
                            if self.candidates.contains(word) {
                                self.rejected.insert(word.to_string());
                            }
                        }
                    }
                }
                Statement::Assign {
                    target: Expr::Var(name),
                    value,
                    compound,
                } => self.write(name, value, *compound),
                Statement::Assign { target, value, .. } => {
                    self.deny(target);
                    self.read(value, false);
                }
                Statement::If {
                    condition,
                    then_body,
                    else_body,
                    ..
                } => {
                    self.read(condition, false);
                    self.scan(then_body);
                    self.scan(else_body);
                }
                Statement::While { condition, body } | Statement::DoWhile { condition, body } => {
                    self.deny(condition);
                    self.scan(body);
                }
                Statement::For { var, count, body } => {
                    self.rejected.insert(var.clone());
                    self.deny(count);
                    self.scan(body);
                }
                Statement::ForC {
                    init,
                    condition,
                    update,
                    body,
                } => {
                    self.loop_header(init);
                    self.deny(condition);
                    self.loop_header(update);
                    self.scan(body);
                }
                Statement::Block { body } => self.scan(body),
                Statement::Switch {
                    expr,
                    cases,
                    default,
                } => {
                    self.deny(expr);
                    for (_, body) in cases {
                        self.scan(body);
                    }
                    self.scan(default);
                }
                Statement::Return { value: Some(value) } | Statement::Expr(value) => {
                    self.deny(value);
                }
                Statement::CircBuf(CircBuf::Init { size, .. }) => self.deny(size),
                Statement::Return { value: None }
                | Statement::UnrollHint { .. }
                | Statement::Break
                | Statement::Continue
                | Statement::Comment(_)
                | Statement::CircBuf(_) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper_registry::HelperRegistry;
    use crate::registry::Registry;

    fn prove(body: &[Statement], old: &RustRenderCtx) -> HashSet<String> {
        super::prove(
            body,
            old,
            &HashSet::from([
                "startIdx".into(),
                "endIdx".into(),
                "inReal".into(),
                "optInPeriod".into(),
                "outReal".into(),
            ]),
        )
    }

    fn fixture(extra: &str) -> (Vec<Statement>, RustRenderCtx) {
        let source = format!("TA_RetCode test(int startIdx, int endIdx, const double inReal[], int optInPeriod, double outReal[]) {{ int seed, alias, destination; seed = -1; alias = seed; destination = alias; {extra} return TA_SUCCESS; }}");
        let body = crate::parser::c_source::parse_c_source_str(&source)
            .functions
            .remove(0)
            .body;
        let mut ctx = RustRenderCtx::empty();
        ctx.is_lookback = false;
        super::super::collect_var_types(
            &body,
            &mut ctx.index_vars,
            &mut ctx.real_vars,
            &mut ctx.vec_vars,
            &mut ctx.real_array_vars,
            &mut ctx.int_vec_vars,
        );
        super::super::collect_sentinel_vars(&body, &mut ctx.sentinel_vars);
        super::super::collect_signed_int_vars(
            &body,
            &ctx.index_vars,
            &ctx.real_vars,
            &mut ctx.sentinel_vars,
        );
        for name in &ctx.sentinel_vars {
            ctx.index_vars.remove(name);
        }
        ctx.int_params.insert("optInPeriod".into());
        (body, ctx)
    }

    fn expression(left: &str, right: &str) -> Expr {
        Expr::BinOp(
            Box::new(Expr::Var(left.into())),
            BinOp::Less,
            Box::new(Expr::Var(right.into())),
        )
    }

    fn render(expression: &Expr, ctx: &RustRenderCtx) -> String {
        let registry = Registry::from_dir(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        super::super::render_expr(expression, ctx, &[], &registry, &HelperRegistry::empty())
    }

    #[test]
    // These casts deliberately model the old renderer's wrapping representation.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::invalid_upcast_comparisons
    )]
    fn signed_local_comparison_preserves_old_downcast_in_both_orders() {
        for condition in ["alias < seed", "seed < alias"] {
            let (body, old) = fixture(&format!("if ({condition}) destination = alias;"));
            let aliases = prove(&body, &old);
            assert!(aliases.contains("alias"));
            let expression = if condition.starts_with("alias") {
                expression("alias", "seed")
            } else {
                expression("seed", "alias")
            };
            assert!(render(&expression, &old).contains("as i32"));
            let mut new = old.clone();
            new.index_vars.remove("alias");
            new.sentinel_vars.insert("alias".into());
            assert!(!render(&expression, &new).contains("as i32"));
            for value in [i32::MIN, -2, -1, 0, 1, i32::MAX] {
                let before = value as usize;
                assert_eq!((before as i32) < -1, value < -1);
                assert_eq!(-1 < before as i32, -1 < value);
                assert_eq!(before, value as usize);
            }
        }
    }

    #[test]
    fn intrinsic_i32_parameter_comparison_is_not_the_old_downcast() {
        let (body, old) = fixture("if (alias < optInPeriod) destination = alias;");
        let rendered = render(&expression("alias", "optInPeriod"), &old);
        assert!(rendered.contains("as usize"), "{rendered}");
        assert!(!rendered.contains("as i32"), "{rendered}");
        assert!(!prove(&body, &old).contains("alias"));
    }

    #[test]
    #[allow(clippy::cast_sign_loss)] // The rejected old usize ordering is the test.
    fn literal_and_new_alias_ordering_do_not_inherit_signedness() {
        for condition in [
            "alias < 1",
            "alias > -1",
            "alias < destination",
            "alias < 1.0",
        ] {
            let (body, old) = fixture(&format!("if ({condition}) destination = alias;"));
            assert!(!prove(&body, &old).contains("alias"), "{condition}");
        }
        let old_wrapped = -1_i32 as usize;
        assert_ne!(old_wrapped < 1, -1_i32 < 1);
    }

    #[test]
    fn index_destination_stays_usize_while_source_copy_can_be_signed() {
        let (body, old) =
            fixture("outReal[destination] = 0.0; if (alias < seed) destination = alias;");
        let aliases = prove(&body, &old);
        assert!(aliases.contains("alias"));
        assert!(!aliases.contains("destination"));
        let mut new = old.clone();
        for name in aliases {
            new.index_vars.remove(&name);
            new.sentinel_vars.insert(name);
        }
        let registry = Registry::from_dir(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        let statement = Statement::Assign {
            target: Expr::Var("destination".into()),
            value: Expr::Var("alias".into()),
            compound: false,
        };
        let rendered = super::super::render_statement(
            &statement,
            0,
            &new,
            &[],
            &HashMap::new(),
            &[],
            &[],
            &HashMap::new(),
            &registry,
            &HelperRegistry::empty(),
            &std::cell::Cell::new(0),
        );
        assert!(rendered.contains("(alias) as usize"), "{rendered}");
    }

    #[test]
    fn non_numeric_roles_fail_closed() {
        for extra in [
            "outReal[alias] = 0.0;",
            "while (alias < seed) { break; }",
            "do { destination = 0; } while (alias < seed);",
            "for (alias = 0; alias < seed; alias++) { destination = 0; }",
            "for (destination = 0; destination < alias; destination++) { destination = 0; }",
            "alias += 1;",
            "alias++;",
            "alias = seed + 1;",
            "alias = startIdx;",
            "alias = 2147483648;",
            "unknown(alias);",
            "unknown(&alias);",
            "return alias;",
            "switch (alias) { case 0: break; }",
            "int slots[alias];",
            "{ int alias; alias = seed; }",
        ] {
            let (body, old) = fixture(extra);
            assert!(!prove(&body, &old).contains("alias"), "{extra}");
        }
    }

    #[test]
    fn hidden_capacity_and_ambiguous_sources_are_rejected() {
        let (body, old) = fixture("");
        let mut capacity = body.clone();
        capacity.push(Statement::CircBuf(CircBuf::Init {
            id: "ring".into(),
            layout: crate::ir::CircBufLayout::Plain(VarType::Real),
            size: Expr::Var("alias".into()),
        }));
        assert!(!prove(&capacity, &old).contains("alias"));
        for value in [
            Expr::Literal(1.0),
            Expr::FuncCall("unknown".into(), vec![]),
            Expr::Cast(VarType::Integer, Box::new(Expr::Literal(1.0))),
        ] {
            let mut changed = body.clone();
            changed.push(Statement::Assign {
                target: Expr::Var("alias".into()),
                value,
                compound: false,
            });
            assert!(!prove(&changed, &old).contains("alias"));
        }
        let mut enum_context = old.clone();
        enum_context
            .enum_vars
            .insert("alias".into(), "MAType".into());
        assert!(!prove(&body, &enum_context).contains("alias"));
    }

    #[test]
    fn ambiguous_copy_sinks_and_all_signature_shadows_are_rejected() {
        let (body, old) = fixture("{ int destination; destination = alias; }");
        assert!(!prove(&body, &old).contains("alias"));
        let (body, old) = fixture("");
        for name in ["alias", "destination", "seed"] {
            assert!(!super::prove(&body, &old, &HashSet::from([name.into()])).contains("alias"));
        }
    }

    #[test]
    fn circular_buffer_hidden_names_cannot_be_numeric_aliases() {
        let layout = crate::ir::CircBufLayout::Class(vec![("value".into(), VarType::Real)]);
        for operation in [
            CircBuf::Prolog {
                id: "ring".into(),
                layout: layout.clone(),
                static_size: 4,
            },
            CircBuf::Init {
                id: "ring".into(),
                layout: layout.clone(),
                size: Expr::IntLiteral(4),
            },
            CircBuf::InitLocalOnly {
                id: "ring".into(),
                layout: layout.clone(),
            },
            CircBuf::Destroy {
                id: "ring".into(),
                layout,
            },
        ] {
            for name in [
                "ring",
                "ring_Idx",
                "maxIdx_ring",
                "ring_value",
                "local_ring_value",
                "heap_ring_value",
            ] {
                let (mut body, mut old) = fixture("");
                body.push(Statement::VarDecl {
                    name: name.into(),
                    var_type: VarType::Integer,
                    init: None,
                });
                body.push(Statement::Assign {
                    target: Expr::Var(name.into()),
                    value: Expr::Var("seed".into()),
                    compound: false,
                });
                body.push(Statement::CircBuf(operation.clone()));
                old.index_vars.insert(name.into());
                assert!(!prove(&body, &old).contains(name), "{name}");
            }
        }
        let (mut body, old) = fixture("");
        body.push(Statement::CircBuf(CircBuf::Next { id: "alias".into() }));
        assert!(!prove(&body, &old).contains("alias"));
    }

    #[test]
    fn string_named_pointer_array_and_call_roles_are_rejected() {
        for expression in [
            Expr::PointerDeref("alias".into()),
            Expr::ArrayAccess("alias".into(), Box::new(Expr::IntLiteral(0))),
            Expr::FuncCall("alias".into(), vec![]),
        ] {
            let (mut body, old) = fixture("");
            body.push(Statement::Expr(expression));
            assert!(!prove(&body, &old).contains("alias"));
        }
    }

    #[test]
    fn enum_or_undeclared_signed_sources_are_not_i32_witnesses() {
        let (body, mut old) = fixture("");
        old.enum_vars.insert("seed".into(), "MAType".into());
        assert!(!prove(&body, &old).contains("alias"));
        let (mut body, mut old) = fixture("");
        old.sentinel_vars.insert("phantom".into());
        for statement in &mut body {
            if let Statement::Assign {
                target: Expr::Var(name),
                value,
                ..
            } = statement
            {
                if name == "alias" {
                    *value = Expr::Var("phantom".into());
                }
            }
        }
        assert!(!prove(&body, &old).contains("alias"));
    }

    #[test]
    fn every_nested_write_and_signed_witness_are_required() {
        let (body, old) = fixture("if (optInPeriod > 0) { alias = startIdx; }");
        assert!(!prove(&body, &old).contains("alias"));
        let (mut body, old) = fixture("");
        for statement in &mut body {
            if let Statement::Assign {
                target: Expr::Var(name),
                value,
                ..
            } = statement
            {
                if name == "alias" {
                    *value = Expr::Var("optInPeriod".into());
                }
            }
        }
        assert!(!prove(&body, &old).contains("alias"));
        let (mut body, old) = fixture("alias = destination;");
        assert!(prove(&body, &old).contains("alias"));
        body.reverse();
        assert!(prove(&body, &old).contains("alias"));
        let (mut body, old) = fixture("alias = destination;");
        for statement in &mut body {
            if let Statement::Assign {
                target: Expr::Var(name),
                value,
                ..
            } = statement
            {
                if name == "alias" {
                    *value = Expr::Var("destination".into());
                }
            }
        }
        assert!(!prove(&body, &old).contains("alias"));
    }
}
