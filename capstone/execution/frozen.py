"""Build broker-shaped instructions from immutable inputs, without using a broker."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from capstone.execution.bundle import load_bundle
from capstone.execution.contracts import (
    ExecutionContractBlocked,
    execution_contract,
    program_positions,
    validate_program_instruments,
    verify_execution_contract,
    verify_plan_specifications,
)
from capstone.execution.daily_planning import lifecycle_plan, plan_completed_session
from capstone.execution.evidence import (
    FINAL,
    FINAL_PENDING,
    INCUBATION_ROUTE,
    INCUBATION_UNQUALIFIED,
    admission,
    incubation_eligible,
    plan_admission,
)
from capstone.execution.planning import portfolio_risk
from capstone.factors.reference import sessions
from capstone.strategy.portfolio import allocation, portfolio_target
from capstone.strategy.research import canonical, digest
from capstone.strategy.saved_portfolio import MODE, SavedProgramPortfolioSpec
from capstone.strategy.schedule_dispatch import decision_due

CAP_EXCEEDED = "incubation_cap_exceeded"
ROUTE_KEY = "transmission_route"
ROUTE_KEYS = {"route", "incubation_cap", "account"}


class IncubationRouteRefused(ExecutionContractBlocked):
    """A plan asked for (or claims) the incubation route but does not qualify."""


def registered_incubation_cap():
    """The registered incubation cap: `RiskLimits.incubation_cap`, a fraction of equity.

    The runner is imported lazily (it imports half the package). Where it is not available
    the cap is unknown, so an incubation route is refused rather than guessed at.
    """
    try:
        from capstone.execution.runner import RiskLimits
    except ImportError as error:
        raise IncubationRouteRefused(
            INCUBATION_UNQUALIFIED, "the registered incubation cap is unavailable"
        ) from error
    return RiskLimits.incubation_cap


def _route_record(route):
    """The plan's `transmission_route` for a requested route, or the typed refusal."""
    if not isinstance(route, dict) or set(route) != {"route", "incubation_cap"}:
        raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, "route needs route, incubation_cap")
    cap = route["incubation_cap"]
    if route["route"] != INCUBATION_ROUTE:
        raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, f"unknown route {route['route']!r}")
    if isinstance(cap, bool) or not isinstance(cap, int | float) or not math.isfinite(cap):
        raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, "incubation_cap must be a number")
    if not 0 < cap <= registered_incubation_cap():
        raise IncubationRouteRefused(
            INCUBATION_UNQUALIFIED,
            f"incubation_cap {cap} is outside (0, {registered_incubation_cap()}] (registered)",
        )
    return {"route": INCUBATION_ROUTE, "incubation_cap": float(cap), "account": "paper"}


def incubation_exposure(content):
    """(account equity, capital, projected gross notional) at the plan's reference prices."""
    prices = {int(n): float(p) for n, p in content["reference_prices"].items()}
    held = {int(n): int(q) for n, q in content["account_positions"].items()}
    projected = {int(n): int(q) for n, q in content["projected_shares"].items()}
    equity = float(content["account_cash"]) + sum(q * prices[n] for n, q in held.items())
    gross = sum(abs(q) * prices[n] for n, q in projected.items())
    return equity, float(content["capital"]), gross


def incubation_cap_exceeded(content):
    """True unless capital and the projected book are within `incubation_cap` x equity."""
    equity, capital, gross = incubation_exposure(content)
    limit = content[ROUTE_KEY]["incubation_cap"] * equity
    return not (equity > 0 and capital <= limit + 1e-6 and gross <= limit + 1e-6)


THRESHOLD_FILE = "registered-thresholds.json"  # `registered_thresholds.BUNDLE_FILE`


def _bundle_thresholds(bundle):
    """{strategy id: threshold file} a bundle recorded; {} for a bundle that recorded none.

    A bundle that carries a record needs `registered_thresholds` to verify it; where that
    module is not available the ImportError propagates and no plan is built (fail closed).
    """
    if not (Path(bundle) / THRESHOLD_FILE).is_file():
        return {}
    from capstone.pipeline.registered_thresholds import bundle_records  # import cycle

    return bundle_records(bundle)


def account_hash(account):
    content = {
        "positions": {str(n): int(q) for n, q in sorted(account["positions"].items()) if q},
        "open_orders": sorted(account.get("open_orders", []), key=canonical),
        "capital": account["capital"],
        "cash": account["cash"],
    }
    if account.get("receivables"):
        content["receivables"] = sorted(account["receivables"], key=lambda row: row["event_id"])
    if "corporate_events" in account:
        content["corporate_events"] = account["corporate_events"]
    return digest(content)


def identify_plan(content):
    """Derive new-plan identities without mutating content or granting admission."""
    if type(content.get("schema_version")) is not int or content["schema_version"] != 3:
        raise ExecutionContractBlocked("unsupported_plan_identity_schema")
    if (
        type(content.get("transmittable")) is not bool
        or not isinstance(content.get("orders"), list)
        or any(not isinstance(order, dict) for order in content["orders"])
    ):
        raise ExecutionContractBlocked("plan_identity_content_incompatible")
    base = {
        k: v for k, v in content.items() if k not in {"document_hash", "plan_id", "transmittable"}
    }
    base["orders"] = [
        {k: v for k, v in order.items() if k != "instruction_id"} for order in content["orders"]
    ]
    identifier = digest(base)
    identified = base | {
        "plan_id": identifier,
        "transmittable": content["transmittable"],
        "orders": [
            order | {"instruction_id": digest({"plan": identifier, "sequence": sequence})}
            for sequence, order in enumerate(base["orders"])
        ],
    }
    return identified | {"document_hash": digest(identified)}


def build_plan(bundle, *, lifecycle=None, route=None):
    """The frozen plan of a bundle. `lifecycle` ({strategy id: "retired" | "suspended"},
    from the registry; `lifecycle plan --close`) plans through `lifecycle_plan` and is
    recorded as `lifecycle_states`; without it the plan is exactly as before.

    `route` (`{"route": "incubation-only", "incubation_cap": c}`) asks for the
    incubation transmission route: every strategy's re-verified evidence must be
    `incubation_eligible` (else `IncubationRouteRefused`); the sealed-holdout blocker is
    then not a validation blocker, the plan records `transmission_route`, and capital or a
    book above `c` x account equity is the planning blocker `incubation_cap_exceeded`.
    `formal_admission` stays false. Without `route` the plan is exactly as before."""
    data, panel, specs, returns = load_bundle(bundle)
    contract = execution_contract(
        specs,
        purpose="frozen",
        version=(data.get("execution_contract") or {}).get("version", "fixed-share-next-open-v2"),
    )
    validate_program_instruments(specs, panel)
    saved = (
        specs[0] if len(specs) == 1 and isinstance(specs[0], SavedProgramPortfolioSpec) else None
    )
    if (
        contract["route"] == "program-v1"
        and data["allocation_mode"] != "equal_family"
        and saved is None
    ):
        raise ExecutionContractBlocked("variable_program_allocations_unqualified")
    saved_contract = data.get("execution_contract")
    if saved_contract != contract and (
        saved_contract is not None or contract["route"] == "program-v1"
    ):
        raise ExecutionContractBlocked("frozen_execution_contract_incompatible")
    at = pd.Timestamp(data["as_of"])
    calendar = sessions(panel.calendar, at)
    execution = calendar[calendar.get_loc(at) + 1]
    if not specs or len({s.side for s in specs}) != 1:
        raise ValueError("exactly one nonempty portfolio profile is required")
    account = data["account"]
    capital = float(account["capital"])
    if not math.isfinite(capital) or capital <= 0 or not math.isfinite(float(account["cash"])):
        raise ValueError("account capital and cash must be finite")
    positions = (
        program_positions(account["positions"])
        if contract["route"] == "program-v1"
        else {int(n): q for n, q in account["positions"].items()}
    )
    if any(isinstance(q, bool) or not isinstance(q, int) for q in positions.values()):
        raise ValueError("positions must be whole shares")
    positions = {n: q for n, q in positions.items() if q}
    receivable_value = sum(row["amount"] for row in account.get("receivables", []))
    if panel.universe.index[-1] != at:
        raise ValueError("stale market snapshot")
    if saved is not None:
        if data["allocation_mode"] != MODE:
            raise ExecutionContractBlocked("saved_allocation_mode_required")
        weights, allocated, details = saved.plan_inputs(panel, at)
    else:
        allocated = allocation(specs, returns, at, mode=data["allocation_mode"])
        weights, details = portfolio_target(specs, panel, at, allocated)
    prices = {int(n): float(v) for n, v in data["prices"].items()}
    scheduled = [decision_due(spec, panel, at) for spec in specs]
    rebalance = any(scheduled)
    if lifecycle:
        weights, sized = lifecycle_plan(
            specs,
            panel,
            at,
            positions,
            prices,
            capital - max(receivable_value, 0),
            weights=weights,
            attribution=details["attribution"]["strategies"],
            states=lifecycle,
            fee_bps=data["execution_cost_bps"] or 0,
        )
    else:
        sized = plan_completed_session(
            specs,
            panel,
            at,
            positions,
            prices,
            capital - max(receivable_value, 0),
            weights=weights,
            fee_bps=data["execution_cost_bps"] or 0,
        )
    side = specs[0].side
    risk = portfolio_risk(
        sized["projected"],
        prices,
        capital,
        side=side,
        orders=sized["orders"],
        **data["limits"],
    )
    blockers = list(risk["blockers"])
    if data["execution_cost_bps"] is None:
        blockers.append("execution_cost_assumption_missing")
    if account.get("open_orders"):
        blockers.append("outstanding_orders_require_reconciliation")
    cash_after = (
        float(account["cash"])
        - sum(o["shares"] * prices[o["permno"]] for o in sized["orders"])
        - sized["estimated_fees"]
    )
    if side == "long_only" and cash_after < -1e-8:
        blockers.append("rounded_cash_deficit")
    required = set(sized["targets"]) | set(positions)
    instruments = data["instruments"]
    for n in required:
        item = instruments.get(str(n))
        if (
            not item
            or not item.get("symbol")
            or item.get("sec_type") != "STK"
            or item.get("currency") != "USD"
        ):
            raise ValueError(f"missing or unsupported instrument mapping for {n}")
    recorded = _bundle_thresholds(bundle)  # the bundle's thresholds, never the active ledger's
    validation = {
        s.id: plan_admission(s, data["evidence"].get(s.id), recorded.get(s.id)) for s in specs
    }
    admission_blockers = sorted(
        {reason for evidence in validation.values() for reason in evidence["blockers"]}
    )
    transmission_route = None
    if route is not None:
        transmission_route = _route_record(route)
        if contract["route"] != "program-v1":
            raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, "only program plans")
        unqualified = sorted(sid for sid, v in validation.items() if not incubation_eligible(v))
        if unqualified:
            detail = "no search-stage gate report re-verifies for " + ", ".join(unqualified)
            raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, detail)
        admission_blockers = [b for b in admission_blockers if b != FINAL_PENDING]
    qualification = []
    if account.get("receivables"):
        qualification.append("broker_receivable_recovery_unqualified")
    if data["release"]["dirty"] or not data["release"]["deployment_tag"]:
        qualification.append("reviewed_deployment_release_required")
    for flag in (
        "paper_account_verified",
        "permissions_verified",
        "funds_verified",
        "data_verified",
    ):
        if account.get(flag) is not True:
            qualification.append(flag)
    if side == "long_short" and account.get("shorting_verified") is not True:
        qualification.append("shorting_verified")
    if any(
        not instruments[str(n)].get("con_id") or not instruments[str(n)].get("qualified")
        for n in required
    ):
        qualification.append("contracts_unqualified")
    orders = []
    for original in sized["orders"]:
        n, q = original["permno"], original["shares"]
        held, target = positions.get(n, 0), sized["projected"][n]
        legs = (
            [(-held, "reduce"), (target, "open")]
            if held * target < 0
            else [(q, "reduce" if abs(target) < abs(held) else "open")]
        )
        for quantity, purpose in legs:
            orders.append(
                {
                    "permno": n,
                    "instrument": instruments[str(n)],
                    "action": "BUY" if quantity > 0 else "SELL",
                    "quantity": abs(quantity),
                    "signed_shares": quantity,
                    "purpose": purpose,
                    "reference_price": prices[n],
                    "order_type": "MKT",
                    "tif": "DAY",
                    "outside_rth": False,
                }
            )
    orders.sort(key=lambda o: (o["purpose"] != "reduce", o["permno"]))
    content = {
        "runtime": data["runtime"],
        "release": data["release"],
        "schema_version": 3,
        "execution_contract": contract,
        "planning_intent": sized["intent"],
        "formal_admission": all(
            v.get("formal_admission", v["admitted"]) for v in validation.values()
        ),
        "reference_prices": data["prices"],
        "instruments": instruments,
        "input_hash": data["input_hash"],
        "profile": side,
        "decision_session": data["as_of"],
        "execution_session": str(execution.date()),
        "strategy_ids": sorted(s.id for s in specs),
        "specifications": {s.id: s.to_toml() for s in sorted(specs, key=lambda s: s.id)},
        "allocation": allocated,
        "construction_verified": True,
        "validation": validation,
        "account_hash": account_hash(account),
        "account_cash": float(account["cash"]),
        "account_receivables": account.get("receivables", []),
        "capital": capital,
        "account_positions": {str(n): int(q) for n, q in sorted(positions.items()) if q},
        "target_weights": {str(n): round(float(w), 12) for n, w in weights.items()},
        "target_shares": {str(n): q for n, q in sized["targets"].items()},
        "projected_shares": {str(n): q for n, q in sized["projected"].items()},
        "rounding_and_omissions": sized["omitted"],
        "projected_cash": cash_after,
        "sizing_capital": sized["sizing_capital"],
        "execution_cost_bps": data["execution_cost_bps"],
        "estimated_fees": sized["estimated_fees"],
        "attribution": details["attribution"],
        "risk": risk,
        "limits": data["limits"],
        "planning_blockers": sorted(set(blockers)),
        "validation_blockers": admission_blockers,
        "broker_blockers": sorted(qualification),
        "rebalance": rebalance,
        "orders": orders,
    }
    if "formation_eligibility" in sized:
        content["formation_eligibility"] = sized["formation_eligibility"]
    if "corporate_events" in account:
        content["account_corporate_events"] = account["corporate_events"]
    if lifecycle:
        content["lifecycle_states"] = dict(sorted(lifecycle.items()))
    if transmission_route is not None:
        content[ROUTE_KEY] = transmission_route
        if incubation_cap_exceeded(content):
            blockers.append(CAP_EXCEEDED)
            content["planning_blockers"] = sorted(set(blockers))
    if execution_contract(specs, purpose="frozen", version=contract["version"]) != contract:
        raise ExecutionContractBlocked("execution_source_changed_during_plan")
    content["transmittable"] = not (blockers or admission_blockers or qualification)
    return identify_plan(content)


def verify_plan(plan, *, inspection=False):
    if digest({k: v for k, v in plan.items() if k != "document_hash"}) != plan["document_hash"]:
        raise ValueError("order plan changed")
    if plan["schema_version"] == 2:
        if inspection:
            return plan
        raise ExecutionContractBlocked("legacy_plan_requires_explicit_inspection")
    if plan["schema_version"] != 3:
        raise ValueError("unsupported order plan schema")
    if "execution_contract" not in plan:
        raise ExecutionContractBlocked("execution_contract_missing")
    verify_execution_contract(plan["execution_contract"], inspection=inspection)
    if plan.get("profile") != plan["execution_contract"]["profile"]:
        raise ExecutionContractBlocked("plan_execution_profile_incompatible")
    specs = () if inspection else verify_plan_specifications(plan)
    if not inspection and ROUTE_KEY in plan:
        _verify_incubation(plan, specs)
    elif not inspection and plan["execution_contract"]["route"] == "program-v1":
        validation = plan.get("validation")
        pending = {"program_financial_admission_pending", FINAL_PENDING}
        if (
            not isinstance(plan.get("validation_blockers"), list)
            or not isinstance(validation, dict)
            or not validation
            or set(validation) != set(plan.get("strategy_ids", []))
        ):
            raise ExecutionContractBlocked("program_financial_admission_pending")
        if plan.get("formal_admission") is not False or plan.get("transmittable") is not False:
            # Only a final gate report under a spent, authorized holdout protocol, re-verified
            # now, admits; a self-consistent plan document cannot claim it.
            if plan.get("formal_admission") is not True or any(
                not isinstance(validation[s.id], dict)
                or validation[s.id].get("validation_status") != FINAL
                or validation[s.id] != admission(s, validation[s.id].get("report_path"))
                for s in specs
            ):
                raise ExecutionContractBlocked("program_financial_admission_pending")
        elif pending.isdisjoint(plan["validation_blockers"]) or any(
            not isinstance(value, dict)
            or value.get("admitted") is not False
            or value.get("formal_admission") is not False
            or not isinstance(value.get("blockers"), list)
            or pending.isdisjoint(value["blockers"])
            for value in validation.values()
        ):
            raise ExecutionContractBlocked("program_financial_admission_pending")
    expected = identify_plan(plan)
    if plan.get("plan_id") != expected["plan_id"]:
        raise ExecutionContractBlocked("plan_identity_incompatible")
    if any(
        order.get("instruction_id") != identified["instruction_id"]
        for order, identified in zip(plan["orders"], expected["orders"], strict=True)
    ):
        raise ExecutionContractBlocked("instruction_identity_incompatible")
    if not inspection:
        groups = [
            plan.get(key) for key in ("planning_blockers", "validation_blockers", "broker_blockers")
        ]
        if any(not isinstance(group, list) for group in groups) or plan["transmittable"] is not (
            not any(groups)
        ):
            raise ExecutionContractBlocked("plan_transmission_status_incompatible")
    return plan


def _verify_incubation(plan, specs):
    """Re-checked whenever the plan is read or sent: the route record, a program
    plan, no formal admission, and each strategy's search-stage report re-verified now
    now with the sealed holdout as its only blocker; the cap blocker matches the plan."""
    route = plan[ROUTE_KEY]
    validation = plan.get("validation")
    try:
        if not isinstance(route, dict) or set(route) != ROUTE_KEYS or route["account"] != "paper":
            raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, "malformed transmission_route")
        requested = {"route": route["route"], "incubation_cap": route["incubation_cap"]}
        if _route_record(requested) != route:
            raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, "transmission_route changed")
        if plan["execution_contract"]["route"] != "program-v1":
            raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, "only program plans")
        if (
            plan.get("formal_admission") is not False
            or plan.get("validation_blockers") != []
            or not isinstance(validation, dict)
            or set(validation) != set(plan.get("strategy_ids", []))
            or set(validation) != {s.id for s in specs}
        ):
            raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, "admission fields changed")
        for spec in specs:
            value = validation[spec.id]
            if not incubation_eligible(value) or value != admission(spec, value["report_path"]):
                raise IncubationRouteRefused(
                    INCUBATION_UNQUALIFIED, f"{spec.id}: its gate report does not re-verify"
                )
        if incubation_cap_exceeded(plan) != (CAP_EXCEEDED in plan.get("planning_blockers", [])):
            raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, "incubation cap check changed")
    except (KeyError, TypeError, AttributeError) as error:
        raise IncubationRouteRefused(INCUBATION_UNQUALIFIED, repr(error)) from error


def write_plan(plan, path):
    verify_plan(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != canonical(plan) + "\n":
        raise FileExistsError("refusing to replace a different frozen plan")
    path.write_text(canonical(plan) + "\n", encoding="utf-8", newline="\n")
    lines = [
        f"# Order plan {plan['plan_id']}",
        f"Profile: {plan['profile']}",
        f"Decision: {plan['decision_session']}; execution: {plan['execution_session']}",
        f"Input hash: {plan['input_hash']}",
        f"Document hash: {plan['document_hash']}",
        f"Strategies: {', '.join(plan['strategy_ids'])}",
        f"Allocation: {plan['allocation']['mode']} / {plan['allocation']['id']}",
        f"Construction verified: {plan['construction_verified']}",
        f"Execution contract: {plan.get('execution_contract', {}).get('id', 'legacy-unversioned')}",
        f"Planning intent: {plan.get('planning_intent', 'legacy-unspecified')}",
        f"Validation admitted: {not plan['validation_blockers']}",
        f"Broker qualified: {not plan['broker_blockers']}",
        f"Rebalance scheduled: {plan['rebalance']}",
        f"Transmittable: {plan['transmittable']}",
        f"Projected cash: ${plan['projected_cash']:.2f}; "
        f"estimated trading costs: ${plan['estimated_fees']:.2f}",
        f"Rounded gross: {plan['risk']['gross']:.6f}; net: {plan['risk']['net']:.6f}; "
        f"largest name: {plan['risk']['largest_weight']:.6f}",
        "",
        "| Instrument / permno | Target weight | Current shares | Target shares | "
        "After orders | Net trade |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(plan["projected_shares"], key=int):
        held = plan["account_positions"].get(name, 0)
        projected = plan["projected_shares"][name]
        symbol = plan["instruments"][name]["symbol"]
        lines.append(
            f"| {symbol} / {name} | {plan['target_weights'].get(name, 0):.8f} | {held} | "
            f"{plan['target_shares'][name]} | {projected} | {projected - held:+d} |"
        )
    lines += [
        "",
        "| Instruction | Instrument / contract | Action | Shares | Type |",
        "|---|---|---|---:|---|",
    ]
    lines += [
        f"| {o['instruction_id']} | {o['instrument']['symbol']} / "
        f"{o['instrument'].get('con_id', 'unqualified')} | {o['action']} | {o['quantity']} | "
        "MKT / DAY / regular session |"
        for o in plan["orders"]
    ]
    lines += [
        "",
        f"Rounding and minimum-notional omissions: {canonical(plan['rounding_and_omissions'])}",
    ]
    for kind in ("planning_blockers", "validation_blockers", "broker_blockers"):
        lines.extend(["", f"{kind}: {', '.join(plan[kind]) or 'none'}"])
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_plan(path, *, inspection=False):
    """Historical inspection never grants execution under the current contract."""
    return verify_plan(json.loads(Path(path).read_text(encoding="utf-8")), inspection=inspection)
