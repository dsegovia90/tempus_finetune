from random import Random

from tempus import surface
from tempus.assemble import assemble
from tempus.planning import MIX, Plan, make_plans
from tempus.schema import Example, Extraction

PLANS = make_plans(2000, "train", seed=0)


def test_plans_are_reproducible_and_round_trip():
    assert make_plans(50, "train", seed=3) == make_plans(50, "train", seed=3)
    assert [Plan.from_row(p.to_row()) for p in PLANS[:50]] == PLANS[:50]


def test_slice_mix_is_close_to_target():
    for name, share in MIX.items():
        actual = sum(p.slice == name for p in PLANS) / len(PLANS)
        assert abs(actual - share) < 0.04, name


def test_every_planned_span_is_required_text():
    for plan in PLANS:
        required = " ".join(plan.must_include)
        assert all(t.text in required for t in plan.times), plan
        assert all(0 <= t.minute < 1440 for t in plan.times), plan
        assert (plan.slice == "none") == (not plan.times)


def test_fallback_plans_are_complete_examples():
    for plan in (p for p in PLANS if p.slice == "fallback"):
        assert Example(plan.id, plan.slice, plan.text, plan.times).problems() == []


def test_eval_split_uses_held_out_scenarios():
    train = {p.brief for p in PLANS if p.slice == "implicit"}
    held_out = {p.brief for p in make_plans(500, "eval") if p.slice == "implicit"}
    assert not train & held_out


def test_surface_forms():
    rng = Random(0)
    assert surface.bare(900) == "3"
    assert surface.bare(1170) == "7:30"
    assert surface.bare(0) == "12"
    assert surface.with_meridiem(900, rng).lower().replace(" ", "") in ("3pm", "3p.m.")
    assert surface.fallback_minute(900) == 180
    assert surface.fallback_minute(765) == 765
    assert surface.fallback_minute(45) == 765


def test_assemble_gates():
    time = Extraction("7", 1140)
    plans = [
        Plan("a", "implicit", "", "", ("7",), (time,)),
        Plan("b", "implicit", "", "", ("7",), (time,)),
        Plan("c", "implicit", "", "", ("7",), (time,)),
        Plan("d", "fallback", "", "", ("3",), (Extraction("3", 180),), text="at 3?"),
    ]
    messages = {"a": "dinner at 7?", "b": "see you at 7", "c": "dinner at seven"}
    checks = {"a": (time,), "b": (Extraction("7", 420),), "c": (time,)}
    examples, rejects = assemble(plans, messages, checks)
    assert [e.id for e in examples] == ["a", "d"]
    assert {r["id"] for r in rejects} == {"b", "c"}


def test_negatives_split_is_all_no_time_plans():
    plans = make_plans(200, "negatives")
    assert all(p.slice == "none" and not p.times and p.text is None for p in plans)
    assert len({p.brief for p in plans}) > 50


def test_hard_training_split_pairs_traps_with_real_times_and_holds_some_out():
    from tempus.planning import HELD_OUT_TRAPS, TRAIN_TRAPS, TRAPS
    plans = make_plans(600, "train2")
    assert len(HELD_OUT_TRAPS) == 3 and len(TRAIN_TRAPS) == len(TRAPS) - 3
    assert not any(trap in p.brief for p in plans for trap in HELD_OUT_TRAPS)
    with_times = [p for p in plans if p.times]
    assert 0.35 < len(with_times) / len(plans) < 0.6
    assert all("also include" in p.brief for p in with_times)
    assert all(p.text is None for p in plans)
