"""Join plans, written messages and blind checks into validated examples."""

from tempus.planning import Plan
from tempus.schema import Example, Extraction


def assemble(
    plans: list[Plan],
    messages: dict[str, str],
    checks: dict[str, tuple[Extraction, ...]],
    held_out: tuple[str, ...] = (),
) -> tuple[list[Example], list[dict]]:
    """Return the examples that pass every gate, and a reject row for each that fails.

    `held_out` is the text of messages in other sets (the eval sets, when assembling
    training data); a message that repeats one of them is rejected as a duplicate.
    """
    examples, rejects, seen = [], [], {_normalize(text) for text in held_out}
    for plan in plans:
        text = plan.text if plan.text is not None else messages.get(plan.id)
        reasons = _reasons(plan, text, checks, seen)
        if reasons:
            rejects.append({"id": plan.id, "slice": plan.slice, "text": text, "reasons": reasons})
        else:
            seen.add(_normalize(text))
            examples.append(Example(plan.id, plan.slice, text, plan.times))
    return examples, rejects


def _reasons(plan: Plan, text: str | None, checks: dict, seen: set[str]) -> list[str]:
    if text is None:
        return ["no message written"]
    reasons = Example(plan.id, plan.slice, text, plan.times).problems()
    reasons += [f"missing phrase {p!r}" for p in plan.must_include if p not in text]
    if _normalize(text) in seen:
        reasons.append("duplicate message")
    if plan.text is None:  # written by an agent, so it needs the blind check
        if plan.id not in checks:
            reasons.append("not checked")
        elif checks[plan.id] != plan.times:
            reasons.append(f"checker read it as {[(t.text, t.minute) for t in checks[plan.id]]}")
    return reasons


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
