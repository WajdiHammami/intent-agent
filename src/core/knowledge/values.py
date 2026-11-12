"""What values a slot accepts, and coercion of raw (LLM-parsed) values into them."""

from __future__ import annotations

from typing import Any

from models.ontology import SlotDefinition

_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


class InvalidValue(ValueError):
    """A value the slot does not accept."""


def allowed_values(slot: SlotDefinition) -> list[Any] | None:
    """The closed set of values for enum and bool slots; None for free-form slots."""
    if slot.dtype == "enum":
        return list(slot.enum_values)
    if slot.dtype == "bool":
        return [True, False]
    return None


def coerce_value(slot: SlotDefinition, raw: Any) -> Any:
    """Return ``raw`` as a valid value of ``slot`` or raise InvalidValue."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise InvalidValue(f"no value for {slot.field}")
    try:
        if slot.dtype == "enum":
            for option in slot.enum_values:
                if str(raw).strip().lower() == option.lower():
                    return option
            raise InvalidValue(f"{raw!r} is not one of {slot.enum_values}")
        if slot.dtype == "bool":
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            if text in _TRUE | _FALSE:
                return text in _TRUE
            raise InvalidValue(f"{raw!r} is not a yes/no value")
        if slot.dtype == "float":
            return float(raw)
        if slot.dtype == "int":
            return int(float(raw))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InvalidValue):
            raise
        raise InvalidValue(f"{raw!r} is not a valid {slot.dtype} for {slot.field}") from exc
    return raw
