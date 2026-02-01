"""Repository-wide constants for sleep staging."""

from __future__ import annotations

from typing import Dict

# Canonical mapping between integer labels and state names
STATE_NAMES: Dict[int, str] = {
    0: "NREM",
    1: "REM",
    2: "Wake",
}

NUM_STATES: int = len(STATE_NAMES)

# Reverse lookup helpers ----------------------------------------------------
STATE_ID_BY_NAME: Dict[str, int] = {name: idx for idx, name in STATE_NAMES.items()}

# Allow case-insensitive lookups (e.g. "rem", "wake")
STATE_ID_BY_ALIAS: Dict[str, int] = {
    alias: idx
    for name, idx in STATE_ID_BY_NAME.items()
    for alias in {name.lower(), name.upper(), name.title()}
}


def resolve_state_identifier(identifier: str | int) -> int:
    """Resolve a string/int identifier to the canonical integer state id."""
    if isinstance(identifier, int):
        if identifier not in STATE_NAMES:
            raise KeyError(f"Unknown sleep state id: {identifier}")
        return identifier
    key = identifier.strip()
    if key.isdigit():
        as_int = int(key)
        if as_int in STATE_NAMES:
            return as_int
    try:
        return STATE_ID_BY_ALIAS[key]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise KeyError(f"Unknown sleep state identifier: {identifier}") from exc

