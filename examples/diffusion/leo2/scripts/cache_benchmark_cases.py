"""Parse versioned Leo2 cache benchmark case matrices."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

METHODS = {
    "off",
    "first_block",
    "taylor",
    "magcache",
    "magcache_calibrate",
    "fastercache_dfr",
    "cfg_cache",
    "fastercache_dfr+cfg_cache",
}
FIELD_SEPARATOR = "\x1f"
TSV_FIELDS = (
    "name",
    "method",
    "cache_threshold",
    "flow_shift_video",
    "guidance_scale",
    "baseline_case",
    "reference_root",
    "taylor_max_extrapolation",
    "magcache_profile",
    "magcache_threshold",
    "magcache_max_skip_steps",
    "magcache_retention_ratio",
    "dfr_start_step",
    "dfr_end_step",
    "dfr_interval",
    "dfr_layers",
    "cfg_start_step",
    "cfg_end_step",
    "cfg_interval",
    "cfg_low_frequency_weight",
    "cfg_high_frequency_weight",
    "cfg_low_frequency_start_step",
    "cfg_low_frequency_end_step",
    "cfg_high_frequency_start_step",
    "cfg_high_frequency_end_step",
    "diff_infer_steps",
)


@dataclass(frozen=True)
class CacheBenchmarkCase:
    """Hold one validated v1- or v2-compatible benchmark case."""

    name: str
    method: str
    cache_threshold: float | None
    flow_shift_video: float
    guidance_scale: float
    baseline_case: str | None = None
    reference_root: str | None = None
    taylor_max_extrapolation: float | None = None
    magcache_profile: str | None = None
    magcache_threshold: float | None = None
    magcache_max_skip_steps: int | None = None
    magcache_retention_ratio: float | None = None
    dfr_start_step: int | None = None
    dfr_end_step: int | None = None
    dfr_interval: int | None = None
    dfr_layers: str | None = None
    cfg_start_step: int | None = None
    cfg_end_step: int | None = None
    cfg_interval: int | None = None
    cfg_low_frequency_weight: float | None = None
    cfg_high_frequency_weight: float | None = None
    cfg_low_frequency_start_step: int | None = None
    cfg_low_frequency_end_step: int | None = None
    cfg_high_frequency_start_step: int | None = None
    cfg_high_frequency_end_step: int | None = None
    diff_infer_steps: int | None = None
    schema_version: int = 2

    @property
    def cache_enabled(self) -> bool:
        """Return whether this case installs an acceleration method."""
        return self.method != "off"

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible case description."""
        return asdict(self)


def _die(message: str) -> None:
    """Raise one consistently typed case-schema error."""
    raise ValueError(message)


def _text(row: dict[str, str | None], key: str) -> str:
    """Read and trim one optional CSV field."""
    return (row.get(key) or "").strip()


def _float(raw: str, *, field: str, context: str, minimum: float = 0.0) -> float:
    """Parse one finite numeric field with a lower bound."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {field} in {context}: {raw!r}") from exc
    if not math.isfinite(value) or value < minimum:
        _die(f"{field} must be finite and >= {minimum:g} in {context}: {raw!r}")
    return value


def _optional_float(
    row: dict[str, str | None], key: str, *, context: str, default: float | None = None
) -> float | None:
    """Parse an optional finite non-negative float."""
    raw = _text(row, key)
    return default if not raw else _float(raw, field=key, context=context)


def _optional_int(row: dict[str, str | None], key: str, *, context: str, default: int | None = None) -> int | None:
    """Parse an optional non-negative integer."""
    raw = _text(row, key)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {key} in {context}: {raw!r}") from exc
    if value < 0:
        _die(f"{key} must be non-negative in {context}: {raw!r}")
    return value


def _threshold(raw: str, *, context: str) -> float | None:
    """Parse the legacy threshold field, including cache-off aliases."""
    if raw.lower() in {"", "off", "none", "disabled"}:
        return None
    return _float(raw, field="cache_threshold", context=context)


def _safe_case_name(raw: str, *, field: str, context: str) -> str:
    """Validate a case-directory basename."""
    if not raw or raw in {".", ".."} or Path(raw).name != raw:
        _die(f"Unsafe or empty {field} in {context}: {raw!r}")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in raw):
        _die(f"Invalid character in {field} in {context}: {raw!r}")
    return raw


def _optional_path(raw: str, *, source: Path, field: str, context: str) -> str | None:
    """Resolve an optional path relative to the case CSV."""
    if not raw:
        return None
    if any(character in raw for character in "\t\r\n"):
        _die(f"{field} contains a control character in {context}")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return str(path.resolve())


def _reject_fields(row: dict[str, str | None], fields: tuple[str, ...], *, method: str, context: str) -> None:
    """Reject method-specific values that the selected method cannot consume."""
    unexpected = [field for field in fields if _text(row, field)]
    if unexpected:
        _die(f"Method {method!r} does not accept {', '.join(unexpected)} in {context}")


def _parse_v1(row: dict[str, str | None], *, context: str) -> CacheBenchmarkCase:
    """Map the original three-column matrix onto schema v2 defaults."""
    threshold = _threshold(_text(row, "cache_threshold"), context=context)
    return CacheBenchmarkCase(
        name=_safe_case_name(_text(row, "name"), field="name", context=context),
        method="off" if threshold is None else "first_block",
        cache_threshold=threshold,
        flow_shift_video=_float(_text(row, "flow_shift_video"), field="flow_shift_video", context=context),
        guidance_scale=1.0,
        schema_version=1,
    )


def _parse_v2(row: dict[str, str | None], *, source: Path, context: str) -> CacheBenchmarkCase:
    """Parse one explicit method-aware schema-v2 row."""
    method = _text(row, "method").lower()
    if method not in METHODS:
        _die(f"Unsupported method in {context}: {method!r}; expected one of {sorted(METHODS)}")
    name = _safe_case_name(_text(row, "name"), field="name", context=context)
    baseline_case = _safe_case_name(_text(row, "baseline_case"), field="baseline_case", context=context)
    threshold = _threshold(_text(row, "cache_threshold"), context=context)
    shift = _float(_text(row, "flow_shift_video"), field="flow_shift_video", context=context)
    guidance = _float(_text(row, "guidance_scale"), field="guidance_scale", context=context)
    reference_root = _optional_path(
        _text(row, "reference_root"), source=source, field="reference_root", context=context
    )
    diff_infer_steps = _optional_int(row, "diff_infer_steps", context=context)
    if diff_infer_steps is None:
        matches = re.findall(r"(?:^|_)s(6|28|50)(?:_|$)", name)
        if len(set(matches)) == 1:
            diff_infer_steps = int(matches[0])
    if diff_infer_steps == 0:
        _die(f"diff_infer_steps must be positive in {context}")

    common = dict(
        name=name,
        method=method,
        cache_threshold=threshold,
        flow_shift_video=shift,
        guidance_scale=guidance,
        baseline_case=baseline_case,
        reference_root=reference_root,
        diff_infer_steps=diff_infer_steps,
    )
    taylor_fields = ("taylor_max_extrapolation",)
    magcache_fields = (
        "magcache_profile",
        "magcache_threshold",
        "magcache_max_skip_steps",
        "magcache_retention_ratio",
    )
    dfr_fields = ("dfr_start_step", "dfr_end_step", "dfr_interval", "dfr_layers")
    cfg_fields = (
        "cfg_start_step",
        "cfg_end_step",
        "cfg_interval",
        "cfg_low_frequency_weight",
        "cfg_high_frequency_weight",
        "cfg_low_frequency_start_step",
        "cfg_low_frequency_end_step",
        "cfg_high_frequency_start_step",
        "cfg_high_frequency_end_step",
    )

    if method == "off":
        if threshold is not None:
            _die(f"Method 'off' requires an empty/off cache_threshold in {context}")
        _reject_fields(row, taylor_fields + magcache_fields + dfr_fields + cfg_fields, method=method, context=context)
        return CacheBenchmarkCase(**common)
    if method == "first_block":
        if threshold is None:
            _die(f"Method 'first_block' requires cache_threshold in {context}")
        _reject_fields(row, taylor_fields + magcache_fields + dfr_fields + cfg_fields, method=method, context=context)
        return CacheBenchmarkCase(**common)
    if method == "taylor":
        if threshold is None:
            _die(f"Method 'taylor' requires cache_threshold in {context}")
        _reject_fields(row, magcache_fields + dfr_fields + cfg_fields, method=method, context=context)
        return CacheBenchmarkCase(
            **common,
            taylor_max_extrapolation=_optional_float(row, "taylor_max_extrapolation", context=context, default=1.0),
        )
    if method in {"magcache", "magcache_calibrate"}:
        if threshold is not None:
            _die(f"Method {method!r} uses magcache_threshold, not cache_threshold, in {context}")
        _reject_fields(row, taylor_fields + dfr_fields + cfg_fields, method=method, context=context)
        magcache_threshold = _optional_float(row, "magcache_threshold", context=context)
        if magcache_threshold is None:
            _die(f"Method {method!r} requires magcache_threshold in {context}")
        profile = _optional_path(
            _text(row, "magcache_profile"), source=source, field="magcache_profile", context=context
        )
        if method == "magcache" and profile is None:
            _die(f"Method 'magcache' requires magcache_profile in {context}")
        return CacheBenchmarkCase(
            **common,
            magcache_profile=profile,
            magcache_threshold=magcache_threshold,
            magcache_max_skip_steps=_optional_int(row, "magcache_max_skip_steps", context=context, default=4),
            magcache_retention_ratio=_optional_float(row, "magcache_retention_ratio", context=context, default=0.2),
        )

    if threshold is not None:
        _die(f"Method {method!r} does not use cache_threshold in {context}")
    _reject_fields(row, taylor_fields + magcache_fields, method=method, context=context)

    values: dict[str, Any] = {}
    if method in {"fastercache_dfr", "fastercache_dfr+cfg_cache"}:
        start = _optional_int(row, "dfr_start_step", context=context, default=4)
        end = _optional_int(row, "dfr_end_step", context=context, default=46)
        interval = _optional_int(row, "dfr_interval", context=context, default=2)
        if interval == 0:
            _die(f"dfr_interval must be positive in {context}")
        if start is not None and end is not None and start >= end:
            _die(f"dfr_start_step must be less than dfr_end_step in {context}")
        values.update(
            dfr_start_step=start,
            dfr_end_step=end,
            dfr_interval=interval,
            dfr_layers=(
                _text(row, "dfr_layers")
                or ("24-47" if method == "fastercache_dfr+cfg_cache" else None)
            ),
        )
    else:
        _reject_fields(row, dfr_fields, method=method, context=context)

    if method in {"cfg_cache", "fastercache_dfr+cfg_cache"}:
        cfg_start = _optional_int(row, "cfg_start_step", context=context, default=1)
        cfg_end = _optional_int(row, "cfg_end_step", context=context)
        cfg_interval = _optional_int(row, "cfg_interval", context=context, default=5)
        if cfg_end is None:
            _die(f"Method {method!r} requires cfg_end_step in {context}")
        if cfg_interval == 0:
            _die(f"cfg_interval must be positive in {context}")
        if cfg_start is not None and cfg_start >= cfg_end:
            _die(f"cfg_start_step must be less than cfg_end_step in {context}")
        low_start = _optional_int(row, "cfg_low_frequency_start_step", context=context, default=cfg_start)
        low_end = _optional_int(row, "cfg_low_frequency_end_step", context=context, default=cfg_end)
        high_start = _optional_int(row, "cfg_high_frequency_start_step", context=context, default=cfg_start)
        high_end = _optional_int(row, "cfg_high_frequency_end_step", context=context, default=cfg_end)
        if low_start is None or low_end is None or low_start >= low_end:
            _die(f"invalid low-frequency CFG step range in {context}")
        if high_start is None or high_end is None or high_start >= high_end:
            _die(f"invalid high-frequency CFG step range in {context}")
        values.update(
            cfg_start_step=cfg_start,
            cfg_end_step=cfg_end,
            cfg_interval=cfg_interval,
            cfg_low_frequency_weight=_optional_float(
                row, "cfg_low_frequency_weight", context=context, default=1.1
            ),
            cfg_high_frequency_weight=_optional_float(
                row, "cfg_high_frequency_weight", context=context, default=1.1
            ),
            cfg_low_frequency_start_step=low_start,
            cfg_low_frequency_end_step=low_end,
            cfg_high_frequency_start_step=high_start,
            cfg_high_frequency_end_step=high_end,
        )
    else:
        _reject_fields(row, cfg_fields, method=method, context=context)
    return CacheBenchmarkCase(**common, **values)


def load_cases(path: Path) -> tuple[CacheBenchmarkCase, ...]:
    """Load and validate a v1 or v2 benchmark matrix."""
    path = path.resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        v1 = {"name", "cache_threshold", "flow_shift_video"}
        v2 = v1 | {"method", "guidance_scale", "baseline_case", "reference_root"}
        if not v1.issubset(fields):
            _die(f"Cases CSV must contain {sorted(v1)}: {path}")
        if fields.intersection(v2 - v1) and not v2.issubset(fields):
            _die(f"Partial schema v2 header in {path}; required fields are {sorted(v2)}")
        cases = []
        for line_number, row in enumerate(reader, 2):
            context = f"{path}:{line_number}"
            cases.append(
                _parse_v2(row, source=path, context=context) if v2.issubset(fields) else _parse_v1(row, context=context)
            )
    if not cases:
        _die(f"Cases CSV is empty: {path}")
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        _die(f"Duplicate case name in {path}")
    return tuple(cases)


def _format_record(case: CacheBenchmarkCase) -> str:
    """Serialize one case for the Bash launcher without losing empty fields."""
    values = case.as_dict()
    fields = []
    for name in TSV_FIELDS:
        value = values[name]
        text = "" if value is None else str(value)
        if any(character in text for character in f"{FIELD_SEPARATOR}\r\n"):
            _die(f"Case {case.name!r} field {name!r} contains a record delimiter")
        fields.append(text)
    return FIELD_SEPARATOR.join(fields)


def main() -> None:
    """Emit normalized rows for the Bash benchmark launcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-records", type=Path, required=True)
    args = parser.parse_args()
    for case in load_cases(args.emit_records):
        print(_format_record(case))


if __name__ == "__main__":
    main()
