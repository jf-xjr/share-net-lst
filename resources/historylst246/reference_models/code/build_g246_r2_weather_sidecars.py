#!/usr/bin/env python3
"""Build target-free NASA POWER DAILY weather sidecars for public G246 views.

The builder materialises only ``fit`` and ``validation`` scene identities from
``g246_data.load_splits``.  Scene NPZ files are opened as ZIP containers and
only their small ``metadata.npy`` member is read to recover the registered city
longitude/latitude.  Target, valid, eligible, and other scientific arrays are
never read.  The locked-test descriptor is never resolved by this module.

Each city is queried once for the date span covering its public scenes.  Raw
POWER responses are reduced immediately to the requested scene dates, a small
set of provider metadata, and a canonical response digest.  City snapshots and
scene sidecars are immutable resume assets.  Normalization is fitted from fit
scenes only with equal-region, equal-city, equal-scene moments.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import requests

try:
    from . import g246_data
except ImportError:  # pragma: no cover - direct script entry point
    import g246_data


WORKSPACE = Path(__file__).resolve().parents[1]
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
DEFAULT_SPLIT_RECEIPT = g246_data.DEFAULT_SPLIT_RECEIPT
DEFAULT_OUTPUT = WORKSPACE / "artifacts/g246_r2_weather_sidecars_v1"
PARAMETERS = (
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS2M",
    "ALLSKY_SFC_SW_DWN",
    "PRECTOTCORR",
    "GWETTOP",
)
REGIONS = ("us", "china", "europe")
ROLES = ("fit", "validation")
TIME_STANDARD = "UTC"
COMMUNITY = "AG"
MAX_CONCURRENCY = 5
MAX_METADATA_BYTES = 1_000_000
MISSING_VALUE_CUTOFF = -900.0
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)

STATE_SCHEMA = "uhi-cdc-g246-r2-weather-build-state-v1"
SNAPSHOT_SCHEMA = "uhi-cdc-g246-r2-power-snapshot-v1"
SIDECAR_SCHEMA = "uhi-cdc-g246-r2-weather-sidecar-v1"
NORMALIZATION_SCHEMA = "uhi-cdc-g246-r2-weather-normalization-v1"
MANIFEST_SCHEMA = "uhi-cdc-g246-r2-weather-manifest-v1"


class WeatherSidecarError(RuntimeError):
    """Raised when collection, remote data, or an immutable artifact is unsafe."""


@dataclass(frozen=True)
class SceneRequest:
    city: str
    year: int
    region: str
    role: str
    item_id: str
    datetime: str
    date: str
    longitude: float
    latitude: float
    source_scene_sha256: str

    @property
    def scene_id(self) -> str:
        return f"{self.city}:{self.year}:{self.item_id}"


@dataclass(frozen=True)
class SourceCollection:
    source_binding: Mapping[str, Any]
    scenes: tuple[SceneRequest, ...]


@dataclass(frozen=True)
class CityQuery:
    city: str
    region: str
    role: str
    longitude: float
    latitude: float
    dates: tuple[str, ...]
    scenes: tuple[SceneRequest, ...]

    @property
    def start(self) -> str:
        return self.dates[0].replace("-", "")

    @property
    def end(self) -> str:
        return self.dates[-1].replace("-", "")

    def api_params(self) -> dict[str, Any]:
        return {
            "parameters": ",".join(PARAMETERS),
            "community": COMMUNITY,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "start": self.start,
            "end": self.end,
            "format": "JSON",
            "time-standard": TIME_STANDARD,
        }

    def public_record(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "region": self.region,
            "role": self.role,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "dates": list(self.dates),
            "start": self.start,
            "end": self.end,
            "parameters": list(PARAMETERS),
            "community": COMMUNITY,
            "time_standard": TIME_STANDARD,
        }

    @property
    def query_sha256(self) -> str:
        return canonical_sha256(self.public_record())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_forbidden_path(path: str | os.PathLike[str], label: str) -> Path:
    folded = os.fspath(path).replace("\\", "/").casefold()
    if "locked_test" in folded or "locked-test" in folded or "sealed" in folded:
        raise WeatherSidecarError(f"{label} may not reference locked/sealed data")
    return Path(path)


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise WeatherSidecarError(f"{label} must be a regular file: {path}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise WeatherSidecarError(f"refusing to replace symlink: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WeatherSidecarError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise WeatherSidecarError(f"{label} root must be an object")
    return payload


@contextmanager
def builder_lock(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.builder.lock"
    if lock_path.is_symlink():
        raise WeatherSidecarError("builder lock may not be a symlink")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WeatherSidecarError(f"another builder owns {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _scene_date(timestamp: str) -> str:
    raw = str(timestamp).strip()
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError as exc:
        raise WeatherSidecarError(f"scene datetime is not ISO-8601: {timestamp!r}") from exc
    if parsed.tzinfo is None:
        raise WeatherSidecarError("scene datetime must carry a timezone")
    return parsed.astimezone(timezone.utc).date().isoformat()


def read_registered_location(entry: g246_data.G246Scene) -> tuple[float, float]:
    """Read only ``metadata.npy`` from one registered scene NPZ."""

    path = _reject_forbidden_path(entry.file, "public scene metadata")
    _regular_file(path, "public scene NPZ")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            matches = [info for info in archive.infolist() if info.filename == "metadata.npy"]
            if len(matches) != 1:
                raise WeatherSidecarError(
                    f"{entry.scene_id} requires exactly one metadata.npy member"
                )
            info = matches[0]
            if info.file_size <= 0 or info.file_size > MAX_METADATA_BYTES:
                raise WeatherSidecarError(f"{entry.scene_id} metadata.npy size is unsafe")
            with archive.open(info, "r") as handle:
                raw = handle.read(MAX_METADATA_BYTES + 1)
    except (OSError, zipfile.BadZipFile) as exc:
        raise WeatherSidecarError(f"cannot read registered metadata for {entry.scene_id}") from exc
    if len(raw) > MAX_METADATA_BYTES:
        raise WeatherSidecarError(f"{entry.scene_id} metadata.npy exceeds size limit")
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(array, np.ndarray) or array.shape != ():
            raise ValueError("metadata array must be scalar")
        scalar = array.item()
        if not isinstance(scalar, (str, np.str_)):
            raise ValueError("metadata scalar must be text")
        metadata = json.loads(str(scalar))
    except (ValueError, OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise WeatherSidecarError(f"malformed metadata member for {entry.scene_id}") from exc
    if not isinstance(metadata, Mapping):
        raise WeatherSidecarError(f"metadata root is not an object for {entry.scene_id}")
    try:
        identity_matches = (
            metadata.get("city") == entry.city
            and int(metadata.get("year", -1)) == entry.year
            and metadata.get("item_id") == entry.item_id
            and str(metadata.get("datetime")) == entry.datetime
        )
    except (TypeError, ValueError) as exc:
        raise WeatherSidecarError(f"malformed metadata identity for {entry.scene_id}") from exc
    if not identity_matches:
        raise WeatherSidecarError(f"metadata identity differs for {entry.scene_id}")
    location = metadata.get("requested_center_lonlat")
    if not isinstance(location, Sequence) or isinstance(location, (str, bytes)) or len(location) != 2:
        raise WeatherSidecarError(f"metadata lacks requested_center_lonlat for {entry.scene_id}")
    longitude, latitude = float(location[0]), float(location[1])
    if (
        not math.isfinite(longitude)
        or not math.isfinite(latitude)
        or not -180.0 <= longitude <= 180.0
        or not -90.0 <= latitude <= 90.0
    ):
        raise WeatherSidecarError(f"metadata location is invalid for {entry.scene_id}")
    return longitude, latitude


def collect_public_scenes(
    split_receipt: Path = DEFAULT_SPLIT_RECEIPT,
) -> SourceCollection:
    """Collect public scene identity/date/location without opening target arrays."""

    guarded = _reject_forbidden_path(split_receipt, "split receipt")
    splits = g246_data.load_splits(guarded, role="fit+validation")
    requests_: list[SceneRequest] = []
    city_locations: dict[str, tuple[float, float]] = {}
    city_roles: dict[str, str] = {}
    for entry in (*splits.fit, *splits.validation):
        if entry.view_role not in ROLES:
            raise WeatherSidecarError(f"unexpected public role: {entry.view_role}")
        longitude, latitude = read_registered_location(entry)
        prior_location = city_locations.setdefault(entry.city, (longitude, latitude))
        if not (
            math.isclose(prior_location[0], longitude, rel_tol=0.0, abs_tol=1e-8)
            and math.isclose(prior_location[1], latitude, rel_tol=0.0, abs_tol=1e-8)
        ):
            raise WeatherSidecarError(f"city location changes across scenes: {entry.city}")
        prior_role = city_roles.setdefault(entry.city, entry.view_role)
        if prior_role != entry.view_role:
            raise WeatherSidecarError(f"city crosses fit/validation roles: {entry.city}")
        requests_.append(SceneRequest(
            city=entry.city,
            year=entry.year,
            region=g246_data.canonical_region(entry.region),
            role=entry.view_role,
            item_id=entry.item_id,
            datetime=entry.datetime,
            date=_scene_date(entry.datetime),
            longitude=longitude,
            latitude=latitude,
            source_scene_sha256=entry.sha256,
        ))
    requests_.sort(key=lambda item: (item.role, item.region, item.city, item.date, item.item_id))
    if len({item.scene_id for item in requests_}) != len(requests_):
        raise WeatherSidecarError("duplicate public scene identity")
    public = splits.public_record()
    binding = {
        "split_receipt": {
            "path": public["receipt_path"],
            "sha256": public["receipt_sha256"],
        },
        "source_manifest": {
            "path": public["source_manifest"],
            "sha256": public["source_manifest_sha256"],
        },
        "campaign_id": public["campaign_id"],
        "fit_view_sha256": public["fit_view_sha256"],
        "validation_view_sha256": public["validation_view_sha256"],
        "fit_city_count": public["fit_city_count"],
        "fit_scene_count": public["fit_scene_count"],
        "validation_city_count": public["validation_city_count"],
        "validation_scene_count": public["validation_scene_count"],
        "roles_opened": ["fit", "validation"],
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }
    return SourceCollection(binding, tuple(requests_))


def group_city_queries(scenes: Sequence[SceneRequest]) -> tuple[CityQuery, ...]:
    grouped: dict[str, list[SceneRequest]] = defaultdict(list)
    for scene in scenes:
        grouped[scene.city].append(scene)
    result: list[CityQuery] = []
    for city, records in sorted(grouped.items()):
        roles = {record.role for record in records}
        regions = {record.region for record in records}
        locations = {(record.longitude, record.latitude) for record in records}
        dates = tuple(sorted({record.date for record in records}))
        if len(roles) != 1 or len(regions) != 1 or len(locations) != 1:
            raise WeatherSidecarError(f"city query identity is inconsistent: {city}")
        if len(dates) != len(records):
            raise WeatherSidecarError(f"city has multiple public scenes on one date: {city}")
        longitude, latitude = next(iter(locations))
        result.append(CityQuery(
            city=city,
            region=next(iter(regions)),
            role=next(iter(roles)),
            longitude=longitude,
            latitude=latitude,
            dates=dates,
            scenes=tuple(sorted(records, key=lambda item: (item.date, item.item_id))),
        ))
    return tuple(result)


def fetch_power_response(
    query: CityQuery,
    *,
    get: Callable[..., Any] = requests.get,
    sleep: Callable[[float], None] = time.sleep,
    retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Fetch one city with bounded retry for transport/429/5xx failures."""

    attempts = len(retry_delays) + 1
    last_error: BaseException | None = None
    for attempt in range(attempts):
        status: int | None = None
        try:
            response = get(
                POWER_API,
                params=query.api_params(),
                headers={"User-Agent": "uhi-cdc-g246-r2-weather-sidecars/1.0"},
                timeout=float(timeout_seconds),
            )
            status = int(getattr(response, "status_code", 200))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("POWER JSON root is not an object")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if status is None:
                status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or status in {408, 425, 429} or status >= 500
            if not retryable or attempt == attempts - 1:
                break
            delay = float(retry_delays[attempt])
            if not math.isfinite(delay) or delay < 0:
                raise WeatherSidecarError("retry delays must be finite and nonnegative")
            sleep(delay)
    raise WeatherSidecarError(
        f"NASA POWER request failed for {query.city} after {attempts} attempts"
    ) from last_error


def _finite_power_value(value: Any, parameter: str, date: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeatherSidecarError(f"POWER {parameter}/{date} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= MISSING_VALUE_CUTOFF:
        raise WeatherSidecarError(f"POWER {parameter}/{date} is missing/nonfinite")
    return result


def compact_power_snapshot(query: CityQuery, response: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a POWER response to requested dates and minimal source metadata."""

    properties = response.get("properties")
    parameter_block = properties.get("parameter") if isinstance(properties, Mapping) else None
    if not isinstance(parameter_block, Mapping):
        raise WeatherSidecarError(f"POWER response lacks properties.parameter for {query.city}")
    descriptor_block = response.get("parameters")
    if not isinstance(descriptor_block, Mapping):
        raise WeatherSidecarError(f"POWER response lacks parameter descriptors for {query.city}")
    units: dict[str, str] = {}
    values: dict[str, dict[str, float]] = {date: {} for date in query.dates}
    for parameter in PARAMETERS:
        series = parameter_block.get(parameter)
        descriptor = descriptor_block.get(parameter)
        if not isinstance(series, Mapping) or not isinstance(descriptor, Mapping):
            raise WeatherSidecarError(f"POWER response lacks requested parameter {parameter}")
        unit = descriptor.get("units")
        if not isinstance(unit, str) or not unit.strip():
            raise WeatherSidecarError(f"POWER response lacks units for {parameter}")
        units[parameter] = unit.strip()
        for date in query.dates:
            api_date = date.replace("-", "")
            if api_date not in series:
                raise WeatherSidecarError(f"POWER response lacks {parameter}/{date}")
            values[date][parameter] = _finite_power_value(series[api_date], parameter, date)
    header = response.get("header")
    if not isinstance(header, Mapping):
        header = {}
    api = header.get("api")
    api_version = api.get("version") if isinstance(api, Mapping) else None
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "provider": "NASA POWER",
        "endpoint": POWER_API,
        "query": query.public_record(),
        "query_sha256": query.query_sha256,
        "requested_dates": list(query.dates),
        "parameters": list(PARAMETERS),
        "units": units,
        "values": values,
        "provider_metadata": {
            "api_version": api_version,
            "fill_value": header.get("fill_value"),
            "title": header.get("title"),
            "geometry": response.get("geometry"),
        },
        "fetched_utc": utc_now(),
        "source_response_canonical_sha256": canonical_sha256(response),
        "snapshot_policy": "requested scene dates plus minimal provider metadata only",
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }
    validate_power_snapshot(snapshot, query)
    return snapshot


def validate_power_snapshot(snapshot: Mapping[str, Any], query: CityQuery) -> None:
    if (
        snapshot.get("schema") != SNAPSHOT_SCHEMA
        or snapshot.get("provider") != "NASA POWER"
        or snapshot.get("endpoint") != POWER_API
        or snapshot.get("query") != query.public_record()
        or snapshot.get("query_sha256") != query.query_sha256
        or snapshot.get("requested_dates") != list(query.dates)
        or snapshot.get("parameters") != list(PARAMETERS)
        or snapshot.get("locked_test_opened") is not False
        or snapshot.get("target_arrays_opened") is not False
    ):
        raise WeatherSidecarError(f"POWER snapshot contract differs for {query.city}")
    if "times" in snapshot or "raw_response" in snapshot:
        raise WeatherSidecarError("POWER snapshot is not minimal")
    units = snapshot.get("units")
    values = snapshot.get("values")
    if not isinstance(units, Mapping) or set(units) != set(PARAMETERS):
        raise WeatherSidecarError(f"POWER snapshot units differ for {query.city}")
    if not isinstance(values, Mapping) or set(values) != set(query.dates):
        raise WeatherSidecarError(f"POWER snapshot dates differ for {query.city}")
    for date in query.dates:
        record = values.get(date)
        if not isinstance(record, Mapping) or set(record) != set(PARAMETERS):
            raise WeatherSidecarError(f"POWER snapshot features differ for {query.city}/{date}")
        for parameter in PARAMETERS:
            _finite_power_value(record[parameter], parameter, date)
    digest = snapshot.get("source_response_canonical_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise WeatherSidecarError(f"POWER snapshot lacks source digest for {query.city}")


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9._-]+", "_", value.casefold()).strip("._-")
    if not result:
        raise WeatherSidecarError(f"cannot derive safe filename from {value!r}")
    return result


def snapshot_relative_path(query: CityQuery) -> Path:
    suffix = hashlib.sha256(query.city.encode("utf-8")).hexdigest()[:10]
    return Path("snapshots") / f"{_slug(query.city)}_{suffix}.json"


def sidecar_relative_path(scene: SceneRequest) -> Path:
    suffix = hashlib.sha256(scene.item_id.encode("utf-8")).hexdigest()[:12]
    return Path("sidecars") / scene.role / (
        f"{_slug(scene.city)}_{scene.date.replace('-', '')}_{suffix}.json"
    )


def _sidecar_payload(
    scene: SceneRequest,
    snapshot_relative: Path,
    snapshot_sha256: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    values = snapshot["values"][scene.date]
    return {
        "schema": SIDECAR_SCHEMA,
        "scene_id": scene.scene_id,
        "city": scene.city,
        "year": scene.year,
        "region": scene.region,
        "role": scene.role,
        "item_id": scene.item_id,
        "datetime": scene.datetime,
        "date_utc": scene.date,
        "longitude": scene.longitude,
        "latitude": scene.latitude,
        "source_scene_sha256": scene.source_scene_sha256,
        "features": {parameter: float(values[parameter]) for parameter in PARAMETERS},
        "units": dict(snapshot["units"]),
        "source_snapshot": {
            "path": snapshot_relative.as_posix(),
            "sha256": snapshot_sha256,
            "query_sha256": snapshot["query_sha256"],
        },
        "target_free": True,
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }


def validate_sidecar(
    payload: Mapping[str, Any], scene: SceneRequest,
    snapshot_relative: Path, snapshot_sha256: str,
) -> None:
    expected_identity = {
        "schema": SIDECAR_SCHEMA,
        "scene_id": scene.scene_id,
        "city": scene.city,
        "year": scene.year,
        "region": scene.region,
        "role": scene.role,
        "item_id": scene.item_id,
        "datetime": scene.datetime,
        "date_utc": scene.date,
        "longitude": scene.longitude,
        "latitude": scene.latitude,
        "source_scene_sha256": scene.source_scene_sha256,
        "target_free": True,
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise WeatherSidecarError(f"sidecar identity differs for {scene.scene_id}: {key}")
    source = payload.get("source_snapshot")
    if not isinstance(source, Mapping) or (
        source.get("path") != snapshot_relative.as_posix()
        or source.get("sha256") != snapshot_sha256
    ):
        raise WeatherSidecarError(f"sidecar snapshot binding differs for {scene.scene_id}")
    features = payload.get("features")
    units = payload.get("units")
    if not isinstance(features, Mapping) or set(features) != set(PARAMETERS):
        raise WeatherSidecarError(f"sidecar features differ for {scene.scene_id}")
    if not isinstance(units, Mapping) or set(units) != set(PARAMETERS):
        raise WeatherSidecarError(f"sidecar units differ for {scene.scene_id}")
    for parameter in PARAMETERS:
        _finite_power_value(features[parameter], parameter, scene.date)


def _contract_record() -> dict[str, Any]:
    return {
        "provider": "NASA POWER",
        "endpoint": POWER_API,
        "temporal": "DAILY",
        "spatial": "point",
        "parameters": list(PARAMETERS),
        "community": COMMUNITY,
        "time_standard": TIME_STANDARD,
        "city_batching": "one request per city spanning all public scene dates",
        "maximum_concurrency": MAX_CONCURRENCY,
        "missing_value_cutoff_exclusive": MISSING_VALUE_CUTOFF,
        "snapshot_policy": "requested scene dates plus minimal provider metadata only",
        "normalization_scope": "fit_only_equal_region_equal_city_equal_scene",
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }


def _initial_state(collection: SourceCollection, queries: Sequence[CityQuery]) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": "collecting",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "source_binding": dict(collection.source_binding),
        "contract": _contract_record(),
        "city_transactions": {
            query.city: {
                "status": "pending",
                "query": query.public_record(),
                "query_sha256": query.query_sha256,
                "attempts": 0,
            }
            for query in queries
        },
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }


def _validate_state_binding(
    state: Mapping[str, Any], collection: SourceCollection, queries: Sequence[CityQuery]
) -> None:
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("source_binding") != dict(collection.source_binding)
        or state.get("contract") != _contract_record()
        or state.get("locked_test_opened") is not False
        or state.get("target_arrays_opened") is not False
    ):
        raise WeatherSidecarError("resume state changes the public/source contract")
    transactions = state.get("city_transactions")
    if not isinstance(transactions, Mapping) or set(transactions) != {q.city for q in queries}:
        raise WeatherSidecarError("resume state city set differs")
    by_city = {query.city: query for query in queries}
    for city, transaction in transactions.items():
        if not isinstance(transaction, Mapping):
            raise WeatherSidecarError(f"malformed city transaction: {city}")
        query = by_city[city]
        if (
            transaction.get("query") != query.public_record()
            or transaction.get("query_sha256") != query.query_sha256
            or transaction.get("status") not in {"pending", "running", "failed", "complete"}
        ):
            raise WeatherSidecarError(f"resume transaction differs for {city}")


def _write_or_validate(path: Path, payload: Mapping[str, Any], label: str) -> str:
    if path.exists() or path.is_symlink():
        existing = read_json(path, label)
        if existing != dict(payload):
            raise WeatherSidecarError(f"existing immutable {label} differs: {path}")
    else:
        atomic_json(path, payload)
    return sha256_file(path)


def _process_city(
    output: Path,
    query: CityQuery,
    fetcher: Callable[[CityQuery], Mapping[str, Any]],
) -> dict[str, Any]:
    snapshot_relative = snapshot_relative_path(query)
    snapshot_path = output / snapshot_relative
    if snapshot_path.exists() or snapshot_path.is_symlink():
        snapshot = read_json(snapshot_path, f"POWER snapshot for {query.city}")
        validate_power_snapshot(snapshot, query)
    else:
        response = fetcher(query)
        if not isinstance(response, Mapping):
            raise WeatherSidecarError(f"fetcher returned malformed response for {query.city}")
        snapshot = compact_power_snapshot(query, response)
        atomic_json(snapshot_path, snapshot)
    snapshot_sha = sha256_file(snapshot_path)
    sidecars: list[dict[str, Any]] = []
    for scene in query.scenes:
        relative = sidecar_relative_path(scene)
        payload = _sidecar_payload(scene, snapshot_relative, snapshot_sha, snapshot)
        digest = _write_or_validate(output / relative, payload, f"sidecar {scene.scene_id}")
        sidecars.append({
            "scene_id": scene.scene_id,
            "city": scene.city,
            "year": scene.year,
            "region": scene.region,
            "role": scene.role,
            "item_id": scene.item_id,
            "datetime": scene.datetime,
            "date_utc": scene.date,
            "path": relative.as_posix(),
            "sha256": digest,
            "source_scene_sha256": scene.source_scene_sha256,
        })
    return {
        "snapshot": {
            "path": snapshot_relative.as_posix(),
            "sha256": snapshot_sha,
            "source_response_canonical_sha256": snapshot["source_response_canonical_sha256"],
            "units": dict(snapshot["units"]),
        },
        "sidecars": sidecars,
    }


def _load_transaction_artifacts(
    output: Path, query: CityQuery, transaction: Mapping[str, Any]
) -> list[dict[str, Any]]:
    snapshot_record = transaction.get("snapshot")
    sidecar_records = transaction.get("sidecars")
    if not isinstance(snapshot_record, Mapping) or not isinstance(sidecar_records, list):
        raise WeatherSidecarError(f"complete transaction lacks artifacts: {query.city}")
    snapshot_relative = Path(str(snapshot_record.get("path", "")))
    if snapshot_relative.is_absolute() or ".." in snapshot_relative.parts:
        raise WeatherSidecarError(f"snapshot path escapes output: {query.city}")
    snapshot_path = output / snapshot_relative
    snapshot = read_json(snapshot_path, f"POWER snapshot for {query.city}")
    snapshot_sha = sha256_file(snapshot_path)
    if snapshot_sha != snapshot_record.get("sha256"):
        raise WeatherSidecarError(f"snapshot hash differs for {query.city}")
    validate_power_snapshot(snapshot, query)
    by_scene = {scene.scene_id: scene for scene in query.scenes}
    if {str(item.get("scene_id")) for item in sidecar_records if isinstance(item, Mapping)} != set(by_scene):
        raise WeatherSidecarError(f"sidecar set differs for {query.city}")
    loaded: list[dict[str, Any]] = []
    for record in sidecar_records:
        if not isinstance(record, Mapping):
            raise WeatherSidecarError(f"malformed sidecar record for {query.city}")
        scene = by_scene[str(record["scene_id"])]
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise WeatherSidecarError(f"sidecar path escapes output: {scene.scene_id}")
        path = output / relative
        payload = read_json(path, f"sidecar {scene.scene_id}")
        digest = sha256_file(path)
        if digest != record.get("sha256"):
            raise WeatherSidecarError(f"sidecar hash differs for {scene.scene_id}")
        validate_sidecar(payload, scene, snapshot_relative, snapshot_sha)
        loaded.append(payload)
    return loaded


def build_region_balanced_normalization(
    sidecars: Sequence[Mapping[str, Any]],
    *,
    fit_view_sha256: str,
) -> dict[str, Any]:
    """Fit equal-region/equal-city/equal-scene moments from fit sidecars only."""

    fit = [record for record in sidecars if record.get("role") == "fit"]
    if not fit or any(record.get("role") not in ROLES for record in sidecars):
        raise WeatherSidecarError("normalization requires public fit/validation sidecars")
    by_region_city: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        region: defaultdict(list) for region in REGIONS
    }
    for record in fit:
        region = str(record.get("region"))
        city = str(record.get("city"))
        if region not in by_region_city or not city:
            raise WeatherSidecarError("fit normalization has invalid city/region")
        by_region_city[region][city].append(record)
    if any(not by_region_city[region] for region in REGIONS):
        raise WeatherSidecarError("fit normalization requires every macro-region")
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    region_means: dict[str, dict[str, float]] = {region: {} for region in REGIONS}
    for parameter in PARAMETERS:
        regional_first: list[float] = []
        regional_second: list[float] = []
        for region in REGIONS:
            city_first: list[float] = []
            city_second: list[float] = []
            for records in by_region_city[region].values():
                values = [
                    _finite_power_value(record["features"][parameter], parameter, str(record["date_utc"]))
                    for record in records
                ]
                city_first.append(fmean(values))
                city_second.append(fmean(value * value for value in values))
            first = fmean(city_first)
            second = fmean(city_second)
            region_means[region][parameter] = first
            regional_first.append(first)
            regional_second.append(second)
        mean = fmean(regional_first)
        second_moment = fmean(regional_second)
        means[parameter] = mean
        stds[parameter] = max(math.sqrt(max(second_moment - mean * mean, 0.0)), 1e-6)
    region_city_counts = {region: len(by_region_city[region]) for region in REGIONS}
    region_scene_counts = {
        region: sum(len(records) for records in by_region_city[region].values())
        for region in REGIONS
    }
    return {
        "schema": NORMALIZATION_SCHEMA,
        "scope": "fit_only",
        "weighting": "equal_region_then_equal_city_then_equal_scene",
        "moment_formula": (
            "E[x^k]=(1/3) sum_region (1/n_city_region) sum_city "
            "(1/n_scene_city) sum_scene x^k, k in {1,2}"
        ),
        "parameters": list(PARAMETERS),
        "mean": means,
        "std": stds,
        "region_mean": region_means,
        "fit_view_sha256": fit_view_sha256,
        "fit_city_count": len({str(record["city"]) for record in fit}),
        "fit_scene_count": len(fit),
        "fit_region_city_counts": region_city_counts,
        "fit_region_scene_counts": region_scene_counts,
        "validation_values_used": False,
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }


def _count_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_role = Counter(str(record.get("role")) for record in records)
    by_region = Counter(str(record.get("region")) for record in records)
    by_role_region = Counter(
        (str(record.get("role")), str(record.get("region"))) for record in records
    )
    return {
        "scenes": len(records),
        "cities": len({str(record.get("city")) for record in records}),
        "scenes_by_role": {role: by_role[role] for role in ROLES},
        "scenes_by_region": {region: by_region[region] for region in REGIONS},
        "scenes_by_role_region": {
            role: {region: by_role_region[(role, region)] for region in REGIONS}
            for role in ROLES
        },
    }


def _validate_complete_inventory(output: Path, state: Mapping[str, Any]) -> None:
    expected = {Path("build_state.json"), Path("manifest.json"), Path("normalization.json")}
    transactions = state.get("city_transactions")
    if not isinstance(transactions, Mapping):
        raise WeatherSidecarError("complete output lacks city transactions")
    for city, transaction in transactions.items():
        if not isinstance(transaction, Mapping) or transaction.get("status") != "complete":
            raise WeatherSidecarError(f"complete inventory has unfinished city: {city}")
        snapshot = transaction.get("snapshot")
        sidecars = transaction.get("sidecars")
        if not isinstance(snapshot, Mapping) or not isinstance(sidecars, list):
            raise WeatherSidecarError(f"complete inventory lacks artifacts: {city}")
        expected.add(Path(str(snapshot.get("path", ""))))
        for record in sidecars:
            if not isinstance(record, Mapping):
                raise WeatherSidecarError(f"complete inventory has malformed sidecar: {city}")
            expected.add(Path(str(record.get("path", ""))))
    actual: set[Path] = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise WeatherSidecarError(f"complete output contains a symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(output))
        elif not path.is_dir():
            raise WeatherSidecarError(f"complete output contains a special file: {path}")
    if actual != expected:
        missing = sorted(item.as_posix() for item in expected - actual)
        extra = sorted(item.as_posix() for item in actual - expected)
        raise WeatherSidecarError(
            f"complete output inventory differs; missing={missing}, extra={extra}"
        )


def _finalize(
    output: Path,
    state: dict[str, Any],
    collection: SourceCollection,
    queries: Sequence[CityQuery],
) -> dict[str, Any]:
    by_city = {query.city: query for query in queries}
    sidecars: list[dict[str, Any]] = []
    snapshot_records: list[dict[str, Any]] = []
    units: dict[str, str] | None = None
    for city in sorted(by_city):
        transaction = state["city_transactions"][city]
        loaded = _load_transaction_artifacts(output, by_city[city], transaction)
        sidecars.extend(loaded)
        snapshot = transaction["snapshot"]
        snapshot_units = dict(snapshot["units"])
        if units is None:
            units = snapshot_units
        elif units != snapshot_units:
            raise WeatherSidecarError("POWER parameter units change across city snapshots")
        snapshot_records.append({
            "city": city,
            "region": by_city[city].region,
            "role": by_city[city].role,
            **dict(snapshot),
        })
    sidecars.sort(key=lambda item: (item["role"], item["region"], item["city"], item["date_utc"]))
    normalization = build_region_balanced_normalization(
        sidecars,
        fit_view_sha256=str(collection.source_binding["fit_view_sha256"]),
    )
    normalization_path = output / "normalization.json"
    atomic_json(normalization_path, normalization)
    normalization_sha = sha256_file(normalization_path)
    scene_records = []
    for transaction in state["city_transactions"].values():
        scene_records.extend(transaction["sidecars"])
    scene_records.sort(key=lambda item: (item["role"], item["region"], item["city"], item["date_utc"]))
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "created_utc": state["created_utc"],
        "completed_utc": utc_now(),
        "source_binding": dict(collection.source_binding),
        "contract": _contract_record(),
        "counts": _count_records(scene_records),
        "units": units or {},
        "snapshots": snapshot_records,
        "scenes": scene_records,
        "normalization": {
            "path": "normalization.json",
            "sha256": normalization_sha,
            "schema": NORMALIZATION_SCHEMA,
            "scope": "fit_only",
        },
        "locked_test_opened": False,
        "target_arrays_opened": False,
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)
    state.update({
        "status": "complete",
        "completed_utc": manifest["completed_utc"],
        "manifest": {
            "path": "manifest.json",
            "sha256": sha256_file(manifest_path),
            "content_sha256": manifest["manifest_content_sha256"],
        },
        "normalization": manifest["normalization"],
        "updated_utc": utc_now(),
    })
    atomic_json(output / "build_state.json", state)
    _validate_complete_inventory(output, state)
    return manifest


def build(
    *,
    collection: SourceCollection,
    output: Path,
    resume: bool,
    workers: int = 4,
    max_new_cities: int | None = None,
    fetcher: Callable[[CityQuery], Mapping[str, Any]] = fetch_power_response,
) -> dict[str, Any]:
    """Build or resume sidecars; ``max_new_cities`` creates a safe partial run."""

    output = _reject_forbidden_path(output, "weather sidecar output").resolve()
    if workers <= 0 or workers > MAX_CONCURRENCY:
        raise WeatherSidecarError(f"workers must lie in [1,{MAX_CONCURRENCY}]")
    if max_new_cities is not None and max_new_cities <= 0:
        raise WeatherSidecarError("max_new_cities must be positive")
    queries = group_city_queries(collection.scenes)
    if not queries:
        raise WeatherSidecarError("public scene collection is empty")
    state_path = output / "build_state.json"
    with builder_lock(output):
        if output.exists() and (output.is_symlink() or not output.is_dir()):
            raise WeatherSidecarError("output must be a regular directory")
        if resume:
            if not state_path.is_file():
                raise WeatherSidecarError("--resume requires build_state.json")
            state = read_json(state_path, "weather build state")
            _validate_state_binding(state, collection, queries)
        else:
            if output.exists() and any(output.iterdir()):
                raise WeatherSidecarError("nonempty output requires --resume")
            output.mkdir(parents=True, exist_ok=True)
            state = _initial_state(collection, queries)
            atomic_json(state_path, state)
        if state.get("status") == "complete":
            return verify_output(output=output, collection=collection, _lock_held=True)

        by_city = {query.city: query for query in queries}
        pending: list[CityQuery] = []
        for city in sorted(by_city):
            transaction = state["city_transactions"][city]
            if transaction["status"] == "complete":
                _load_transaction_artifacts(output, by_city[city], transaction)
            else:
                pending.append(by_city[city])
        if max_new_cities is not None:
            pending = pending[:max_new_cities]
        for query in pending:
            transaction = state["city_transactions"][query.city]
            transaction["status"] = "running"
            transaction["attempts"] = int(transaction.get("attempts", 0)) + 1
            transaction.pop("error", None)
        state["status"] = "collecting"
        state["updated_utc"] = utc_now()
        atomic_json(state_path, state)

        failures: list[tuple[str, BaseException]] = []
        if pending:
            with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
                futures = {
                    executor.submit(_process_city, output, query, fetcher): query
                    for query in pending
                }
                for future in as_completed(futures):
                    query = futures[future]
                    transaction = state["city_transactions"][query.city]
                    try:
                        result = future.result()
                    except Exception as exc:  # persist ordinary worker failures before raising
                        transaction["status"] = "failed"
                        transaction["error"] = {
                            "type": type(exc).__name__,
                            "message": str(exc)[:1000],
                            "utc": utc_now(),
                        }
                        failures.append((query.city, exc))
                    else:
                        transaction.update({
                            "status": "complete",
                            "snapshot": result["snapshot"],
                            "sidecars": result["sidecars"],
                            "completed_utc": utc_now(),
                        })
                    state["updated_utc"] = utc_now()
                    atomic_json(state_path, state)
        incomplete = [
            city for city, transaction in state["city_transactions"].items()
            if transaction["status"] != "complete"
        ]
        if failures:
            state["status"] = "failed"
            state["updated_utc"] = utc_now()
            atomic_json(state_path, state)
            names = ", ".join(city for city, _ in failures[:5])
            raise WeatherSidecarError(
                f"{len(failures)} city transaction(s) failed ({names}); resume is safe"
            ) from failures[0][1]
        if incomplete:
            state["status"] = "partial"
            state["remaining_city_count"] = len(incomplete)
            state["updated_utc"] = utc_now()
            atomic_json(state_path, state)
            return {
                "schema": MANIFEST_SCHEMA,
                "status": "partial",
                "completed_city_count": len(queries) - len(incomplete),
                "remaining_city_count": len(incomplete),
                "locked_test_opened": False,
                "target_arrays_opened": False,
            }
        state.pop("remaining_city_count", None)
        return _finalize(output, state, collection, queries)


def verify_output(
    *, output: Path, collection: SourceCollection, _lock_held: bool = False
) -> dict[str, Any]:
    """Verify all present artifacts without issuing any network request."""

    output = _reject_forbidden_path(output, "weather sidecar output").resolve()

    def verify() -> dict[str, Any]:
        state = read_json(output / "build_state.json", "weather build state")
        queries = group_city_queries(collection.scenes)
        _validate_state_binding(state, collection, queries)
        by_city = {query.city: query for query in queries}
        loaded_sidecars: list[dict[str, Any]] = []
        complete_cities = 0
        for city, query in sorted(by_city.items()):
            transaction = state["city_transactions"][city]
            if transaction["status"] == "complete":
                loaded_sidecars.extend(_load_transaction_artifacts(output, query, transaction))
                complete_cities += 1
        if state.get("status") == "complete":
            if complete_cities != len(queries):
                raise WeatherSidecarError("complete state has unfinished cities")
            normalization_record = state.get("normalization")
            manifest_record = state.get("manifest")
            if not isinstance(normalization_record, Mapping) or not isinstance(manifest_record, Mapping):
                raise WeatherSidecarError("complete state lacks final artifact bindings")
            normalization_path = output / "normalization.json"
            normalization = read_json(normalization_path, "weather normalization")
            if sha256_file(normalization_path) != normalization_record.get("sha256"):
                raise WeatherSidecarError("weather normalization hash differs")
            expected_normalization = build_region_balanced_normalization(
                loaded_sidecars,
                fit_view_sha256=str(collection.source_binding["fit_view_sha256"]),
            )
            if normalization != expected_normalization:
                raise WeatherSidecarError("weather normalization cannot be reproduced")
            manifest_path = output / "manifest.json"
            manifest = read_json(manifest_path, "weather manifest")
            if sha256_file(manifest_path) != manifest_record.get("sha256"):
                raise WeatherSidecarError("weather manifest hash differs")
            content = dict(manifest)
            recorded_content_sha = content.pop("manifest_content_sha256", None)
            if canonical_sha256(content) != recorded_content_sha:
                raise WeatherSidecarError("weather manifest content hash differs")
            if (
                manifest.get("schema") != MANIFEST_SCHEMA
                or manifest.get("status") != "complete"
                or manifest.get("source_binding") != dict(collection.source_binding)
                or manifest.get("contract") != _contract_record()
                or manifest.get("normalization") != dict(normalization_record)
                or manifest.get("locked_test_opened") is not False
                or manifest.get("target_arrays_opened") is not False
            ):
                raise WeatherSidecarError("weather manifest contract differs")
            transaction_records = []
            for transaction in state["city_transactions"].values():
                transaction_records.extend(transaction["sidecars"])
            transaction_records.sort(
                key=lambda item: (
                    item["role"], item["region"], item["city"], item["date_utc"]
                )
            )
            expected_snapshots = [
                {
                    "city": city,
                    "region": by_city[city].region,
                    "role": by_city[city].role,
                    **dict(state["city_transactions"][city]["snapshot"]),
                }
                for city in sorted(by_city)
            ]
            if (
                manifest.get("scenes") != transaction_records
                or manifest.get("snapshots") != expected_snapshots
                or manifest.get("counts") != _count_records(transaction_records)
            ):
                raise WeatherSidecarError("weather manifest scene inventory differs")
            _validate_complete_inventory(output, state)
        return {
            "schema": "uhi-cdc-g246-r2-weather-offline-verification-v1",
            "valid": True,
            "status": state.get("status"),
            "complete_city_count": complete_cities,
            "total_city_count": len(queries),
            "verified_sidecar_count": len(loaded_sidecars),
            "network_requests": 0,
            "locked_test_opened": False,
            "target_arrays_opened": False,
        }

    if _lock_held:
        return verify()
    with builder_lock(output):
        return verify()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-receipt", type=Path, default=DEFAULT_SPLIT_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-new-cities", type=int,
        help="safe smoke/partial mode: process at most this many unfinished cities",
    )
    parser.add_argument(
        "--offline-verify", action="store_true",
        help="verify local snapshots/sidecars/manifest without network access",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    collection = collect_public_scenes(args.split_receipt)
    if args.offline_verify:
        if args.resume or args.max_new_cities is not None:
            raise WeatherSidecarError("--offline-verify cannot be combined with build flags")
        result = verify_output(output=args.output, collection=collection)
    else:
        result = build(
            collection=collection,
            output=args.output,
            resume=args.resume,
            workers=args.workers,
            max_new_cities=args.max_new_cities,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WeatherSidecarError as exc:
        print(f"G246 R2 weather sidecar build blocked: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
