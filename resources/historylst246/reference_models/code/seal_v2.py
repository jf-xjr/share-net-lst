"""Independent cryptographic preflight for the v2 sealed evaluation.

This module performs local file and schema checks only; it never imports a
network client or opens a sealed data product.  Builder/evaluator entrypoints
should call :func:`authorize_sealed_evaluation` (or the callback wrapper
:func:`run_after_sealed_preflight`) before their first network or sealed-data
operation.

Security boundary
-----------------
The official policy is byte-pinned by ``OFFICIAL_POLICY_SHA256`` and contains
the ordered primary and reserve name/longitude/latitude identities independently
of mutable city/reserve configs.  The unlock manifest is not trusted to select
files: the caller supplies the exact artifact paths, including a nonempty list
covering every frozen frontier-baseline file.  The fixed experiment-plan path
is likewise bound to its final bytes by the unlock manifest, allowing planned
novel-family additions before their first run.  The manifest must name and
hash those same regular workspace files.  This prevents placeholder tokens
and path-role substitution within the checked process.

The policy and JSONL audit remain local unsigned files.  An actor able to edit
the policy, verifier, and any externally retained chain head can rewrite the
entire mechanism.  SHA-256 proves byte identity, not scientific adequacy or
the absence of leakage before freeze.  Hash checking also has an unavoidable
time-of-check/time-of-use window unless callers retain immutable inputs or
reverify immediately before use.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_POLICY_PATH = WORKSPACE_ROOT / "data/v2/seal_policy.json"
OFFICIAL_POLICY_SHA256 = "5407b9739a37b782afb872a46b501be68eb947e683825432bd6c5c2dbc0d24d2"
ZERO_SHA256 = "0" * 64
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
CITY_NAME = re.compile(r"^[a-z0-9_]+$")

POLICY_SCHEMA = "uhi-cdc-v2-seal-policy-3"
UNLOCK_SCHEMA = "uhi-cdc-v2-unlock-3"
AUDIT_SCHEMA = "uhi-cdc-v2-seal-audit-2"
UNLOCK_INTENT = "single_confirmatory_evaluation"

SINGLE_ARTIFACT_ROLES = (
    "config",
    "protocol",
    "reserve_config",
    "build_contract",
    "experiment_plan",
    "candidate_model_code",
    "candidate_checkpoint",
    "evaluator_code",
    "metrics_code",
)
LIST_ARTIFACT_ROLES = ("source_artifacts", "validation_artifacts", "baseline_artifacts")
AUDIT_EVENTS = {
    "preflight_pass",
    "acquisition_started",
    "acquisition_completed",
    "evaluation_started",
    "evaluation_completed",
    "failure",
}


class SealError(RuntimeError):
    """Base class for a refused sealed operation."""


class PolicyError(SealError):
    pass


class CityIdentityError(SealError):
    pass


class UnlockError(SealError):
    pass


class ArtifactMismatchError(UnlockError):
    pass


class AuditError(SealError):
    pass


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Exact files selected by the caller before unlock verification."""

    source_artifacts: tuple[Path, ...]
    validation_artifacts: tuple[Path, ...]
    baseline_artifacts: tuple[Path, ...]
    candidate_model_code: Path
    candidate_checkpoint: Path
    evaluator_code: Path
    metrics_code: Path

    @classmethod
    def from_paths(
        cls,
        *,
        source_artifacts: Sequence[os.PathLike[str] | str],
        validation_artifacts: Sequence[os.PathLike[str] | str],
        baseline_artifacts: Sequence[os.PathLike[str] | str],
        candidate_model_code: os.PathLike[str] | str,
        candidate_checkpoint: os.PathLike[str] | str,
        evaluator_code: os.PathLike[str] | str,
        metrics_code: os.PathLike[str] | str,
    ) -> "EvaluationArtifacts":
        return cls(
            source_artifacts=tuple(Path(path) for path in source_artifacts),
            validation_artifacts=tuple(Path(path) for path in validation_artifacts),
            baseline_artifacts=tuple(Path(path) for path in baseline_artifacts),
            candidate_model_code=Path(candidate_model_code),
            candidate_checkpoint=Path(candidate_checkpoint),
            evaluator_code=Path(evaluator_code),
            metrics_code=Path(metrics_code),
        )


@dataclass(frozen=True)
class SealedAuthorization:
    evaluation_id: str
    policy_sha256: str
    unlock_manifest_sha256: str
    artifact_set_sha256: str
    verified_at_utc: str
    sealed_city_names: tuple[str, ...]
    sealed_reserve_names: tuple[str, ...]
    artifacts: tuple[tuple[str, str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "policy_sha256": self.policy_sha256,
            "unlock_manifest_sha256": self.unlock_manifest_sha256,
            "artifact_set_sha256": self.artifact_set_sha256,
            "verified_at_utc": self.verified_at_utc,
            "sealed_city_names": list(self.sealed_city_names),
            "sealed_reserve_names": list(self.sealed_reserve_names),
            "artifacts": [
                {"role": role, "path": path, "sha256": digest}
                for role, path, digest in self.artifacts
            ],
        }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def _require_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise UnlockError(f"{location} schema mismatch; missing={missing}, extra={extra}")


def _require_sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise UnlockError(f"{location} must be exactly 64 hexadecimal characters")
    return value.casefold()


def _load_json_bytes(path: Path, error_class: type[SealError]) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise error_class(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(raw, parse_float=Decimal, parse_int=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise error_class(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise error_class(f"JSON root must be an object: {path}")
    return value, raw


def _workspace_file(path: os.PathLike[str] | str, workspace_root: Path, role: str) -> tuple[Path, str]:
    root = workspace_root.resolve()
    supplied = Path(path)
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        unresolved_relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise ArtifactMismatchError(f"{role} path is outside workspace: {path}") from exc
    cursor = root
    for component in unresolved_relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ArtifactMismatchError(f"{role} path contains a symlink: {path}")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ArtifactMismatchError(f"{role} is absent or outside workspace: {path}") from exc
    if not resolved.is_file():
        raise ArtifactMismatchError(f"{role} is not a regular file: {path}")
    return resolved, relative.as_posix()


def _validate_policy(policy: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "evaluation_id",
        "created_utc",
        "expected_city_count",
        "immutable_sealed_cities",
        "expected_reserve_count",
        "immutable_sealed_reserves",
        "fixed_paths",
        "required_single_artifact_roles",
        "required_list_artifact_roles",
        "unlock_schema_version",
        "audit_schema_version",
        "threat_model",
    }
    actual_keys = set(policy)
    if actual_keys != expected_keys:
        raise PolicyError(
            f"policy schema mismatch; missing={sorted(expected_keys-actual_keys)}, "
            f"extra={sorted(actual_keys-expected_keys)}"
        )
    if policy["schema_version"] != POLICY_SCHEMA:
        raise PolicyError(f"unsupported policy schema: {policy['schema_version']!r}")
    if policy["unlock_schema_version"] != UNLOCK_SCHEMA or policy["audit_schema_version"] != AUDIT_SCHEMA:
        raise PolicyError("policy names unsupported unlock or audit schema")
    evaluation_id = policy["evaluation_id"]
    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        raise PolicyError("policy evaluation_id must be a nonempty string")
    if policy["required_single_artifact_roles"] != list(SINGLE_ARTIFACT_ROLES):
        raise PolicyError("policy single artifact roles differ from verifier")
    if policy["required_list_artifact_roles"] != list(LIST_ARTIFACT_ROLES):
        raise PolicyError("policy list artifact roles differ from verifier")
    fixed = policy["fixed_paths"]
    fixed_roles = {"config", "protocol", "reserve_config", "build_contract", "experiment_plan"}
    if not isinstance(fixed, dict) or set(fixed) != fixed_roles:
        raise PolicyError(f"policy fixed_paths must contain exactly {sorted(fixed_roles)}")
    for role, path in fixed.items():
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise PolicyError(f"policy {role} path must be a normalized relative path")

    cities = policy["immutable_sealed_cities"]
    if not isinstance(cities, list) or len(cities) != int(policy["expected_city_count"]):
        raise PolicyError("policy sealed city count does not match expected_city_count")
    names: set[str] = set()
    for index, city in enumerate(cities):
        if not isinstance(city, dict) or set(city) != {"name", "split", "lon", "lat"}:
            raise PolicyError(f"policy city {index} has invalid schema")
        name = city["name"]
        if not isinstance(name, str) or CITY_NAME.fullmatch(name) is None or name in names:
            raise PolicyError(f"policy city {index} has invalid or duplicate name")
        names.add(name)
        if city["split"] != "sealed_test":
            raise PolicyError(f"policy city {name} is not sealed_test")
        try:
            lon, lat = Decimal(city["lon"]), Decimal(city["lat"])
        except Exception as exc:
            raise PolicyError(f"policy city {name} has invalid coordinates") from exc
        if not Decimal("-180") <= lon <= Decimal("180") or not Decimal("-90") <= lat <= Decimal("90"):
            raise PolicyError(f"policy city {name} coordinates are out of range")

    reserves = policy["immutable_sealed_reserves"]
    if not isinstance(reserves, list) or len(reserves) != int(policy["expected_reserve_count"]):
        raise PolicyError("policy sealed reserve count does not match expected_reserve_count")
    reserve_names: set[str] = set()
    for index, reserve in enumerate(reserves):
        if not isinstance(reserve, dict) or set(reserve) != {"name", "lon", "lat"}:
            raise PolicyError(f"policy sealed reserve {index} has invalid schema")
        name = reserve["name"]
        if (
            not isinstance(name, str)
            or CITY_NAME.fullmatch(name) is None
            or name in reserve_names
            or name in names
        ):
            raise PolicyError(f"policy sealed reserve {index} has invalid, duplicate, or primary name")
        reserve_names.add(name)
        try:
            lon, lat = Decimal(reserve["lon"]), Decimal(reserve["lat"])
        except Exception as exc:
            raise PolicyError(f"policy sealed reserve {name} has invalid coordinates") from exc
        if not Decimal("-180") <= lon <= Decimal("180") or not Decimal("-90") <= lat <= Decimal("90"):
            raise PolicyError(f"policy sealed reserve {name} coordinates are out of range")

    threat_model = policy["threat_model"]
    if not isinstance(threat_model, dict) or set(threat_model) != {"protects_against", "caveats"}:
        raise PolicyError("policy threat_model must contain protects_against and caveats")
    for field in ("protects_against", "caveats"):
        statements = threat_model[field]
        if not isinstance(statements, list) or not statements or any(
            not isinstance(statement, str) or not statement.strip() for statement in statements
        ):
            raise PolicyError(f"policy threat_model.{field} must be a nonempty string list")


def load_policy(
    policy_path: os.PathLike[str] | str = OFFICIAL_POLICY_PATH,
    *,
    expected_policy_sha256: str = OFFICIAL_POLICY_SHA256,
    workspace_root: os.PathLike[str] | str = WORKSPACE_ROOT,
) -> tuple[dict[str, Any], str]:
    """Load a byte-pinned policy; a caller cannot silently trust a replacement."""
    expected = _require_sha256(expected_policy_sha256, "expected policy sha256")
    resolved, _ = _workspace_file(policy_path, Path(workspace_root), "seal policy")
    policy, raw = _load_json_bytes(resolved, PolicyError)
    actual = sha256_bytes(raw)
    if actual != expected:
        raise PolicyError(f"seal policy sha256 mismatch: expected {expected}, found {actual}")
    _validate_policy(policy)
    return policy, actual


def validate_sealed_city_identities(config_path: os.PathLike[str] | str, policy: Mapping[str, Any]) -> tuple[str, ...]:
    """Reject ordered primary relabeling, removal, addition, duplication, or movement."""
    config, _ = _load_json_bytes(Path(config_path), CityIdentityError)
    cities = config.get("cities")
    if not isinstance(cities, list):
        raise CityIdentityError("config must contain a cities list")
    policy_cities = {city["name"]: city for city in policy["immutable_sealed_cities"]}
    policy_casefold = {name.casefold(): name for name in policy_cities}
    configured: list[Mapping[str, Any]] = []
    for index, city in enumerate(cities):
        if not isinstance(city, dict):
            raise CityIdentityError(f"config city {index} is not an object")
        name = city.get("name")
        split = city.get("split")
        if not isinstance(name, str) or not isinstance(split, str):
            raise CityIdentityError(f"config city {index} lacks string name/split")
        official_name = policy_casefold.get(name.casefold())
        if official_name is not None and name != official_name:
            raise CityIdentityError(f"sealed city relabeled by case/spelling: {name!r}")
        if official_name is not None and split != "sealed_test":
            raise CityIdentityError(f"sealed city {name} was relabeled to split {split!r}")
        if split == "sealed_test":
            if any(existing["name"] == name for existing in configured):
                raise CityIdentityError(f"duplicate sealed city: {name}")
            configured.append(city)

    expected_order = [city["name"] for city in policy["immutable_sealed_cities"]]
    actual_order = [city["name"] for city in configured]
    if expected_order != actual_order:
        expected_names = set(expected_order)
        actual_names = set(actual_order)
        raise CityIdentityError(
            f"sealed city identity/order changed; expected_order={expected_order}, actual_order={actual_order}, "
            f"missing={sorted(expected_names-actual_names)}, added={sorted(actual_names-expected_names)}"
        )
    for actual, expected in zip(configured, policy["immutable_sealed_cities"]):
        name = expected["name"]
        if set(("lon", "lat")).difference(actual):
            raise CityIdentityError(f"sealed city {name} lacks coordinates")
        try:
            actual_lon, actual_lat = Decimal(actual["lon"]), Decimal(actual["lat"])
            expected_lon, expected_lat = Decimal(expected["lon"]), Decimal(expected["lat"])
        except Exception as exc:
            raise CityIdentityError(f"sealed city {name} has invalid coordinates") from exc
        if actual_lon != expected_lon or actual_lat != expected_lat:
            raise CityIdentityError(
                f"sealed city {name} moved: expected ({expected_lon},{expected_lat}), "
                f"found ({actual_lon},{actual_lat})"
            )
    return tuple(city["name"] for city in policy["immutable_sealed_cities"])


def validate_sealed_reserve_identities(
    reserve_path: os.PathLike[str] | str, policy: Mapping[str, Any]
) -> tuple[str, ...]:
    """Validate the exact ordered reserve list without consulting primary config."""
    reserve_config, _ = _load_json_bytes(Path(reserve_path), CityIdentityError)
    configured = reserve_config.get("sealed_test")
    if not isinstance(configured, list):
        raise CityIdentityError("reserve config must contain a sealed_test list")
    expected = policy["immutable_sealed_reserves"]
    if len(configured) != len(expected):
        raise CityIdentityError(
            f"sealed reserve count changed: expected {len(expected)}, found {len(configured)}"
        )
    expected_order = [reserve["name"] for reserve in expected]
    actual_order: list[Any] = []
    for index, reserve in enumerate(configured):
        if not isinstance(reserve, dict) or set(reserve) != {"name", "lon", "lat"}:
            raise CityIdentityError(f"sealed reserve {index} has invalid schema")
        actual_order.append(reserve["name"])
    if actual_order != expected_order:
        raise CityIdentityError(
            f"sealed reserve identity/order changed; expected_order={expected_order}, actual_order={actual_order}"
        )
    for index, (actual, frozen) in enumerate(zip(configured, expected)):
        try:
            actual_lon, actual_lat = Decimal(actual["lon"]), Decimal(actual["lat"])
            expected_lon, expected_lat = Decimal(frozen["lon"]), Decimal(frozen["lat"])
        except Exception as exc:
            raise CityIdentityError(f"sealed reserve {index} has invalid coordinates") from exc
        if actual_lon != expected_lon or actual_lat != expected_lat:
            raise CityIdentityError(
                f"sealed reserve {frozen['name']} moved: expected ({expected_lon},{expected_lat}), "
                f"found ({actual_lon},{actual_lat})"
            )
    return tuple(expected_order)


def _expected_paths(
    policy: Mapping[str, Any],
    artifacts: EvaluationArtifacts,
    config_path: os.PathLike[str] | str,
    protocol_path: os.PathLike[str] | str,
    reserve_path: os.PathLike[str] | str,
    build_contract_path: os.PathLike[str] | str,
    experiment_plan_path: os.PathLike[str] | str,
) -> dict[str, Path | tuple[Path, ...]]:
    return {
        "config": Path(config_path),
        "protocol": Path(protocol_path),
        "reserve_config": Path(reserve_path),
        "build_contract": Path(build_contract_path),
        "experiment_plan": Path(experiment_plan_path),
        "source_artifacts": artifacts.source_artifacts,
        "validation_artifacts": artifacts.validation_artifacts,
        "baseline_artifacts": artifacts.baseline_artifacts,
        "candidate_model_code": artifacts.candidate_model_code,
        "candidate_checkpoint": artifacts.candidate_checkpoint,
        "evaluator_code": artifacts.evaluator_code,
        "metrics_code": artifacts.metrics_code,
    }


def _verify_descriptor(
    descriptor: Any,
    expected_path: Path,
    workspace_root: Path,
    location: str,
) -> tuple[str, str]:
    if not isinstance(descriptor, dict):
        raise UnlockError(f"{location} must be an artifact descriptor object")
    _require_keys(descriptor, {"path", "sha256"}, location)
    path_value = descriptor["path"]
    if not isinstance(path_value, str) or not path_value or "\\" in path_value:
        raise UnlockError(f"{location}.path must be a nonempty POSIX workspace-relative path")
    manifest_path = Path(path_value)
    if manifest_path.is_absolute() or ".." in manifest_path.parts or manifest_path.as_posix() != path_value:
        raise UnlockError(f"{location}.path is not normalized workspace-relative POSIX")
    actual_path, relative = _workspace_file(expected_path, workspace_root, location)
    if path_value != relative:
        raise ArtifactMismatchError(
            f"{location} path mismatch: unlock names {path_value!r}, caller selected {relative!r}"
        )
    declared_hash = _require_sha256(descriptor["sha256"], f"{location}.sha256")
    actual_hash = sha256_file(actual_path)
    if declared_hash != actual_hash:
        raise ArtifactMismatchError(
            f"{location} sha256 mismatch: declared {declared_hash}, actual {actual_hash}"
        )
    return relative, actual_hash


def _verify_artifacts(
    manifest_artifacts: Any,
    expected: Mapping[str, Path | tuple[Path, ...]],
    workspace_root: Path,
) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(manifest_artifacts, dict):
        raise UnlockError("unlock artifacts must be an object")
    required_roles = set(SINGLE_ARTIFACT_ROLES) | set(LIST_ARTIFACT_ROLES)
    _require_keys(manifest_artifacts, required_roles, "unlock.artifacts")
    verified: list[tuple[str, str, str]] = []
    for role in SINGLE_ARTIFACT_ROLES:
        expected_path = expected[role]
        if not isinstance(expected_path, Path):
            raise ArtifactMismatchError(f"internal expected-path type error for {role}")
        relative, digest = _verify_descriptor(
            manifest_artifacts[role], expected_path, workspace_root, f"unlock.artifacts.{role}"
        )
        verified.append((role, relative, digest))
    for role in LIST_ARTIFACT_ROLES:
        descriptors = manifest_artifacts[role]
        paths = expected[role]
        if not isinstance(descriptors, list) or not descriptors:
            raise UnlockError(f"unlock.artifacts.{role} must be a nonempty list")
        if not isinstance(paths, tuple) or not paths:
            raise ArtifactMismatchError(f"caller must select at least one {role} file")
        canonical_expected = sorted(
            (_workspace_file(path, workspace_root, f"expected {role}")[1], path) for path in paths
        )
        if len({relative for relative, _ in canonical_expected}) != len(canonical_expected):
            raise ArtifactMismatchError(f"caller selected duplicate {role} paths")
        if len(descriptors) != len(canonical_expected):
            raise ArtifactMismatchError(
                f"unlock.artifacts.{role} count differs from caller selection"
            )
        for index, ((_, expected_path), descriptor) in enumerate(zip(canonical_expected, descriptors)):
            relative, digest = _verify_descriptor(
                descriptor,
                expected_path,
                workspace_root,
                f"unlock.artifacts.{role}[{index}]",
            )
            verified.append((role, relative, digest))
    source_paths = {path for role, path, _ in verified if role == "source_artifacts"}
    validation_paths = {path for role, path, _ in verified if role == "validation_artifacts"}
    overlap = source_paths.intersection(validation_paths)
    if overlap:
        raise ArtifactMismatchError(f"source/validation artifact overlap: {sorted(overlap)}")
    return tuple(verified)


def authorize_sealed_evaluation(
    unlock_manifest: os.PathLike[str] | str | None,
    artifacts: EvaluationArtifacts,
    *,
    config_path: os.PathLike[str] | str | None = None,
    protocol_path: os.PathLike[str] | str | None = None,
    reserve_path: os.PathLike[str] | str | None = None,
    build_contract_path: os.PathLike[str] | str | None = None,
    experiment_plan_path: os.PathLike[str] | str | None = None,
    policy_path: os.PathLike[str] | str = OFFICIAL_POLICY_PATH,
    expected_policy_sha256: str = OFFICIAL_POLICY_SHA256,
    workspace_root: os.PathLike[str] | str = WORKSPACE_ROOT,
) -> SealedAuthorization:
    """Return authorization only after identities, schema, paths, and bytes match."""
    if unlock_manifest is None:
        raise UnlockError("sealed evaluation is locked: an unlock manifest is required")
    root = Path(workspace_root).resolve()
    policy, policy_hash = load_policy(
        policy_path,
        expected_policy_sha256=expected_policy_sha256,
        workspace_root=root,
    )
    fixed = policy["fixed_paths"]
    selected_config = Path(config_path) if config_path is not None else root / fixed["config"]
    selected_protocol = Path(protocol_path) if protocol_path is not None else root / fixed["protocol"]
    selected_reserve = Path(reserve_path) if reserve_path is not None else root / fixed["reserve_config"]
    selected_contract = (
        Path(build_contract_path)
        if build_contract_path is not None
        else root / fixed["build_contract"]
    )
    selected_experiment_plan = (
        Path(experiment_plan_path)
        if experiment_plan_path is not None
        else root / fixed["experiment_plan"]
    )
    _, config_relative = _workspace_file(selected_config, root, "config")
    _, protocol_relative = _workspace_file(selected_protocol, root, "protocol")
    _, reserve_relative = _workspace_file(selected_reserve, root, "reserve config")
    _, contract_relative = _workspace_file(selected_contract, root, "build contract")
    _, experiment_plan_relative = _workspace_file(
        selected_experiment_plan, root, "experiment plan"
    )
    selected_fixed = {
        "config": config_relative,
        "protocol": protocol_relative,
        "reserve_config": reserve_relative,
        "build_contract": contract_relative,
        "experiment_plan": experiment_plan_relative,
    }
    if selected_fixed != fixed:
        raise ArtifactMismatchError("fixed artifact path differs from the byte-pinned policy")
    city_names = validate_sealed_city_identities(selected_config, policy)
    reserve_names = validate_sealed_reserve_identities(selected_reserve, policy)

    unlock_path, _ = _workspace_file(unlock_manifest, root, "unlock manifest")
    unlock, unlock_raw = _load_json_bytes(unlock_path, UnlockError)
    _require_keys(
        unlock,
        {
            "schema_version",
            "evaluation_id",
            "policy_sha256",
            "intent",
            "issued_at_utc",
            "authorized_by",
            "artifacts",
        },
        "unlock",
    )
    if unlock["schema_version"] != policy["unlock_schema_version"]:
        raise UnlockError(f"unsupported unlock schema: {unlock['schema_version']!r}")
    if unlock["evaluation_id"] != policy["evaluation_id"]:
        raise UnlockError("unlock evaluation_id does not match the one byte-pinned evaluation")
    if _require_sha256(unlock["policy_sha256"], "unlock.policy_sha256") != policy_hash:
        raise UnlockError("unlock policy_sha256 does not match the verified policy")
    if unlock["intent"] != UNLOCK_INTENT:
        raise UnlockError(f"unlock intent must be {UNLOCK_INTENT!r}")
    if not isinstance(unlock["authorized_by"], str) or not unlock["authorized_by"].strip():
        raise UnlockError("unlock authorized_by must be a nonempty string")
    timestamp = unlock["issued_at_utc"]
    if not isinstance(timestamp, str):
        raise UnlockError("unlock issued_at_utc must be an RFC3339 string")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UnlockError("unlock issued_at_utc is not valid RFC3339") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise UnlockError("unlock issued_at_utc must include a timezone")

    expected = _expected_paths(
        policy,
        artifacts,
        selected_config,
        selected_protocol,
        selected_reserve,
        selected_contract,
        selected_experiment_plan,
    )
    verified = _verify_artifacts(unlock["artifacts"], expected, root)
    artifact_set_hash = sha256_bytes(
        _canonical_json(
            [{"role": role, "path": path, "sha256": digest} for role, path, digest in verified]
        )
    )
    return SealedAuthorization(
        evaluation_id=policy["evaluation_id"],
        policy_sha256=policy_hash,
        unlock_manifest_sha256=sha256_bytes(unlock_raw),
        artifact_set_sha256=artifact_set_hash,
        verified_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        sealed_city_names=city_names,
        sealed_reserve_names=reserve_names,
        artifacts=verified,
    )


def _audit_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "record_sha256"}


_AUDIT_TRANSITIONS: dict[tuple[str, ...], set[str]] = {
    (): {"preflight_pass"},
    ("preflight_pass",): {"acquisition_started", "evaluation_started", "failure"},
    ("preflight_pass", "acquisition_started"): {"acquisition_completed", "failure"},
    ("preflight_pass", "acquisition_started", "acquisition_completed"): {
        "evaluation_started",
        "failure",
    },
    ("preflight_pass", "evaluation_started"): {"evaluation_completed", "failure"},
    (
        "preflight_pass",
        "acquisition_started",
        "acquisition_completed",
        "evaluation_started",
    ): {"evaluation_completed", "failure"},
}


def _require_audit_transition(events: Sequence[str], event: str) -> None:
    allowed = _AUDIT_TRANSITIONS.get(tuple(events), set())
    if event not in allowed:
        state = events[-1] if events else "empty"
        raise AuditError(
            f"invalid or repeated audit event {event!r} after state {state!r}; "
            f"allowed={sorted(allowed)}"
        )


def audit_state(records: Sequence[Mapping[str, Any]]) -> str:
    """Return the validated one-shot state represented by an audit prefix."""
    _validate_audit_records(records)
    if not records:
        return "empty"
    events = tuple(str(record["event"]) for record in records)
    if events[-1] == "failure":
        return "failed"
    if events[-1] == "evaluation_completed":
        return "completed"
    return {
        "preflight_pass": "preflight_passed",
        "acquisition_started": "acquisition_in_progress",
        "acquisition_completed": "acquisition_completed",
        "evaluation_started": "evaluation_in_progress",
    }[events[-1]]


def _validate_audit_records(records: Sequence[Mapping[str, Any]]) -> None:
    previous = ZERO_SHA256
    events: list[str] = []
    expected_keys = {
        "schema_version",
        "evaluation_id",
        "sequence",
        "timestamp_utc",
        "event",
        "policy_sha256",
        "unlock_manifest_sha256",
        "artifact_set_sha256",
        "previous_record_sha256",
        "details",
        "record_sha256",
    }
    first_identity: tuple[str, str, str, str] | None = None
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise AuditError(f"audit record {index} has invalid schema")
        if record["schema_version"] != AUDIT_SCHEMA or record["sequence"] != index:
            raise AuditError(f"audit record {index} has invalid schema version/sequence")
        if record["event"] not in AUDIT_EVENTS:
            raise AuditError(f"audit record {index} has invalid event")
        _require_audit_transition(events, str(record["event"]))
        for field in ("policy_sha256", "unlock_manifest_sha256", "artifact_set_sha256", "previous_record_sha256", "record_sha256"):
            try:
                _require_sha256(record[field], f"audit record {index}.{field}")
            except UnlockError as exc:
                raise AuditError(str(exc)) from exc
        if record["previous_record_sha256"] != previous:
            raise AuditError(f"audit record {index} breaks the previous-record chain")
        calculated = sha256_bytes(_canonical_json(_audit_payload(record)))
        if record["record_sha256"] != calculated:
            raise AuditError(f"audit record {index} hash mismatch")
        identity = (
            record["evaluation_id"],
            record["policy_sha256"],
            record["unlock_manifest_sha256"],
            record["artifact_set_sha256"],
        )
        if first_identity is None:
            first_identity = identity
        elif identity != first_identity:
            raise AuditError(f"audit record {index} changes bound evaluation/artifacts")
        previous = record["record_sha256"]
        events.append(str(record["event"]))


def verify_audit_log(path: os.PathLike[str] | str) -> tuple[dict[str, Any], ...]:
    audit_path = Path(path)
    if not audit_path.exists():
        return ()
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuditError(f"cannot parse audit log {audit_path}: {exc}") from exc
    if any(not line.strip() for line in lines):
        raise AuditError("audit log contains a blank record")
    _validate_audit_records(records)
    return tuple(records)


def append_audit_record(
    path: os.PathLike[str] | str,
    authorization: SealedAuthorization,
    event: str,
    *,
    details: Mapping[str, Any] | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Lock, validate the full JSONL hash chain, and append one fsynced record."""
    if event not in AUDIT_EVENTS:
        raise AuditError(f"unsupported audit event: {event!r}")
    audit_path = Path(path)
    if audit_path.exists() and audit_path.is_symlink():
        raise AuditError("audit log must not be a symlink")
    if not audit_path.parent.is_dir():
        raise AuditError("audit log parent directory does not exist")
    timestamp = timestamp_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    detail_value = dict(details or {})
    # Validate serializability before taking the file lock.
    _canonical_json(detail_value)
    with audit_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            lines = handle.read().splitlines()
            if any(not line.strip() for line in lines):
                raise AuditError("audit log contains a blank record")
            try:
                records = [json.loads(line) for line in lines]
            except json.JSONDecodeError as exc:
                raise AuditError(f"existing audit log is invalid JSONL: {exc}") from exc
            _validate_audit_records(records)
            _require_audit_transition([str(record["event"]) for record in records], event)
            if records:
                first = records[0]
                bound = (
                    first["evaluation_id"],
                    first["policy_sha256"],
                    first["unlock_manifest_sha256"],
                    first["artifact_set_sha256"],
                )
                current = (
                    authorization.evaluation_id,
                    authorization.policy_sha256,
                    authorization.unlock_manifest_sha256,
                    authorization.artifact_set_sha256,
                )
                if current != bound:
                    raise AuditError("audit log is already bound to a different evaluation/unlock/artifact set")
            previous = records[-1]["record_sha256"] if records else ZERO_SHA256
            record: dict[str, Any] = {
                "schema_version": AUDIT_SCHEMA,
                "evaluation_id": authorization.evaluation_id,
                "sequence": len(records),
                "timestamp_utc": timestamp,
                "event": event,
                "policy_sha256": authorization.policy_sha256,
                "unlock_manifest_sha256": authorization.unlock_manifest_sha256,
                "artifact_set_sha256": authorization.artifact_set_sha256,
                "previous_record_sha256": previous,
                "details": detail_value,
            }
            record["record_sha256"] = sha256_bytes(_canonical_json(record))
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return record
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


T = TypeVar("T")


def run_after_sealed_preflight(
    action: Callable[[SealedAuthorization], T],
    unlock_manifest: os.PathLike[str] | str | None,
    artifacts: EvaluationArtifacts,
    *,
    audit_path: os.PathLike[str] | str,
    action_kind: str,
    config_path: os.PathLike[str] | str | None = None,
    protocol_path: os.PathLike[str] | str | None = None,
    reserve_path: os.PathLike[str] | str | None = None,
    build_contract_path: os.PathLike[str] | str | None = None,
    experiment_plan_path: os.PathLike[str] | str | None = None,
    policy_path: os.PathLike[str] | str = OFFICIAL_POLICY_PATH,
    expected_policy_sha256: str = OFFICIAL_POLICY_SHA256,
    workspace_root: os.PathLike[str] | str = WORKSPACE_ROOT,
) -> T:
    """Verify first, then invoke a builder/evaluator callback.

    ``action_kind`` is ``"acquisition"`` or ``"evaluation"``.  A first
    evaluation is the combined one-shot path.  Alternatively, one acquisition
    callback may complete before one evaluation callback reuses the same audit
    authorization without appending another preflight.  Repeated/terminal
    actions are rejected before callback entry.  An absent or invalid unlock
    likewise raises before a callback can make a network request.
    """
    if action_kind not in {"acquisition", "evaluation"}:
        raise ValueError("action_kind must be 'acquisition' or 'evaluation'")
    authorization = authorize_sealed_evaluation(
        unlock_manifest,
        artifacts,
        config_path=config_path,
        protocol_path=protocol_path,
        reserve_path=reserve_path,
        build_contract_path=build_contract_path,
        experiment_plan_path=experiment_plan_path,
        policy_path=policy_path,
        expected_policy_sha256=expected_policy_sha256,
        workspace_root=workspace_root,
    )
    records = verify_audit_log(audit_path)
    if not records:
        append_audit_record(audit_path, authorization, "preflight_pass")
    append_audit_record(audit_path, authorization, f"{action_kind}_started")
    try:
        result = action(authorization)
    except Exception as exc:
        append_audit_record(
            audit_path,
            authorization,
            "failure",
            details={"action_kind": action_kind, "exception_type": type(exc).__name__},
        )
        raise
    append_audit_record(audit_path, authorization, f"{action_kind}_completed")
    return result


__all__ = [
    "ArtifactMismatchError",
    "AuditError",
    "CityIdentityError",
    "EvaluationArtifacts",
    "OFFICIAL_POLICY_PATH",
    "OFFICIAL_POLICY_SHA256",
    "PolicyError",
    "SealError",
    "SealedAuthorization",
    "UnlockError",
    "append_audit_record",
    "audit_state",
    "authorize_sealed_evaluation",
    "load_policy",
    "run_after_sealed_preflight",
    "sha256_file",
    "validate_sealed_city_identities",
    "validate_sealed_reserve_identities",
    "verify_audit_log",
]
