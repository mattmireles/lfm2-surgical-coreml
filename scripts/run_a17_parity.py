#!/usr/bin/env python3
"""Build, dispatch, run, and validate the frozen Plan 012 iPhone probe.

The command is deliberately narrow: three registered fixed-512 twins, one
physical A17 Pro phone, one explicit development profile, and zero performance
samples. Generated models, apps, signing payloads, and results remain ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_ROOT = REPO_ROOT / "outputs/lfm2_surgical/stage1"
FIXED_PAIR_ROOT = REPO_ROOT / "outputs/lfm2_surgical/selective/fixed_pairs_512"
A17_ROOT = REPO_ROOT / "outputs/lfm2_surgical/a17_parity"
IOS_ROOT = REPO_ROOT / "ios-a17-parity"
GENERATED_RESOURCES = IOS_ROOT / "Generated/Resources"
DEFAULT_DEVICE_ID = "8A12AEE8-0136-50BE-8EB3-91650E467F15"
DEFAULT_DEVICE_UDID = "00008130-000600660EA2001C"
DEFAULT_DEVICE_NAME = "Commas?"
DEFAULT_PRODUCT_TYPE = "iPhone16,2"
DEFAULT_TEAM = "6ETYBAJKY8"
DEFAULT_BUNDLE_ID = "com.mattmireles.LFM2A17Parity"
REGISTERED_PROMPT = (
    "Explain why fixed tensor shapes can help a compiler target an accelerator."
)
REGISTERED_BUCKET = 512
EXPECTED_TOKEN = 941
MAXIMUM_ABSOLUTE_TOLERANCE = 1.0e-2
PROFILE_DIRECTORY = Path.home() / "Library/MobileDevice/Provisioning Profiles"
KOKORO_PYTHON = Path(
    "/Users/mm/Documents/GitHub/kokoro-coreml/.venv/bin/python"
)
TOOLS = {
    "xcodebuild": Path("/usr/bin/xcodebuild"),
    "xcodegen": Path("/opt/homebrew/bin/xcodegen"),
    "xcrun": Path("/usr/bin/xcrun"),
    "security": Path("/usr/bin/security"),
}
SAFE_PATH = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
RESULT_SCHEMA = "a17-parity-result-v1"
DISPATCH_SCHEMA = "a17-parity-dispatch-v1"


@dataclass(frozen=True)
class PairSpec:
    """One immutable fixed-twin contract and its required named outputs."""

    name: str
    fixed_sha256: str
    outputs: tuple[str, ...]


PAIR_SPECS = (
    PairSpec(
        "segment_04_conv_6_7",
        "fd270492253c5f653ead28ff0368ae912278ae21fb201a9910d97c337be29b5b",
        ("hidden_out", "conv_state_6_out", "conv_state_7_out"),
    ),
    PairSpec(
        "segment_00_conv_0_1",
        "dfbc10cad5d42c26b97b69d1ce74f01a28a07f170323529871aa4a4fb8576218",
        ("hidden_out", "conv_state_0_out", "conv_state_1_out"),
    ),
    PairSpec(
        "segment_02_conv_3_4",
        "db264232357277efdc5d0efca961c50ba1971e53018da3efade7b29583e7133a",
        ("hidden_out", "conv_state_3_out", "conv_state_4_out"),
    ),
)
PAIR_BY_NAME = {pair.name: pair for pair in PAIR_SPECS}
COMPARISON_ORDERS = (
    "GPU->ANE",
    "GPU->ANE",
    "ANE->GPU",
    "GPU->ANE",
    "ANE->GPU",
    "GPU->ANE",
)


def _canonical_bytes(payload: Any) -> bytes:
    """Return the stable UTF-8 representation used by every payload hash."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def make_envelope(schema_version: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap one JSON payload with a version and canonical payload SHA-256."""

    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        "schema_version": schema_version,
    }


def validate_envelope(document: dict[str, Any], schema_version: str) -> dict[str, Any]:
    """Validate a versioned envelope and return its hash-bound payload."""

    if set(document) != {"payload", "payload_sha256", "schema_version"}:
        raise ValueError("envelope fields drifted")
    if document["schema_version"] != schema_version:
        raise ValueError("envelope schema version drifted")
    payload = document["payload"]
    expected = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if document["payload_sha256"] != expected:
        raise ValueError("envelope payload SHA-256 mismatch")
    return payload


def write_envelope(path: Path, schema_version: str, payload: dict[str, Any]) -> None:
    """Atomically write one canonical, self-validating JSON evidence file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        make_envelope(schema_version, payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def read_envelope(path: Path, schema_version: str) -> dict[str, Any]:
    """Read and validate one durable evidence envelope."""

    return validate_envelope(json.loads(path.read_text()), schema_version)


def file_sha256(path: Path) -> str:
    """Hash one exact evidence file without interpreting its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_tree_sha256(package: Path) -> str:
    """Match the frozen sorted-relative-path package hash used by prior plans."""

    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in package.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(package)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one host command with the explicit Apple/toolchain PATH."""

    environment = os.environ.copy()
    environment["PATH"] = SAFE_PATH
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode:
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{combined}")
    return result


def _devicectl_json(arguments: list[str], *, timeout: float = 60.0) -> dict[str, Any]:
    """Run a CoreDevice command through its supported JSON-file interface."""

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "devicectl.json"
        _run(
            [str(TOOLS["xcrun"]), "devicectl", *arguments, "--json-output", str(output)],
            timeout=timeout,
        )
        document = json.loads(output.read_text())
    if document.get("info", {}).get("outcome") != "success":
        raise RuntimeError("devicectl did not report success")
    return document["result"]


def live_device_snapshot(device_id: str, device_udid: str) -> dict[str, Any]:
    """Prove the exact physical A17 phone, connection, DDI, and unlock state."""

    details = _devicectl_json(["device", "info", "details", "--device", device_id])
    lock = _devicectl_json(["device", "info", "lockState", "--device", device_id])
    device = details.get("deviceProperties", {})
    hardware = details.get("hardwareProperties", {})
    connection = details.get("connectionProperties", {})
    checks = {
        "identifier": details.get("identifier") == device_id,
        "udid": hardware.get("udid") == device_udid,
        "product_type": hardware.get("productType") == DEFAULT_PRODUCT_TYPE,
        "physical": hardware.get("reality") == "physical",
        "booted": device.get("bootState") == "booted",
        "developer_mode": device.get("developerModeStatus") == "enabled",
        "ddi": device.get("ddiServicesAvailable") is True,
        "paired": connection.get("pairingState") == "paired",
        "wired": connection.get("transportType") == "wired",
        "tunnel": connection.get("tunnelState") == "connected",
        "unlocked": lock.get("passcodeRequired") is False,
        "unlocked_since_boot": lock.get("unlockedSinceBoot") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"exact A17 phone is not ready: {failed}")
    return {
        "boot_state": device.get("bootState"),
        "coredevice_identifier": details.get("identifier"),
        "ddi_services_available": device.get("ddiServicesAvailable"),
        "developer_mode": device.get("developerModeStatus"),
        "hardware_udid": hardware.get("udid"),
        "marketing_name": hardware.get("marketingName"),
        "name": device.get("name"),
        "os_build": device.get("osBuildUpdate"),
        "os_release_type": device.get("releaseType"),
        "os_version": device.get("osVersionNumber"),
        "pairing_state": connection.get("pairingState"),
        "product_type": hardware.get("productType"),
        "reality": hardware.get("reality"),
        "transport_type": connection.get("transportType"),
        "tunnel_state": connection.get("tunnelState"),
        "unlocked": True,
        "unlocked_since_boot": True,
    }


def decode_profile(profile_uuid: str) -> tuple[Path, dict[str, Any]]:
    """Decode one installed profile without emitting its raw signed payload."""

    candidates = (
        PROFILE_DIRECTORY / f"{profile_uuid}.mobileprovision",
        PROFILE_DIRECTORY / f"{profile_uuid.upper()}.mobileprovision",
        PROFILE_DIRECTORY / f"{profile_uuid.lower()}.mobileprovision",
    )
    profile_path = next((path for path in candidates if path.is_file()), None)
    if profile_path is None:
        raise FileNotFoundError(f"installed provisioning profile {profile_uuid} not found")
    result = _run(
        [str(TOOLS["security"]), "cms", "-D", "-i", str(profile_path)],
        timeout=30.0,
    )
    return profile_path, plistlib.loads(result.stdout.encode("utf-8"))


def validate_profile(
    profile_uuid: str,
    *,
    team: str,
    bundle_identifier: str,
    device_udid: str,
) -> dict[str, Any]:
    """Require a live explicit development profile for the exact phone/app."""

    profile_path, profile = decode_profile(profile_uuid)
    entitlements = profile.get("Entitlements", {})
    expiration = profile.get("ExpirationDate")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    checks = {
        "uuid": str(profile.get("UUID", "")).lower() == profile_uuid.lower(),
        "team": team in profile.get("TeamIdentifier", []),
        "bundle": entitlements.get("application-identifier")
        == f"{team}.{bundle_identifier}",
        "device": device_udid in profile.get("ProvisionedDevices", []),
        "development": entitlements.get("get-task-allow") is True,
        "not_expired": isinstance(expiration, datetime) and expiration > now,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"explicit profile failed checks: {failed}")
    return {
        "bundle_identifier": bundle_identifier,
        "exact_hardware_udid_present": True,
        "expiration_utc": expiration.replace(tzinfo=timezone.utc).isoformat(),
        "installed_locally": profile_path.is_file(),
        "name": profile.get("Name"),
        "team": team,
        "uuid": profile.get("UUID"),
    }


def profile_signing_identity(profile_uuid: str) -> str:
    """Return the embedded certificate fingerprint backed by a local key."""

    _, profile = decode_profile(profile_uuid)
    fingerprints = [
        hashlib.sha1(certificate).hexdigest().upper()
        for certificate in profile.get("DeveloperCertificates", [])
    ]
    identities = _run(
        [str(TOOLS["security"]), "find-identity", "-v", "-p", "codesigning"],
        timeout=30.0,
    ).stdout
    matches = [fingerprint for fingerprint in fingerprints if fingerprint in identities]
    if len(matches) != 1:
        raise RuntimeError("profile must contain exactly one usable local signing identity")
    return matches[0]


def _pair_spec(name: str) -> PairSpec:
    """Return one registered pair or fail before any device or artifact action."""

    try:
        return PAIR_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"unregistered A17 pair: {name}") from error


def _source_package_paths(pair: PairSpec) -> list[tuple[str, Path, str]]:
    """Return the complete GPU oracle package set plus one renamed fixed twin."""

    paths = [
        ("embedding", STAGE1_ROOT / "embedding.mlpackage", "embedding"),
        ("lm_head", STAGE1_ROOT / "lm_head.mlpackage", "lm_head"),
    ]
    for index in range(13):
        matches = sorted((STAGE1_ROOT / "segments").glob(f"segment_{index:02d}_*.mlpackage"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one Stage 1 segment at index {index}")
        paths.append((matches[0].stem, matches[0], matches[0].stem))
    paths.append(
        (
            f"{pair.name}_fixed512",
            FIXED_PAIR_ROOT / f"{pair.name}.mlpackage",
            f"{pair.name}_fixed512",
        )
    )
    return paths


def _working_tree_revision() -> str:
    """Bind a build to HEAD plus the complete tracked/untracked source diff."""

    head = _run(["/usr/bin/git", "rev-parse", "HEAD"]).stdout.strip()
    status = _run(["/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"]).stdout
    if not status:
        return head
    diff = _run(["/usr/bin/git", "diff", "--binary", "HEAD"]).stdout.encode()
    untracked = []
    for line in status.splitlines():
        if line.startswith("?? "):
            path = REPO_ROOT / line[3:]
            if path.is_file():
                untracked.append(line[3:].encode() + b"\0" + path.read_bytes())
    fingerprint = hashlib.sha256(diff + b"".join(untracked)).hexdigest()
    return f"{head}+worktree:{fingerprint}"


def _dispatch_paths(pair: PairSpec, directory: Path) -> dict[str, Path]:
    """Return the two exact final dispatch paths for one pair."""

    return {
        "CPU_AND_GPU": directory / f"{pair.name}.cpu_and_gpu.json",
        "CPU_AND_NE": directory / f"{pair.name}.cpu_and_ne.json",
    }


def validate_dispatch_payload(
    payload: dict[str, Any],
    pair: PairSpec,
    policy: str,
    *,
    expected_os_build: str | None = None,
) -> dict[str, Any]:
    """Apply the frozen same-phone placement gate to one dispatch payload."""

    if payload.get("pair") != pair.name or payload.get("policy") != policy:
        raise ValueError("dispatch pair or policy drifted")
    if payload.get("package_sha256") != pair.fixed_sha256:
        raise ValueError("dispatch fixed-twin hash drifted")
    device = payload.get("device", {})
    if (
        device.get("coredevice_identifier") != DEFAULT_DEVICE_ID
        or device.get("hardware_udid") != DEFAULT_DEVICE_UDID
        or device.get("product_type") != DEFAULT_PRODUCT_TYPE
    ):
        raise ValueError("dispatch device identity drifted")
    device_os_build = device.get("os_build")
    if not isinstance(device_os_build, str) or not device_os_build:
        raise ValueError("dispatch device OS build is missing")
    if expected_os_build is not None and device_os_build != expected_os_build:
        raise ValueError("dispatch device OS build drifted")
    plan = payload.get("compute_plan", {})
    provenance = plan.get("provenance", {})
    if provenance.get("where") != "device":
        raise ValueError("dispatch is not physical-device evidence")
    if provenance.get("compute_units") != policy:
        raise ValueError("dispatch compute units drifted")
    if provenance.get("package_sha256") != pair.fixed_sha256:
        raise ValueError("compute-plan package hash drifted")
    if provenance.get("device_identifier") != DEFAULT_DEVICE_ID:
        raise ValueError("compute-plan CoreDevice identifier drifted")
    if provenance.get("device_udid") != DEFAULT_DEVICE_UDID:
        raise ValueError("compute-plan UDID drifted")
    if provenance.get("device_product_type") != DEFAULT_PRODUCT_TYPE:
        raise ValueError("compute-plan product type drifted")
    if provenance.get("device_os_build") != device_os_build:
        raise ValueError("compute-plan OS build drifted")
    total = plan.get("total_costed_ops")
    histogram = plan.get("preferred_device_histogram", {})
    per_op_type = plan.get("per_op_type", {})
    if not isinstance(total, int) or total <= 0:
        raise ValueError("dispatch has no costed operations")
    if policy == "CPU_AND_GPU":
        if histogram.get("neuralEngine", 0) != 0:
            raise ValueError("GPU control contains Neural Engine preferred operations")
    else:
        ane_count = histogram.get("neuralEngine", 0)
        if ane_count / total < 0.80:
            raise ValueError("ANE candidate misses the 80% admission gate")
        heavy = {
            name: counts
            for name, counts in per_op_type.items()
            if name.endswith(".conv") or name.endswith(".matmul")
        }
        if not heavy:
            raise ValueError("ANE candidate reports no convolution or matmul operations")
        for name, counts in heavy.items():
            if counts.get("neuralEngine", 0) != sum(counts.values()):
                raise ValueError(f"heavy operation {name} is not wholly ANE-preferred")
    return {
        "ane_fraction": histogram.get("neuralEngine", 0) / total,
        "passed": True,
        "total_costed_ops": total,
    }


def stage_resources(
    pair_name: str,
    *,
    dispatch_directory: Path | None,
    build_only: bool,
) -> dict[str, Any]:
    """Copy and verify the exact app resources for one registered pair."""

    pair = _pair_spec(pair_name)
    if build_only and dispatch_directory is not None:
        raise ValueError("build-only staging cannot accept dispatch evidence")
    if not build_only and dispatch_directory is None:
        raise ValueError("device-run staging requires validated dispatch evidence")
    if GENERATED_RESOURCES.exists():
        shutil.rmtree(GENERATED_RESOURCES)
    packages_directory = GENERATED_RESOURCES / "Packages"
    packages_directory.mkdir(parents=True)
    artifacts = []
    for name, source, resource_name in _source_package_paths(pair):
        if not source.is_dir():
            raise FileNotFoundError(source)
        source_hash = package_tree_sha256(source)
        if name == f"{pair.name}_fixed512" and source_hash != pair.fixed_sha256:
            raise ValueError(f"{pair.name} frozen package hash mismatch")
        destination = packages_directory / f"{resource_name}.mlpackage"
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        if package_tree_sha256(destination) != source_hash:
            raise ValueError(f"staged package hash mismatch: {name}")
        artifacts.append(
            {
                "name": name,
                "path": str(source.relative_to(REPO_ROOT)),
                "resourceName": resource_name,
                "sha256": source_hash,
            }
        )
    input_source = STAGE1_ROOT / "g1a_input_512.json"
    input_payload = json.loads(input_source.read_text())
    if (
        input_payload.get("bucket") != REGISTERED_BUCKET
        or input_payload.get("prompt") != REGISTERED_PROMPT
        or len(input_payload.get("token_ids", [])) != REGISTERED_BUCKET
    ):
        raise ValueError("Stage 1 input contract drifted")
    shutil.copy2(input_source, GENERATED_RESOURCES / "g1a_input_512.json")

    dispatch_references: list[dict[str, Any]] = []
    if dispatch_directory is not None:
        for policy, path in _dispatch_paths(pair, dispatch_directory).items():
            payload = read_envelope(path, DISPATCH_SCHEMA)
            validate_dispatch_payload(payload, pair, policy)
            dispatch_references.append(
                {
                    "computeUnits": policy,
                    "evidenceSha256": file_sha256(path),
                    "path": str(path.relative_to(A17_ROOT)),
                    "packageSha256": pair.fixed_sha256,
                }
            )
    manifest = {
        "appRevision": _working_tree_revision(),
        "artifacts": artifacts,
        "deviceIdentifier": DEFAULT_DEVICE_ID,
        "dispatch": dispatch_references,
        "fixedPairResourceName": f"{pair.name}_fixed512",
        "hardwareUDID": DEFAULT_DEVICE_UDID,
        "inputResourceName": "g1a_input_512",
        "pair": pair.name,
        "productType": DEFAULT_PRODUCT_TYPE,
        "resultFilename": f"a17-parity-{pair.name}.json",
        "schemaVersion": 1,
    }
    manifest_path = GENERATED_RESOURCES / "a17-parity-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _profile_uuid_from_arguments(args: argparse.Namespace) -> str:
    """Require the explicit profile UUID from CLI or the documented env name."""

    value = getattr(args, "provisioning_profile_uuid", None) or os.environ.get(
        "LFM2_A17_PROFILE_UUID"
    )
    if not value:
        raise ValueError("LFM2_A17_PROFILE_UUID or --provisioning-profile-uuid is required")
    return value


def _build_project(
    pair_name: str,
    profile_uuid: str,
    *,
    dispatch_directory: Path | None,
    build_only: bool,
) -> tuple[Path, dict[str, Any]]:
    """Stage, generate, and sign one Release app without installing it."""

    device = live_device_snapshot(DEFAULT_DEVICE_ID, DEFAULT_DEVICE_UDID)
    profile = validate_profile(
        profile_uuid,
        team=DEFAULT_TEAM,
        bundle_identifier=DEFAULT_BUNDLE_ID,
        device_udid=DEFAULT_DEVICE_UDID,
    )
    signing_identity = profile_signing_identity(profile_uuid)
    manifest = stage_resources(
        pair_name,
        dispatch_directory=dispatch_directory,
        build_only=build_only,
    )
    if not TOOLS["xcodegen"].is_file():
        raise FileNotFoundError("/opt/homebrew/bin/xcodegen is required")
    _run(
        [str(TOOLS["xcodegen"]), "generate", "--spec", "project.yml"],
        cwd=IOS_ROOT,
        timeout=120.0,
    )
    derived = IOS_ROOT / "DerivedData"
    _run(
        [
            str(TOOLS["xcodebuild"]),
            "-project",
            "LFM2A17Parity.xcodeproj",
            "-scheme",
            "LFM2A17Parity",
            "-configuration",
            "Release",
            "-destination",
            "generic/platform=iOS",
            "-derivedDataPath",
            str(derived),
            "CODE_SIGN_STYLE=Manual",
            f"DEVELOPMENT_TEAM={DEFAULT_TEAM}",
            f"PRODUCT_BUNDLE_IDENTIFIER={DEFAULT_BUNDLE_ID}",
            f"CC={(REPO_ROOT / 'scripts/xcode_clang_probe_wrapper.zsh').resolve()}",
            f"CODE_SIGN_IDENTITY={signing_identity}",
            f"PROVISIONING_PROFILE={profile_uuid}",
            "PROVISIONING_PROFILE_SPECIFIER=",
            "build",
        ],
        cwd=IOS_ROOT,
        timeout=7200.0,
    )
    apps = sorted(derived.glob("Build/Products/Release-iphoneos/LFM2A17Parity.app"))
    if len(apps) != 1:
        raise FileNotFoundError("Release iPhone app was not produced")
    app = apps[0]
    embedded = app / "embedded.mobileprovision"
    if not embedded.is_file():
        raise FileNotFoundError("signed app contains no embedded profile")
    signature = _run(["/usr/bin/codesign", "-vvv", str(app)], timeout=60.0)
    if signature.returncode != 0:
        raise RuntimeError("codesign verification failed")
    provenance = {
        "app_bundle_identifier": DEFAULT_BUNDLE_ID,
        "app_revision": manifest["appRevision"],
        "build_configuration": "Release",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "launch_authorized": not build_only,
        "pair": pair_name,
        "profile": profile,
        "signed_app_path": str(app.relative_to(REPO_ROOT)),
    }
    write_envelope(A17_ROOT / "app-build-provenance.json", "a17-parity-build-v1", provenance)
    return app, provenance


def run_readiness(args: argparse.Namespace) -> int:
    """Refresh nonsensitive live readiness while retaining Phase 0 portal proof."""

    profile_uuid = _profile_uuid_from_arguments(args)
    device = live_device_snapshot(args.device_id, args.device_udid)
    profile = validate_profile(
        profile_uuid,
        team=args.development_team,
        bundle_identifier=args.bundle_identifier,
        device_udid=args.device_udid,
    )
    previous = read_envelope(args.out, "a17-parity-device-readiness-v1")
    portal = previous.get("portal", {})
    if (
        portal.get("device_status") != "ENABLED"
        or portal.get("exact_hardware_udid_registered") is not True
    ):
        raise ValueError("Phase 0 portal registration evidence is absent")
    payload = {
        "app": {
            "bundle_identifier": args.bundle_identifier,
            "development_team": args.development_team,
        },
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "portal": portal,
        "provisioning_profile": profile,
        "readiness": True,
    }
    write_envelope(args.out, "a17-parity-device-readiness-v1", payload)
    print(json.dumps({"out": str(args.out), "readiness": True}, sort_keys=True))
    return 0


def run_stage_resources(args: argparse.Namespace) -> int:
    """CLI wrapper for deterministic ignored resource staging."""

    manifest = stage_resources(
        args.pair,
        dispatch_directory=args.dispatch_dir,
        build_only=args.build_only,
    )
    print(
        json.dumps(
            {
                "artifact_count": len(manifest["artifacts"]),
                "dispatch_count": len(manifest["dispatch"]),
                "pair": manifest["pair"],
            },
            sort_keys=True,
        )
    )
    return 0


def run_build(args: argparse.Namespace) -> int:
    """Build and sign the Phase 1 app without installing or launching it."""

    profile_uuid = _profile_uuid_from_arguments(args)
    app, _ = _build_project(
        args.pair,
        profile_uuid,
        dispatch_directory=None,
        build_only=True,
    )
    print(json.dumps({"app": str(app), "installed": False, "pair": args.pair}, sort_keys=True))
    return 0


def run_dispatch(args: argparse.Namespace) -> int:
    """Capture and freeze the two same-phone compute plans for one pair."""

    pair = _pair_spec(args.pair)
    _prior_pair_payloads(pair, A17_ROOT / "pairs")
    expected_directory = A17_ROOT / "dispatch"
    if args.out_dir.resolve() != expected_directory.resolve():
        raise ValueError("dispatch output must use the frozen A17 evidence directory")
    profile_uuid = _profile_uuid_from_arguments(args)
    signing_identity = profile_signing_identity(profile_uuid)
    package = FIXED_PAIR_ROOT / f"{pair.name}.mlpackage"
    if package_tree_sha256(package) != pair.fixed_sha256:
        raise ValueError("fixed-pair package hash mismatch before dispatch")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_paths = _dispatch_paths(pair, args.out_dir)
    existing = [path.is_file() for path in final_paths.values()]
    if any(existing):
        if not all(existing):
            raise RuntimeError("partial dispatch evidence exists; preserve it for audit")
        device = live_device_snapshot(args.device_id, DEFAULT_DEVICE_UDID)
        for policy, path in final_paths.items():
            validate_dispatch_payload(
                read_envelope(path, DISPATCH_SCHEMA),
                pair,
                policy,
                expected_os_build=device["os_build"],
            )
        print(json.dumps({"dispatch": "PASS", "pair": pair.name, "reused": True}, sort_keys=True))
        return 0
    pending: list[tuple[Path, dict[str, Any]]] = []
    for policy, final_path in final_paths.items():
        device = live_device_snapshot(args.device_id, DEFAULT_DEVICE_UDID)
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "plan.json"
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/dump_device_compute_plan.py"),
                    "--package",
                    str(package),
                    "--compute-units",
                    policy,
                    "--device-name",
                    DEFAULT_DEVICE_NAME,
                    "--device-type",
                    "iphone",
                    "--development-team",
                    DEFAULT_TEAM,
                    "--bundle-identifier",
                    DEFAULT_BUNDLE_ID,
                    "--provisioning-profile-uuid",
                    profile_uuid,
                    "--code-sign-identity",
                    signing_identity,
                    "--out",
                    str(raw_path),
                ],
                timeout=7200.0,
            )
            raw = json.loads(raw_path.read_text())
        payload = {
            "compute_plan": raw,
            "device": device,
            "package_sha256": pair.fixed_sha256,
            "pair": pair.name,
            "policy": policy,
        }
        payload["gate"] = validate_dispatch_payload(
            payload, pair, policy, expected_os_build=device["os_build"]
        )
        pending.append((final_path, payload))
    for final_path, payload in pending:
        write_envelope(final_path, DISPATCH_SCHEMA, payload)
    print(json.dumps({"dispatch": "PASS", "pair": pair.name, "reused": False}, sort_keys=True))
    return 0


def _contains_performance_field(value: Any) -> bool:
    """Return true if a result schema contains any prohibited metric key."""

    prohibited = ("timing", "duration", "latency", "throughput", "energy", "elapsed")
    if isinstance(value, dict):
        return any(
            any(fragment in str(key).lower() for fragment in prohibited)
            or _contains_performance_field(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_performance_field(child) for child in value)
    return False


def _require_finite_mapping(values: Any, expected_names: Iterable[str]) -> dict[str, float]:
    """Validate one complete per-output finite-error mapping."""

    expected = set(expected_names)
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("per-output result names drifted")
    normalized = {name: float(value) for name, value in values.items()}
    if not all(math.isfinite(value) and value >= 0.0 for value in normalized.values()):
        raise ValueError("per-output result contains NaN, infinity, or negative error")
    return normalized


def validate_pair_result_payload(
    payload: dict[str, Any],
    pair: PairSpec,
    *,
    dispatch_paths: dict[str, Path] | None = None,
    expected_app_revision: str | None = None,
    expected_os_build: str | None = None,
    not_before_utc: str | None = None,
) -> str:
    """Recompute the complete six-row result verdict and reject schema drift."""

    if _contains_performance_field(payload):
        raise ValueError("A17 result contains a prohibited performance field")
    if payload.get("schema_version") != 1:
        raise ValueError("app result schema drifted")
    if payload.get("pair") != pair.name:
        raise ValueError("app result pair drifted")
    if payload.get("bucket") != REGISTERED_BUCKET or payload.get("prompt") != REGISTERED_PROMPT:
        raise ValueError("app result input drifted")
    if payload.get("precision") != "fp16":
        raise ValueError("app result precision drifted")
    if float(payload.get("maximum_absolute_tolerance", math.nan)) != MAXIMUM_ABSOLUTE_TOLERANCE:
        raise ValueError("app result tolerance drifted")
    if payload.get("performance_samples_recorded") != 0:
        raise ValueError("app result recorded a performance sample")
    device = payload.get("device", {})
    if (
        device.get("coredevice_identifier") != DEFAULT_DEVICE_ID
        or device.get("hardware_udid") != DEFAULT_DEVICE_UDID
        or device.get("product_type") != DEFAULT_PRODUCT_TYPE
    ):
        raise ValueError("app result device identity drifted")
    if expected_os_build is not None and device.get("os_build") != expected_os_build:
        raise ValueError("app result OS build drifted")
    app = payload.get("app", {})
    if app.get("bundle_identifier") != DEFAULT_BUNDLE_ID:
        raise ValueError("signed app bundle identifier drifted")
    if expected_app_revision is not None and app.get("revision") != expected_app_revision:
        raise ValueError("signed app revision drifted")
    captured_text = payload.get("captured_at_utc")
    if not isinstance(captured_text, str):
        raise ValueError("app result completion timestamp is missing")
    try:
        captured = datetime.fromisoformat(captured_text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("app result completion timestamp is invalid") from error
    if captured.tzinfo is None:
        raise ValueError("app result completion timestamp lacks a timezone")
    if not_before_utc is not None:
        not_before = datetime.fromisoformat(not_before_utc.replace("Z", "+00:00"))
        if captured < not_before:
            raise ValueError("app result predates the signed build")
    if captured > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("app result completion timestamp is in the future")
    checkpoint = payload.get("checkpoint", {})
    if (
        checkpoint.get("repository") != "LiquidAI/LFM2.5-350M"
        or checkpoint.get("revision")
        != "b9d6e4e2d75f440b12a2b4d731c808004ecbbd89"
    ):
        raise ValueError("app result checkpoint identity drifted")
    fixed = next(
        (
            artifact
            for artifact in payload.get("artifacts", [])
            if artifact.get("name") == f"{pair.name}_fixed512"
        ),
        None,
    )
    if fixed is None or fixed.get("sha256") != pair.fixed_sha256:
        raise ValueError("app result fixed package hash drifted")
    references = payload.get("dispatch", [])
    if {reference.get("compute_units") for reference in references} != {
        "CPU_AND_GPU",
        "CPU_AND_NE",
    }:
        raise ValueError("app result dispatch references drifted")
    if any(reference.get("package_sha256") != pair.fixed_sha256 for reference in references):
        raise ValueError("app dispatch reference package drifted")
    if dispatch_paths is not None:
        for policy, path in dispatch_paths.items():
            reference = next(
                item for item in references if item.get("compute_units") == policy
            )
            if reference.get("evidence_sha256") != file_sha256(path):
                raise ValueError("app dispatch evidence hash drifted")
            validate_dispatch_payload(
                read_envelope(path, DISPATCH_SCHEMA),
                pair,
                policy,
                expected_os_build=device.get("os_build"),
            )
    oracle_token = payload.get("oracle", {}).get("top_token")
    oracle_passed = payload.get("oracle", {}).get("passed")
    if oracle_passed is not (oracle_token == EXPECTED_TOKEN):
        raise ValueError("oracle pass flag drifted")
    rows = payload.get("comparisons")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("A17 result requires exactly six comparisons")
    computed_passes = []
    for index, (row, order) in enumerate(zip(rows, COMPARISON_ORDERS, strict=True)):
        if row.get("repetition") != index:
            raise ValueError("comparison repetition index drifted")
        expected_kind = "first_use" if index == 0 else "warmed"
        if row.get("repetition_kind") != expected_kind or row.get("candidate_order") != order:
            raise ValueError("comparison kind or order drifted")
        source = _require_finite_mapping(
            row.get("source_twin_max_abs_by_output"), pair.outputs
        )
        candidate = _require_finite_mapping(
            row.get("gpu_ane_max_abs_by_output"), pair.outputs
        )
        source_maximum = float(row.get("source_twin_max_abs", math.nan))
        candidate_maximum = float(row.get("pair_output_max_abs", math.nan))
        suffix = float(row.get("suffix_logit_max_abs", math.nan))
        if not all(math.isfinite(value) and value >= 0.0 for value in (source_maximum, candidate_maximum, suffix)):
            raise ValueError("comparison aggregate contains NaN, infinity, or negative error")
        if source_maximum != max(source.values()) or candidate_maximum != max(candidate.values()):
            raise ValueError("comparison aggregate maximum drifted")
        computed = (
            source_maximum <= MAXIMUM_ABSOLUTE_TOLERANCE
            and candidate_maximum <= MAXIMUM_ABSOLUTE_TOLERANCE
            and suffix <= MAXIMUM_ABSOLUTE_TOLERANCE
            and row.get("oracle_token") == EXPECTED_TOKEN
            and row.get("gpu_token") == EXPECTED_TOKEN
            and row.get("ane_token") == EXPECTED_TOKEN
        )
        if row.get("passed") is not computed:
            raise ValueError("comparison pass flag drifted")
        computed_passes.append(computed)
    verdict = "PAIR PASS" if all(computed_passes) and oracle_token == EXPECTED_TOKEN else "KILL"
    if payload.get("verdict") != verdict:
        raise ValueError("app result verdict drifted")
    return verdict


def authorized_next_pair(completed_payloads: list[dict[str, Any]]) -> str | None:
    """Return the only next pair allowed by the frozen stop-first state machine."""

    if len(completed_payloads) > len(PAIR_SPECS):
        raise ValueError("too many completed pair results")
    for index, payload in enumerate(completed_payloads):
        pair = PAIR_SPECS[index]
        if validate_pair_result_payload(payload, pair) != "PAIR PASS":
            return None
    return PAIR_SPECS[len(completed_payloads)].name if len(completed_payloads) < len(PAIR_SPECS) else None


def _prior_pair_payloads(pair: PairSpec, result_directory: Path) -> list[dict[str, Any]]:
    """Load the exact passing predecessors required before a later pair."""

    index = PAIR_SPECS.index(pair)
    completed = []
    for prior in PAIR_SPECS[:index]:
        path = result_directory / f"{prior.name}.result.json"
        payload = read_envelope(path, RESULT_SCHEMA)
        validate_pair_result_payload(
            payload,
            prior,
            dispatch_paths=_dispatch_paths(prior, A17_ROOT / "dispatch"),
        )
        completed.append(payload)
    if authorized_next_pair(completed) != pair.name:
        raise ValueError(f"state machine does not authorize {pair.name}")
    return completed


def remove_existing_probe_app(device_id: str) -> bool:
    """Remove only the registered probe app so a prior result cannot survive."""

    if device_id != DEFAULT_DEVICE_ID:
        raise ValueError("probe cleanup requires the frozen CoreDevice identifier")
    inventory = _devicectl_json(
        [
            "device",
            "info",
            "apps",
            "--device",
            device_id,
            "--bundle-id",
            DEFAULT_BUNDLE_ID,
        ]
    )
    apps = inventory.get("apps")
    if not isinstance(apps, list):
        raise RuntimeError("could not inspect existing parity app")
    if not apps:
        return False
    if len(apps) != 1:
        raise RuntimeError("exact parity-app lookup returned multiple applications")
    _devicectl_json(
        [
            "device",
            "uninstall",
            "app",
            "--device",
            device_id,
            DEFAULT_BUNDLE_ID,
        ],
        timeout=300.0,
    )
    remaining = _devicectl_json(
        [
            "device",
            "info",
            "apps",
            "--device",
            device_id,
            "--bundle-id",
            DEFAULT_BUNDLE_ID,
        ]
    ).get("apps")
    if remaining != []:
        raise RuntimeError("existing parity app data was not removed")
    return True


def run_pair(args: argparse.Namespace) -> int:
    """Build, install, launch, retrieve, and validate one authorized pair."""

    pair = _pair_spec(args.pair)
    expected_result = A17_ROOT / "pairs" / f"{pair.name}.result.json"
    if args.out.resolve() != expected_result.resolve():
        raise ValueError("pair output must use the frozen A17 evidence path")
    if args.out.exists():
        raise FileExistsError("pair result already exists and is immutable")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    profile_uuid = _profile_uuid_from_arguments(args)
    _prior_pair_payloads(pair, A17_ROOT / "pairs")
    dispatch_paths = _dispatch_paths(pair, A17_ROOT / "dispatch")
    preflight_device = live_device_snapshot(args.device_id, DEFAULT_DEVICE_UDID)
    for policy, path in dispatch_paths.items():
        validate_dispatch_payload(
            read_envelope(path, DISPATCH_SCHEMA),
            pair,
            policy,
            expected_os_build=preflight_device["os_build"],
        )
    app, build = _build_project(
        pair.name,
        profile_uuid,
        dispatch_directory=A17_ROOT / "dispatch",
        build_only=False,
    )
    live_device_snapshot(args.device_id, DEFAULT_DEVICE_UDID)
    remove_existing_probe_app(args.device_id)
    live_device_snapshot(args.device_id, DEFAULT_DEVICE_UDID)
    _run(
        [
            str(TOOLS["xcrun"]),
            "devicectl",
            "device",
            "install",
            "app",
            "--device",
            args.device_id,
            str(app),
        ],
        timeout=1800.0,
    )
    live_device_snapshot(args.device_id, DEFAULT_DEVICE_UDID)
    _run(
        [
            str(TOOLS["xcrun"]),
            "devicectl",
            "device",
            "process",
            "launch",
            "--terminate-existing",
            "--device",
            args.device_id,
            DEFAULT_BUNDLE_ID,
        ],
        timeout=120.0,
    )
    remote = f"Documents/a17-parity-{pair.name}.json"
    raw = args.out.with_name(f".{args.out.name}.device")
    if raw.exists():
        raw.unlink()
    deadline = time.monotonic() + args.poll_timeout_seconds
    last_error = ""
    result_device: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        result_device = live_device_snapshot(args.device_id, DEFAULT_DEVICE_UDID)
        result = _run(
            [
                str(TOOLS["xcrun"]),
                "devicectl",
                "device",
                "copy",
                "from",
                "--device",
                args.device_id,
                "--domain-type",
                "appDataContainer",
                "--domain-identifier",
                DEFAULT_BUNDLE_ID,
                "--source",
                remote,
                "--destination",
                str(raw),
            ],
            timeout=120.0,
            check=False,
        )
        if result.returncode == 0 and raw.is_file():
            break
        last_error = "\n".join(part for part in (result.stdout, result.stderr) if part)
        time.sleep(args.poll_interval_seconds)
    else:
        raise RuntimeError(f"device result was not retrieved: {last_error}")
    payload = json.loads(raw.read_text())
    if payload.get("status") == "INCONCLUSIVE":
        write_envelope(args.out, RESULT_SCHEMA, payload)
        raw.unlink(missing_ok=True)
        print(json.dumps({"pair": pair.name, "verdict": "INCONCLUSIVE"}, sort_keys=True))
        return 2
    if result_device is None:
        raise RuntimeError("device identity was not captured with the result")
    validate_pair_result_payload(
        payload,
        pair,
        dispatch_paths=dispatch_paths,
        expected_app_revision=build["app_revision"],
        expected_os_build=result_device["os_build"],
        not_before_utc=build["captured_at_utc"],
    )
    write_envelope(args.out, RESULT_SCHEMA, payload)
    raw.unlink(missing_ok=True)
    print(json.dumps({"pair": pair.name, "verdict": payload["verdict"]}, sort_keys=True))
    return 0


def run_aggregate(args: argparse.Namespace) -> int:
    """Create a hash-reference-only aggregate after all three pairs pass."""

    if args.root.resolve() != A17_ROOT.resolve():
        raise ValueError("aggregate root must use the frozen A17 evidence directory")
    if args.out.resolve() != (A17_ROOT / "aggregate.json").resolve():
        raise ValueError("aggregate output must use the frozen A17 evidence path")
    if args.out.exists():
        raise FileExistsError("aggregate result already exists and is immutable")
    pairs = {}
    completed = []
    for pair in PAIR_SPECS:
        path = args.root / "pairs" / f"{pair.name}.result.json"
        payload = read_envelope(path, RESULT_SCHEMA)
        if validate_pair_result_payload(payload, pair) != "PAIR PASS":
            raise ValueError(f"aggregate cannot include failing pair {pair.name}")
        completed.append(payload)
        pairs[pair.name] = {"path": str(path.relative_to(args.root)), "sha256": file_sha256(path)}
    if authorized_next_pair(completed) is not None:
        raise ValueError("aggregate state machine did not reach terminal pass")
    dispatch = {}
    for pair in PAIR_SPECS:
        for policy, path in _dispatch_paths(pair, args.root / "dispatch").items():
            validate_dispatch_payload(read_envelope(path, DISPATCH_SCHEMA), pair, policy)
            dispatch[f"{pair.name}:{policy}"] = {
                "path": str(path.relative_to(args.root)),
                "sha256": file_sha256(path),
            }
    payload = {
        "dispatch": dispatch,
        "pair_results": pairs,
        "performance_samples_recorded": 0,
        "verdict": "A17 PARITY PASS",
    }
    write_envelope(args.out, "a17-parity-aggregate-v1", payload)
    print(json.dumps({"verdict": payload["verdict"]}, sort_keys=True))
    return 0


def _add_profile_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared explicit-profile option to one subcommand."""

    parser.add_argument("--provisioning-profile-uuid", default=None)


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed CLI without arbitrary pair or timing options."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    readiness.add_argument("--device-udid", default=DEFAULT_DEVICE_UDID)
    readiness.add_argument("--development-team", default=DEFAULT_TEAM)
    readiness.add_argument("--bundle-identifier", default=DEFAULT_BUNDLE_ID)
    _add_profile_argument(readiness)
    readiness.add_argument("--out", type=Path, required=True)
    readiness.set_defaults(handler=run_readiness)

    stage = subparsers.add_parser("stage-resources")
    stage.add_argument("--pair", choices=tuple(PAIR_BY_NAME), required=True)
    stage.add_argument("--dispatch-dir", type=Path)
    stage.add_argument("--build-only", action="store_true")
    stage.set_defaults(handler=run_stage_resources)

    build = subparsers.add_parser("build")
    build.add_argument("--pair", choices=(PAIR_SPECS[0].name,), required=True)
    _add_profile_argument(build)
    build.set_defaults(handler=run_build)

    dispatch = subparsers.add_parser("dispatch")
    dispatch.add_argument("--pair", choices=tuple(PAIR_BY_NAME), required=True)
    dispatch.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    _add_profile_argument(dispatch)
    dispatch.add_argument("--out-dir", type=Path, required=True)
    dispatch.set_defaults(handler=run_dispatch)

    run = subparsers.add_parser("run")
    run.add_argument("--pair", choices=tuple(PAIR_BY_NAME), required=True)
    run.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    _add_profile_argument(run)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--poll-timeout-seconds", type=float, default=7200.0)
    run.add_argument("--poll-interval-seconds", type=float, default=10.0)
    run.set_defaults(handler=run_pair)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    aggregate.set_defaults(handler=run_aggregate)
    return parser


def main() -> int:
    """Parse and execute exactly one frozen orchestration command."""

    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
