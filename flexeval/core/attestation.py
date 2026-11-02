"""Post-quantum attestation of evaluation results.

This module produces a tamper-evident attestation over the files that a
FlexEval run writes to its output directory (``config.json``, ``metrics.json``
and ``outputs.jsonl``). The attestation is an ML-DSA signature (FIPS 204) over
the SHA-256 digests of those files, so a third party can confirm that a set of
published evaluation results has not been altered since the run.

ML-DSA is a post-quantum signature scheme, so the attestation stays valid even
against an adversary with a quantum computer. The signing uses the optional
``dilithium-py`` dependency; install FlexEval with the ``attestation`` extra to
enable it.

Integrity is always checked (the recorded digests must match the files on
disk). Authenticity, that the attestation was produced by a specific party,
holds when the verifier pins the signer's public key via ``expected_public_key``
using a key the evaluator has published out of band.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

ATTESTATION_FILE_NAME = "attestation.json"
DEFAULT_ATTESTED_FILES = ("config.json", "metrics.json", "outputs.jsonl")
DEFAULT_ALGORITHM = "ML-DSA-65"
SUPPORTED_ALGORITHMS = ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87")


def _load_ml_dsa(algorithm: str) -> object:
    try:
        from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        msg = (
            "Post-quantum attestation requires the optional 'dilithium-py' dependency. "
            "Install FlexEval with the 'attestation' extra, e.g. `pip install flexeval[attestation]`."
        )
        raise ImportError(msg) from e

    table = {"ML-DSA-44": ML_DSA_44, "ML-DSA-65": ML_DSA_65, "ML-DSA-87": ML_DSA_87}
    if algorithm not in table:
        msg = f"Unsupported attestation algorithm: {algorithm}. Supported: {', '.join(SUPPORTED_ALGORITHMS)}"
        raise ValueError(msg)
    return table[algorithm]


def generate_key_pair(algorithm: str = DEFAULT_ALGORITHM) -> tuple[bytes, bytes]:
    """Generate an ML-DSA key pair. Returns ``(public_key, private_key)`` as raw bytes."""
    ml_dsa = _load_ml_dsa(algorithm)
    public_key, private_key = ml_dsa.keygen()
    return public_key, private_key


def _file_digests(output_dir: Path, attested_files: tuple[str, ...]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for name in attested_files:
        path = output_dir / name
        if path.exists():
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _signing_payload(algorithm: str, digests: dict[str, str]) -> bytes:
    # Deterministic canonical bytes over the algorithm and the per-file digests.
    body = {"algorithm": algorithm, "files": digests}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class VerificationResult:
    """The outcome of verifying an attestation."""

    valid: bool
    reason: str


def create_attestation(
    output_dir: str | PathLike[str],
    algorithm: str = DEFAULT_ALGORITHM,
    attested_files: tuple[str, ...] = DEFAULT_ATTESTED_FILES,
    key_pair: tuple[bytes, bytes] | None = None,
) -> dict[str, object]:
    """Create and write a post-quantum attestation for a result directory.

    Args:
        output_dir: The directory holding the run's result files.
        algorithm: One of ``ML-DSA-44``, ``ML-DSA-65`` or ``ML-DSA-87``.
        attested_files: File names to attest, if present in ``output_dir``.
        key_pair: An optional ``(public_key, private_key)`` pair. When omitted a
            fresh ephemeral pair is generated. To let others verify authenticity
            against a known identity, pass a stable key pair whose public key you
            have published.

    Returns:
        The attestation dictionary that was written to ``attestation.json``.
    """
    output_dir = Path(output_dir)
    ml_dsa = _load_ml_dsa(algorithm)

    digests = _file_digests(output_dir, attested_files)
    if not digests:
        msg = f"No result files to attest were found in {output_dir}"
        raise FileNotFoundError(msg)

    if key_pair is None:
        public_key, private_key = ml_dsa.keygen()
    else:
        public_key, private_key = key_pair

    payload = _signing_payload(algorithm, digests)
    signature = ml_dsa.sign(private_key, payload)

    attestation = {
        "version": 1,
        "algorithm": algorithm,
        "created_at": int(time.time()),
        "files": digests,
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }

    with open(output_dir / ATTESTATION_FILE_NAME, "w") as f:
        json.dump(attestation, f, indent=4)

    return attestation


def verify_attestation(
    output_dir: str | PathLike[str],
    expected_public_key: bytes | None = None,
) -> VerificationResult:
    """Verify the attestation written in ``output_dir``.

    The result files on disk are re-hashed and checked against the recorded
    digests, and the ML-DSA signature is verified. When ``expected_public_key``
    is given, the attestation's public key must match it, which establishes that
    the attestation came from that known signer.
    """
    output_dir = Path(output_dir)
    attestation_path = output_dir / ATTESTATION_FILE_NAME
    if not attestation_path.exists():
        return VerificationResult(False, f"No {ATTESTATION_FILE_NAME} found in {output_dir}")

    with open(attestation_path) as f:
        attestation = json.load(f)

    algorithm = attestation.get("algorithm")
    if algorithm not in SUPPORTED_ALGORITHMS:
        return VerificationResult(False, f"Unsupported or missing algorithm: {algorithm}")
    ml_dsa = _load_ml_dsa(algorithm)

    recorded = attestation.get("files", {})
    current = _file_digests(output_dir, tuple(recorded.keys()))
    if current != recorded:
        return VerificationResult(False, "Result files do not match the recorded digests")

    try:
        public_key = base64.b64decode(attestation["public_key"])
        signature = base64.b64decode(attestation["signature"])
    except (KeyError, ValueError):
        return VerificationResult(False, "Malformed attestation fields")

    if expected_public_key is not None and public_key != expected_public_key:
        return VerificationResult(False, "Attestation public key does not match the expected signer")

    payload = _signing_payload(algorithm, recorded)
    valid = ml_dsa.verify(public_key, payload, signature)
    return VerificationResult(valid, "Attestation is valid" if valid else "Signature verification failed")
