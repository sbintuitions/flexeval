from __future__ import annotations

import json
from pathlib import Path

import pytest

from flexeval.core.attestation import (
    ATTESTATION_FILE_NAME,
    create_attestation,
    generate_key_pair,
    verify_attestation,
)


def _make_results(output_dir: Path) -> None:
    (output_dir / "config.json").write_text(json.dumps({"model": "test", "task": "demo"}))
    (output_dir / "metrics.json").write_text(json.dumps({"exact_match": 0.73}))
    (output_dir / "outputs.jsonl").write_text('{"input": "q", "output": "a"}\n')


@pytest.mark.parametrize("algorithm", ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"])
def test_attestation_round_trip(tmp_path: Path, algorithm: str) -> None:
    _make_results(tmp_path)
    public_key, private_key = generate_key_pair(algorithm)

    attestation = create_attestation(tmp_path, algorithm=algorithm, key_pair=(public_key, private_key))
    assert attestation["algorithm"] == algorithm
    assert set(attestation["files"]) == {"config.json", "metrics.json", "outputs.jsonl"}
    assert (tmp_path / ATTESTATION_FILE_NAME).exists()

    result = verify_attestation(tmp_path)
    assert result.valid, result.reason

    pinned = verify_attestation(tmp_path, expected_public_key=public_key)
    assert pinned.valid, pinned.reason


def test_detects_tampered_results(tmp_path: Path) -> None:
    _make_results(tmp_path)
    create_attestation(tmp_path)

    (tmp_path / "metrics.json").write_text(json.dumps({"exact_match": 0.99}))

    result = verify_attestation(tmp_path)
    assert not result.valid


def test_rejects_wrong_signer(tmp_path: Path) -> None:
    _make_results(tmp_path)
    create_attestation(tmp_path)

    other_public_key, _ = generate_key_pair()
    result = verify_attestation(tmp_path, expected_public_key=other_public_key)
    assert not result.valid


def test_requires_result_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        create_attestation(tmp_path)
