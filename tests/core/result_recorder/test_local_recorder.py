import json
import tempfile
from pathlib import Path

import pytest

from flexeval.core.result_recorder.local_recorder import (
    CONFIG_FILE_NAME,
    METRIC_FILE_NAME,
    OUTPUTS_FILE_NAME,
    LocalRecorder,
)


@pytest.fixture()
def temp_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp_path:
        yield tmp_path


def test_record_config(temp_dir: str) -> None:
    local_recorder = LocalRecorder(temp_dir)

    config = {"param1": "value1", "param2": 42}
    local_recorder.record_config(config)

    config_file = Path(temp_dir) / CONFIG_FILE_NAME
    assert config_file.exists()

    with open(config_file) as f:
        saved_config = json.load(f)

    assert saved_config == config


def test_record_metrics(temp_dir: str) -> None:
    local_recorder = LocalRecorder(temp_dir)

    metrics = {"accuracy": 0.95, "f1_score": 0.92}
    local_recorder.record_metrics(metrics)

    metrics_file = Path(temp_dir) / METRIC_FILE_NAME
    assert metrics_file.exists()

    with open(metrics_file) as f:
        saved_metrics = json.load(f)

    assert saved_metrics == metrics


def test_record_model_outputs(temp_dir: str) -> None:
    local_recorder = LocalRecorder(temp_dir)

    outputs = [
        {"input": "test1", "output": "result1"},
        {"input": "test2", "output": "result2"},
    ]
    local_recorder.record_model_outputs(outputs)

    outputs_file = Path(temp_dir) / OUTPUTS_FILE_NAME
    assert outputs_file.exists()

    with open(outputs_file) as f:
        saved_outputs = [json.loads(line) for line in f]

    assert saved_outputs == outputs


def test_record_with_group(temp_dir: str) -> None:
    group = "test_group"
    config = {"param": "value"}
    metrics = {"score": 0.9}
    outputs = [{"input": "test", "output": "result"}]

    local_recorder = LocalRecorder(temp_dir)

    local_recorder.record_config(config, group=group)
    local_recorder.record_metrics(metrics, group=group)
    local_recorder.record_model_outputs(outputs, group=group)

    group_dir = Path(temp_dir) / group
    assert (group_dir / CONFIG_FILE_NAME).exists()
    assert (group_dir / METRIC_FILE_NAME).exists()
    assert (group_dir / OUTPUTS_FILE_NAME).exists()


def test_raise_error_with_file_exists(temp_dir: str) -> None:
    recorder_with_error = LocalRecorder(temp_dir)

    # Create initial recorder and files
    recorder = LocalRecorder(temp_dir)
    recorder.record_config({"initial": "config"})
    recorder.record_metrics({"initial": "metrics"})
    recorder.record_model_outputs([{"initial": "output"}])

    with pytest.raises(FileExistsError):
        recorder_with_error.record_config({"new": "config"})

    with pytest.raises(FileExistsError):
        recorder_with_error.record_metrics({"new": "metrics"})

    with pytest.raises(FileExistsError):
        recorder_with_error.record_model_outputs([{"new": "output"}])


def test_is_metrics_saved(temp_dir: str) -> None:
    local_recorder = LocalRecorder(temp_dir)

    assert not local_recorder.is_metrics_saved()

    metrics = {"accuracy": 0.95}
    local_recorder.record_metrics(metrics)

    assert local_recorder.is_metrics_saved()


def test_is_metrics_saved_with_group(temp_dir: str) -> None:
    group = "test_group"
    local_recorder = LocalRecorder(temp_dir)

    assert not local_recorder.is_metrics_saved(group=group)

    metrics = {"accuracy": 0.95}
    local_recorder.record_metrics(metrics, group=group)

    assert local_recorder.is_metrics_saved(group=group)
