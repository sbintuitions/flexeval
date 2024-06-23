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
        LocalRecorder(temp_dir)

    with pytest.raises(FileExistsError):
        recorder_with_error.record_config({"new": "config"})

    with pytest.raises(FileExistsError):
        recorder_with_error.record_metrics({"new": "metrics"})

    with pytest.raises(FileExistsError):
        recorder_with_error.record_model_outputs([{"new": "output"}])


def test_force_overwrite(temp_dir: str) -> None:
    # Create initial recorder and files
    recorder1 = LocalRecorder(temp_dir)
    recorder1.record_config({"initial": "config"})
    recorder1.record_metrics({"initial": "metrics"})
    recorder1.record_model_outputs([{"initial": "output"}])

    # Create new recorder with force=True
    recorder2 = LocalRecorder(temp_dir, force=True)

    new_config = {"new": "config"}
    new_metrics = {"new": "metrics"}
    new_outputs = [{"new": "output"}]

    # These should not raise exceptions
    recorder2.record_config(new_config)
    recorder2.record_metrics(new_metrics)
    recorder2.record_model_outputs(new_outputs)

    # Check that the new data has been written
    with open(Path(temp_dir) / CONFIG_FILE_NAME) as f:
        assert json.load(f) == new_config

    with open(Path(temp_dir) / METRIC_FILE_NAME) as f:
        assert json.load(f) == new_metrics

    with open(Path(temp_dir) / OUTPUTS_FILE_NAME) as f:
        assert json.loads(f.readline()) == new_outputs[0]
