from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from flexeval.core.result_recorder.local_recorder import CONFIG_FILE_NAME

from .test_flexeval_lm import check_if_eval_results_are_correctly_saved


def test_cli() -> None:
    # fmt: off
    template = (
        "[User Question]"
        "{{ prompt }}"

        "[The Start of Assistant A's Answer]"
        "{{ answer_a }}"
        "[The End of Assistant A's Answer]"

        "[The Start of Assistant B's Answer]"
        "{{ answer_b }}"
        "[The End of Assistant B's Answer]"
    )
    command = [
        "flexeval_reward",
        "--language_model", "tests.dummy_modules.DummyRewardLanguageModel",
        "--eval_setup", "RewardEvalSetup",
        "--eval_setup.eval_dataset", "tests.dummy_modules.DummyRewardBenchDataset",
        "--eval_setup.prompt_template", "Jinja2PromptTemplate",
        "--eval_setup.prompt_template.template", template,
        "--eval_setup.gen_kwargs", "{}",
        "--eval_setup.batch_size", "1",
    ]

    with tempfile.TemporaryDirectory() as f:
        result = subprocess.run([*command, "--save_dir", f], check=False)
        assert result.returncode == os.EX_OK
