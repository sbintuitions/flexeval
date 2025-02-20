from __future__ import annotations

import os
import subprocess
import tempfile


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
        "--reward_model", "PairwiseJudgeRewardModel",
        "--reward_model.language_model", "tests.dummy_modules.DummyRewardLanguageModel",
        "--reward_model.prompt_template", "Jinja2PromptTemplate",
        "--reward_model.prompt_template.template", template,
        "--reward_model.gen_kwargs", "{}",
        "--eval_dataset", "tests.dummy_modules.DummyRewardBenchDataset",
        "--batch_size", "1",
    ]

    with tempfile.TemporaryDirectory() as f:
        result = subprocess.run([*command, "--save_dir", f], check=False)
        assert result.returncode == os.EX_OK
