import tempfile

import pytest

from flexeval.core.prompt_template.jinja2 import Jinja2PromptTemplate
from flexeval.utils import instantiate_from_config
from tests.dummy_modules.generation_dataset import DummyGenerationDataset


def test_jinja2_template() -> None:
    dummy_dataset = DummyGenerationDataset()

    inputs = dummy_dataset[0].inputs
    assert Jinja2PromptTemplate(template="{{ text }}").embed_inputs(inputs) == inputs["text"]


def test_jinja2_template_with_template_path() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonnet") as f:
        f.write("""Hello World!""")
        f.flush()

        prompt_template = Jinja2PromptTemplate(template_path=f.name)

    assert prompt_template.embed_inputs({}) == "Hello World!"


def test_if_jinja2_template_raises_errors_with_conflicting_args() -> None:
    with pytest.raises(ValueError):
        Jinja2PromptTemplate(template="Hello World!", template_path="some_path")


def test_if_jinja2_template_keep_trailing_newline() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonnet") as f:
        f.write("""
{
  class_path: 'Jinja2PromptTemplate',
  init_args: {
    template: |||
      Hello World!
    |||
  }
}
""")
        f.flush()

        prompt_template = instantiate_from_config(f.name)

        # note that the text block (||| ... |||) in the jsonnet file adds a newline at the end
        assert prompt_template.embed_inputs({}) == "Hello World!\n"
