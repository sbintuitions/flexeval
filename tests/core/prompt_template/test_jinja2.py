import tempfile

import pytest

from flexeval.core.prompt_template.jinja2 import Jinja2PromptTemplate, instantiate_prompt_template_from_string
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


def test_instantiate_prompt_template_from_string_with_template_path() -> None:
    # Create a temporary Jinja2 template file
    template_content = "Hello, {{ name }}!"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja2") as temp_file:
        temp_file.write(template_content)
        temp_file.flush()
        temp_file_path = temp_file.name

        prompt_template = instantiate_prompt_template_from_string(temp_file_path)
        assert isinstance(prompt_template, Jinja2PromptTemplate)
        assert prompt_template.template == template_content


def test_instantiate_prompt_template_from_string_with_template_string() -> None:
    template_string = "Hello, {{ name }}!"
    prompt_template = instantiate_prompt_template_from_string(template_string)
    assert isinstance(prompt_template, Jinja2PromptTemplate)
    assert prompt_template.template == template_string
