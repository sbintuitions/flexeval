from flexeval.core.prompt_template.jinja2 import Jinja2PromptTemplate
from tests.dummy_modules.generation_dataset import DummyGenerationDataset


def test_jinja2_template() -> None:
    dummy_dataset = DummyGenerationDataset()

    inputs = dummy_dataset[0].inputs
    assert Jinja2PromptTemplate(template="{{ text }}").embed_input(inputs) == inputs["text"]
