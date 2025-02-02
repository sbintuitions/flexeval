from flexeval import ChatInstance
from flexeval.core.few_shot_generator.fixed import FixedFewShotGenerator


def test_fixed_fewshot_generator() -> None:
    instance = ChatInstance(
        messages=[{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]
    )
    generator = FixedFewShotGenerator(
        instance_class="ChatInstance",
        instance_params=[{"messages": instance.messages} for _ in range(5)],
    )
    assert generator() == [instance for _ in range(5)]
