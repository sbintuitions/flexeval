import pytest

from flexeval.core.few_shot_generator.rand import Dataset, RandomFewShotGenerator
from tests.dummy_modules.chat_dataset import DummyChatDataset
from tests.dummy_modules.generation_dataset import DummyGenerationDataset
from tests.dummy_modules.multiple_choice_dataset import DummyMultipleChoiceDataset


@pytest.mark.parametrize(
    ("num_shots", "dataset"),
    [
        (1, DummyGenerationDataset()),
        (2, DummyMultipleChoiceDataset()),
        (2, DummyChatDataset()),
    ],
)
def test_random_few_shot_generator(num_shots: int, dataset: Dataset) -> None:
    few_shot_sampler = RandomFewShotGenerator(
        dataset=dataset,
        num_shots=num_shots,
    )

    sampled_instances = few_shot_sampler()
    assert len(sampled_instances) == num_shots


@pytest.mark.parametrize(
    ("seed", "dataset"),
    [
        (42, DummyMultipleChoiceDataset()),
        (777, DummyGenerationDataset()),
        (0, DummyChatDataset()),
    ],
)
def test_if_random_seed_is_consistent(seed: int, dataset: Dataset) -> None:
    sampled_instances_1 = RandomFewShotGenerator(dataset=dataset, num_shots=2, seed=seed)()
    sampled_instances_2 = RandomFewShotGenerator(dataset=dataset, num_shots=2, seed=seed)()
    assert sampled_instances_1 == sampled_instances_2


@pytest.mark.parametrize(
    "dataset",
    [
        DummyMultipleChoiceDataset(),
        DummyGenerationDataset(),
        DummyChatDataset(),
    ],
)
def test_if_results_are_random(dataset: Dataset) -> None:
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=2)
    sampled_instances_1 = few_shot_generator()
    sampled_instances_2 = few_shot_generator()
    assert sampled_instances_1 != sampled_instances_2


@pytest.mark.parametrize(
    "dataset",
    [
        DummyMultipleChoiceDataset(),
        DummyGenerationDataset(),
        DummyChatDataset(),
    ],
)
def test_if_few_show_sampler_avoids_leak(dataset: Dataset) -> None:
    eval_inputs = dataset[0].inputs

    # First check if the eval_instance may be in the sampled instances with `num_trials_to_avoid_leak=0`
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=2, num_trials_to_avoid_leak=0)
    leak_found = False
    for _ in range(100):
        sampled_instances = few_shot_generator(eval_inputs=eval_inputs)
        if any(eval_inputs == instance.inputs for instance in sampled_instances):
            leak_found = True
            break
    assert leak_found

    # Now check if the eval_instance is not in the sampled instances with `num_trials_to_avoid_leak=10`
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=2, num_trials_to_avoid_leak=100)
    leak_found = False
    for _ in range(100):
        sampled_instances = few_shot_generator(eval_inputs=eval_inputs)
        if any(eval_inputs == instance.inputs for instance in sampled_instances):
            leak_found = True
            break
    assert not leak_found

    # Finally, check if the exception is raised when the resampling fails to avoid the leak
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=2, num_trials_to_avoid_leak=1)

    with pytest.raises(ValueError):  # noqa: PT012
        for _ in range(100):
            few_shot_generator(eval_inputs=eval_inputs)


@pytest.mark.parametrize(
    "dataset",
    [
        DummyMultipleChoiceDataset(),
        DummyGenerationDataset(),
        DummyChatDataset(),
    ],
)
def test_with_seed_increment_returns_new_instance_with_shared_dataset(dataset: Dataset) -> None:
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=2, seed=42)
    new_few_shot_generator = few_shot_generator.with_seed_increment(1)

    assert new_few_shot_generator is not few_shot_generator
    assert isinstance(new_few_shot_generator, RandomFewShotGenerator)
    assert new_few_shot_generator.dataset is dataset


@pytest.mark.parametrize(
    "dataset",
    [
        DummyMultipleChoiceDataset(),
        DummyGenerationDataset(),
        DummyChatDataset(),
    ],
)
def test_with_seed_increment_is_reproducible_from_seed_alone(dataset: Dataset) -> None:
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=2, seed=42)
    new_few_shot_generator = few_shot_generator.with_seed_increment(1)

    directly_constructed = RandomFewShotGenerator(dataset=dataset, num_shots=2, seed=43)

    assert new_few_shot_generator() == directly_constructed()


@pytest.mark.parametrize(
    "dataset",
    [
        DummyMultipleChoiceDataset(),
        DummyGenerationDataset(),
        DummyChatDataset(),
    ],
)
def test_with_seed_increment_is_independent_of_prior_sampling(dataset: Dataset) -> None:
    # Regression test for the bug where a shared, stateful generator's output
    # for a given repeat depended on how many times it had already been sampled
    # (i.e., execution order), rather than being determined by the seed alone.
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=2, seed=42)
    # Consume some samples from the original generator before deriving a new one.
    for _ in range(5):
        few_shot_generator()
    new_few_shot_generator = few_shot_generator.with_seed_increment(1)

    directly_constructed = RandomFewShotGenerator(dataset=dataset, num_shots=2, seed=43)

    assert new_few_shot_generator() == directly_constructed()
