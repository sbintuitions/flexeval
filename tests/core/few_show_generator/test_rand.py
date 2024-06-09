import pytest

from flexeval.core.few_shot_generator.rand import Dataset, RandomFewShotGenerator
from tests.dummy_modules.generation_dataset import DummyGenerationDataset
from tests.dummy_modules.multiple_choice_dataset import DummyMultipleChoiceDataset


@pytest.mark.parametrize(
    ("num_shots", "dataset"),
    [
        (1, DummyGenerationDataset()),
        (2, DummyMultipleChoiceDataset()),
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
    ],
)
def test_if_results_are_random(dataset: Dataset) -> None:
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=2)
    sampled_instances_1 = few_shot_generator()
    sampled_instances_2 = few_shot_generator()
    assert sampled_instances_1 != sampled_instances_2


def test_if_few_show_sampler_avoids_leak() -> None:
    dataset = DummyGenerationDataset()
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
