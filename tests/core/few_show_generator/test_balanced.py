import pytest

from flexeval.core.few_shot_generator.balanced import BalancedFewShotGenerator
from tests.dummy_modules.generation_dataset import DummyGenerationDataset


@pytest.mark.parametrize(
    "num_shots",
    [1, 2],
)
def test_balanced_few_shot_generator(num_shots: int) -> None:
    dataset = DummyGenerationDataset()
    few_shot_sampler = BalancedFewShotGenerator(
        dataset=dataset,
        num_shots=num_shots,
    )

    sampled_instances = few_shot_sampler()
    assert len(sampled_instances) == num_shots


@pytest.mark.parametrize("seed", range(10))
def test_if_instances_are_always_balanced(seed: int) -> None:
    dataset = DummyGenerationDataset()
    few_shot_sampler = BalancedFewShotGenerator(
        dataset=dataset,
        num_shots=2,
        seed=seed,
    )
    sampled_instances = few_shot_sampler()

    # `DummyGenerationDataset` contains two kinds of labels
    # check if both always appear in the sampled instances
    label_set = {instance.references[0] for instance in sampled_instances}
    assert len(label_set) == 2


@pytest.mark.parametrize("seed", [42, 777])
def test_if_random_seed_is_consistent(seed: int) -> None:
    dataset = DummyGenerationDataset()
    sampled_instances_1 = BalancedFewShotGenerator(dataset=dataset, num_shots=2, seed=seed)()
    sampled_instances_2 = BalancedFewShotGenerator(dataset=dataset, num_shots=2, seed=seed)()
    assert sampled_instances_1 == sampled_instances_2


def test_if_results_are_random() -> None:
    dataset = DummyGenerationDataset()
    few_shot_generator = BalancedFewShotGenerator(dataset=dataset, num_shots=2)
    sampled_instances_1 = few_shot_generator()
    sampled_instances_2 = few_shot_generator()
    assert sampled_instances_1 != sampled_instances_2


def test_if_few_show_sampler_avoids_leak() -> None:
    dataset = DummyGenerationDataset()
    eval_inputs = dataset[0].inputs

    # First check if the eval_instance may be in the sampled instances with `num_trials_to_avoid_leak=0`
    few_shot_generator = BalancedFewShotGenerator(dataset=dataset, num_shots=2, num_trials_to_avoid_leak=0)
    leak_found = False
    for _ in range(100):
        sampled_instances = few_shot_generator(eval_inputs=eval_inputs)
        if any(eval_inputs == instance.inputs for instance in sampled_instances):
            leak_found = True
            break
    assert leak_found

    # Now check if the eval_instance is not in the sampled instances with `num_trials_to_avoid_leak=10`
    few_shot_generator = BalancedFewShotGenerator(dataset=dataset, num_shots=2, num_trials_to_avoid_leak=100)
    leak_found = False
    for _ in range(100):
        sampled_instances = few_shot_generator(eval_inputs=eval_inputs)
        if any(eval_inputs == instance.inputs for instance in sampled_instances):
            leak_found = True
            break
    assert not leak_found

    # Finally, check if the exception is raised when the resampling fails to avoid the leak
    few_shot_generator = BalancedFewShotGenerator(dataset=dataset, num_shots=2, num_trials_to_avoid_leak=1)

    with pytest.raises(ValueError):  # noqa: PT012
        for _ in range(100):
            few_shot_generator(eval_inputs=eval_inputs)


def test_with_seed_increment_returns_new_instance_with_shared_dataset() -> None:
    dataset = DummyGenerationDataset()
    few_shot_generator = BalancedFewShotGenerator(dataset=dataset, num_shots=2, seed=42)
    new_few_shot_generator = few_shot_generator.with_seed_increment(1)

    assert new_few_shot_generator is not few_shot_generator
    assert isinstance(new_few_shot_generator, BalancedFewShotGenerator)
    assert new_few_shot_generator.dataset is dataset


def test_with_seed_increment_is_reproducible_from_seed_alone() -> None:
    dataset = DummyGenerationDataset()
    few_shot_generator = BalancedFewShotGenerator(dataset=dataset, num_shots=2, seed=42)
    new_few_shot_generator = few_shot_generator.with_seed_increment(1)

    directly_constructed = BalancedFewShotGenerator(dataset=dataset, num_shots=2, seed=43)

    assert new_few_shot_generator() == directly_constructed()


def test_with_seed_increment_is_independent_of_prior_sampling() -> None:
    # Regression test for the bug where a shared, stateful generator's output
    # for a given repeat depended on how many times it had already been sampled
    # (i.e., execution order), rather than being determined by the seed alone.
    dataset = DummyGenerationDataset()
    few_shot_generator = BalancedFewShotGenerator(dataset=dataset, num_shots=2, seed=42)
    # Consume some samples from the original generator before deriving a new one.
    for _ in range(5):
        few_shot_generator()
    new_few_shot_generator = few_shot_generator.with_seed_increment(1)

    directly_constructed = BalancedFewShotGenerator(dataset=dataset, num_shots=2, seed=43)

    assert new_few_shot_generator() == directly_constructed()
