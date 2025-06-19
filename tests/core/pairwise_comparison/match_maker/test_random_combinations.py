from flexeval.core.pairwise_comparison.match import Match
from flexeval.core.pairwise_comparison.match_maker.random_combinations import RandomCombinations

model_items = {
    "modelA": [
        {
            "lm_output": "A_1",
            "extra_info": {"chat": [{"role": "user", "content": "A_1"}]},
        },
        {
            "lm_output": "A_2",
            "extra_info": {"chat": [{"role": "user", "content": "A_2"}]},
        },
    ],
    "modelB": [
        {
            "lm_output": "B_1",
            "extra_info": {"chat": [{"role": "user", "content": "B_1"}]},
        },
        {
            "lm_output": "B_2",
            "extra_info": {"chat": [{"role": "user", "content": "B_2"}]},
        },
    ],
    "modelC": [
        {
            "lm_output": "C_1",
            "extra_info": {"chat": [{"role": "user", "content": "C_1"}]},
        },
        {
            "lm_output": "C_2",
            "extra_info": {"chat": [{"role": "user", "content": "C_2"}]},
        },
    ],
}


def test_random_combinations() -> None:
    match_maker = RandomCombinations(n=2)
    actual = list(match_maker.generate_matches(model_items, []))
    assert len(actual) == 2
    assert isinstance(actual[0], Match)
    # For each iteration, the model with the fewest matches is assigned to a new match.
    # Thus, if n = 2 for the three models, all models are finally assigned.
    assigned_models = {actual[0].model1, actual[0].model2, actual[1].model1, actual[1].model2}
    assert assigned_models == {"modelA", "modelB", "modelC"}


def test_if_cache_exists() -> None:
    match_maker = RandomCombinations(n=2)
    cached_matches = [
        Match(
            "model1",
            {
                "lm_output": "1_1",
                "extra_info": {"chat": [{"role": "user", "content": "1_1"}]},
            },
            "model2",
            {
                "lm_output": "2_1",
                "extra_info": {"chat": [{"role": "user", "content": "2_1"}]},
            },
        ),
    ]
    actual = list(match_maker.generate_matches(model_items, cached_matches))
    assert len(actual) == 2
    # The modelC doesn't exist in the cache, it must be assigned to a new match.
    assert "modelC" in {actual[1].model1, actual[1].model2}


def test_if_n_is_larger_than_all_combinations() -> None:
    match_maker = RandomCombinations(n=1000)
    actual = list(match_maker.generate_matches(model_items, []))
    assert len(actual) == 12  # 6 model pairs (3P2) * 2 chats
