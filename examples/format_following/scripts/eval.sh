#!/bin/bash
export VLLM_USE_V1=0
export PRESET_CONFIG_DIR=examples/format_following/configs/EvalSetup


eval_setups=(
    ifeval
    ifeval_for_system_prompt
    m_ifeval
    m_ifeval_for_system_prompt
)

for lm_config in examples/format_following/configs/LanguageModel/*; do
for eval_setup in "${eval_setups[@]}"; do
    if [[ "$lm_config" == *"Qwen"* || "$lm_config" == *"gemma"* ]]; then
        continue
    fi

    lm_name=$(basename $lm_config .jsonnet)
    save_dir=results/$lm_name/$eval_setup
    flexeval_lm --language_model $lm_config \
        --eval_setup $eval_setup \
        --save_dir results/$lm_name/$eval_setup \
        --language_model.default_gen_kwargs.seed 1 \
        --eval_setup.batch_size 10000
done
done


lm_config=examples/format_following/configs/LanguageModel/Qwen3-235B-A22B-FP8_non-thinking.jsonnet
for eval_setup in "${eval_setups[@]}"; do
    lm_name=$(basename $lm_config .jsonnet)
    save_dir=results/$lm_name/$eval_setup
    # Special cares are needed
    pip install vllm==0.8.5
    (python -c \"import pkg_resources, sys; sys.exit(pkg_resources.get_distribution('flexeval').version < '0.11.2')\" || pip install 'flexeval==0.11.2')
    flexeval_lm --language_model $lm_config \
        --eval_setup $eval_setup \
        --save_dir results/$lm_name/$eval_setup \
        --language_model.default_gen_kwargs.seed 1 \
        --eval_setup.batch_size 10000
done

# We use HuggingfaceLM class for gemma.
lm_config=examples/format_following/configs/LanguageModel/gemma-3-27b-it.jsonnet
for eval_setup in "${eval_setups[@]}"; do
    lm_name=$(basename $lm_config .jsonnet)
    save_dir=results/$lm_name/$eval_setup
    flexeval_lm --language_model $lm_config \
        --eval_setup $eval_setup \
        --save_dir results/$lm_name/$eval_setup \
        --language_model.random_seed 1 \
        --eval_setup.batch_size 16
done
