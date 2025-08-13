#!/bin/bash
export VLLM_USE_V1=0
export PRESET_CONFIG_DIR=examples/format_following/configs/EvalSetup

eval_setups=(
    ifeval
    ifeval_for_system_prompt
    m_ifeval
    m_ifeval_for_system_prompt
)

lm_config=examples/format_following/configs/ProprietaryLanguageModel/gpt-4.1-2025-04-14.jsonnet
for eval_setup in "${eval_setups[@]}"; do
    lm_name=$(basename $lm_config .jsonnet)
    save_dir=results/$lm_name/$eval_setup

    flexeval_lm --language_model $lm_config \
        --eval_setup $eval_setup \
        --save_dir results/$lm_name/$eval_setup \
        --language_model.default_gen_kwargs.seed 1 \
        --eval_setup.batch_size 10000
done

lm_config=examples/format_following/configs/ProprietaryLanguageModel/gemini-2.5-pro.jsonnet
for eval_setup in "${eval_setups[@]}"; do
    lm_name=$(basename $lm_config .jsonnet)
    save_dir=results/$lm_name/$eval_setup
    pip install --upgrade litellm==v1.67.2

    flexeval_lm --language_model $lm_config \
        --eval_setup $eval_setup \
        --save_dir results/$lm_name/$eval_setup \
        --language_model.default_gen_kwargs.seed 1 \
        --eval_setup.batch_size 4
done
