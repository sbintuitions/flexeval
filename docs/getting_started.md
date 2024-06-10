# Getting Started

Most evaluations can be done with the `flexeval_lm` command.
With `--eval_setup` option, you can specify the task to evaluate.

## Generation Tasks

The following minimal example evaluates the hugging face model `sbintuitions/tiny-lm` with the `commonsense_qa` task.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model_name "sbintuitions/tiny-lm" \
  --eval_setup "commonsense_qa" \
  --save_dir "results/commonsense_qa"
```

The results are saved in the directory specified by `--save_dir`.

You can find three files: `config.json`, `metrics.json` and `outputs.jsonl`.

### `config.json`

The `config.json` file contains the configuration of the evaluation, as well as metadata useful for replicating the evaluation.

```json
{
    "eval_setup": {
        "class_path": "flexeval.scripts.flexeval_lm.Generation",
        "init_args": {
          "eval_dataset": ...,
          "prompt_template": ...,
          "gen_kwargs": ...,
          "metrics": ...,
          "batch_size": ...,
        },
    },
    "language_model": {
      "class_path": "flexeval.core.language_model.HuggingFaceLM",
      "init_args": {
        "model_name": "sbintuitions/tiny-lm",
        ...
      }
    },
    "save_dir": "results/commonsense_qa",
    "metadata": ...
}
```

!!! tip
    You can replicate the evaluation by specifying the saved config in `flexeval_lm`:

    ```bash
    flexeval_lm --config "results/commonsense_qa/config.json" --save_dir "results/commonsense_qa_replicated"
    ```

### `metrics.json`

The `metrics.json` file contains the evaluation metrics.

```json
{
    "exact_match": 0.004914004914004914,
}
```

### `outputs.jsonl`

The `outputs.jsonl` file contains the outputs of the language model with the following fields:

- `lm_prompt`: The prompt used to generate the output.
- `lm_output`: The output generated by the language model.
- `task_inputs`: The inputs of the task.
- `references`: The references of the task.
- instance-level metrics (e.g., `exact_match`): The metrics computed for each instance.

## Multiple Choice Tasks

Some tasks are implemented as multiple choice tasks.
The following example evaluates the model with the `commonsense_qa_mc` setup, which solves CommonsenseQA by choosing the answer with the highest probability.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model_name "sbintuitions/tiny-lm" \
  --eval_setup "commonsense_qa_mc" \
  --save_dir "results/commonsense_qa_mc"
```

The results are basically the same as the generation tasks, but the `outputs.jsonl` file has a different format:

- `prefix`: The prefix text before the choices.
- `choices`: The choices of the task.
- `answer_index`: The index of the correct choice.
- `log_probs`: The log probabilities of each choice computed by the language model.
- `prediction`: The index of the choice with the highest probability.
- `byte_norm_log_probs`: The byte-normalized log probabilities of each choice.
- `byte_norm_prediction`: The index of the choice with the highest byte-normalized probability.

Whether to use `log_probs` or `byte_norm_log_probs` depends on the task, so both are provided.

## Chat Models

The examples so far are intended to evaluate pretrained language models in zero/few-shot settings.
Evaluating chat models may require a different setup.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"

flexeval_lm \
  --language_model OpenAIChatGPT \
  --language_model.model_name "gpt-3.5-turbo" \
  --eval_setup "mt-en" \
  --save_dir "results/mt-en/gpt-3.5-turbo"
```

!!! note
    You can also specify `HuggingFaceLM` for `--language_model` but the model should have a proper [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating).

`outputs.jsonl` contains the following fields:

- `lm_output`: The response generated by the language model.
- `task_inputs`: The inputs of the task.
  - `messages`: The chat history except for the last turn.
- `references`: The references of the task, if any.
- instance-level metrics (e.g., `output_length`): The metrics computed for each instance.

Usually, the model outputs are evaluated by human evaluation or another LLM.
The preset config only defines simple metrics such as length statistics.

To run automatic evaluation with LLMs, you can use `outputs.jsonl` from the previous command and run the following command:

```bash
flexeval_file \
  --eval_file "results/mt-en/gpt-3.5-turbo/outputs.jsonl" \
  --metrics "assistant_eval_gpt4_en_single_turn" \
  --save_dir "results/mt-en/gpt-3.5-turbo/eval_by_gpt4"
```

In the results, you can see the evaluation result like `{"llm_score": 7.795}`.
You can also check the entire output of the judge LLM including the rationale of the evaluation in `llm_score_output` in `outputs.jsonl`.

For further details and pairwise evaluation, see [Evaluate with LLM Judges](./how_to/eval_with_llm_judges.md).

## Perplexity

You can also compute perplexity of text with the following command:

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model_name "sbintuitions/tiny-lm" \
  --eval_setup "tiny_shakespeare" \
  --save_dir "results/tiny_shakespeare"
```

When evaluating perplexity, there is no `outputs.jsonl` file.
The `metrics.json` file contains the perplexity values normalized by the number of tokens.

```json
{
    "perplexity_per_byte": 9.080868808532346,
    "perplexity_per_character": 9.080868808532346
}
```

!!! tip

    You can get `perplexity_per_token` by specifying the `--tokenizer` option. 
    By default, the command only computes tokenizer-agnostic metrics.

## What's Next?

- Run `flexeval_presets` to check the list of off-the-shelf presets. You can find the details in the [Preset Configs](./preset_configs/index.md) section.
- `flexeval` allows you to evaluate any language model with any task, any prompt, and any metric.
To understand how to configure the evaluation, proceed to [Configuration Guide](./configuration_guide.md).