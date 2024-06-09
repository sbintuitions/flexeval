# Evaluate with LLM Judges

Evaluation of chat model is difficult since the response is open-ended and manually evaluating the responses is not scalable.
One solution is to use a LLM as an evaluator.

## Single Judge Evaluation

First, we need to generate responses from the chat model.
In this example, we use ChatGPT with the following command:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"

flexeval_lm \
  --language_model OpenAIChatGPT \
  --language_model.model_name "gpt-3.5-turbo" \
  --eval_setup "mt-en" \
  --save_dir "results/mt-en_gpt3.5-turbo"
```

Now you have the model outputs in `results/mt-en-gpt3.5-turbo/outputs.jsonl`.

Let's evaluate the responses with GPT4.
The LLM evaluation is implemented as a `Metric` class and we will use a preset metric named `assistant_eval_gpt4_en_single_turn`.
You can check its configuration with the following command:

```bash
flexeval_presets assistant_eval_gpt4_en_single_turn
```

In this metric, GPT4 is asked to rate the responses with the score from 1 to 10.
The score is extracted as the last digit found in the evaluator's output.

!!! tip
    To take a closer look at the prompt template, combine pipeline with `jsonnet` and `jq`:

    ```bash
    flexeval_presets assistant_eval_gpt4_en_single_turn | jsonnet - | jq -r ".init_args.prompt_template.init_args.template"
    ```

Perform automatic evaluation with GPT4 with the following command:

```bash
flexeval_file \
   --eval_file "results/mt-en-gpt3.5-turbo/outputs.jsonl" \
   --metrics "assistant_eval_gpt4_en_single_turn" \
   --save_dir "results/mt-en_gpt3.5-turbo/eval_by_gpt4"
```

☕️ It may take a while to finish the evaluation...

By hitting `cat results/mt-en-gpt3.5-turbo/eval_by_gpt4/metrics.json`, you can see the evaluation result like `{"llm_score": 7.795}`.
The evaluation for each response is stored in `results/mt-en-gpt3.5-turbo/eval_by_gpt4/outputs.jsonl`.

You can check the output of the evaluator LLM in the `llm_score_output` field.

```bash
head -n 1 results/mt-en-gpt3.5-turbo/eval_by_gpt4/outputs.jsonl | jq -r ".llm_output"
```

!!! info
    `flexeval_file` just runs the same evaluation as `flexeval_lm` but with the given file.
    So, theoretically, you can perform the same evaluation with `flexeval_lm` in one go:

    ```bash
    flexeval_lm \
      --language_model OpenAIChatGPT \
      --language_model.model_name "gpt-3.5-turbo" \
      --eval_setup "mt-en" \
      --metrics+="assistant_eval_gpt4_en_single_turn" \
      --save_dir "results/mt-en_gpt3.5-turbo"
    ```

    Yet, we recommend separate the response generation (`flexeval_lm`) and evaluation (`flexeval_file`) so that you don't lost the response by some errors in the evaluation process.

## Pairwise Judge Evaluation

Sometimes, evaluating chat models individually cannot capture a subtle difference between models.
In such cases, pairwise evaluation is useful.

The overview of process is generating the responses using `flexeval_lm` and evaluating them with `flexeval_pairwise`.

In this example, we will compare the responses from GPT3.5 and GPT-4o.

First, generate the responses with GPT3.5.
You can skip this if you have already generated the responses.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"

flexeval_lm \
  --language_model OpenAIChatGPT \
  --language_model.model_name "gpt-3.5-turbo" \
  --eval_setup "mt-en" \
  --save_dir "results/mt-en_gpt3.5-turbo"
```

Generate the responses with GPT-4o.

```bash
flexeval_lm \
  --language_model OpenAIChatGPT \
  --language_model.model_name "gpt-4o" \
  --eval_setup "mt-en" \
  --save_dir "results/mt-en_gpt-4o"
```

Now, compare the responses with GPT-4.

```bash
flexeval_pairwise \
  --lm_output_paths.gpt_3_5 "results/mt-en_gpt3.5-turbo/outputs.jsonl"  \
  --lm_output_paths.gpt_4o "results/mt-en_gpt-4o/outputs.jsonl"  \
  --judge "assistant_judge_gpt4_en_single_turn" \
  --save_dir "results/mt-en_gpt3.5_vs_gpt4o"
```

☕️ It may take a while to finish the evaluation...

You can see the result in `results/mt-en_gpt3.5_vs_gpt4o/scores.json`.

```json
{
    "win_rate": {
        "gpt_4o": 85.3125,
        "gpt_3_5": 14.6875
    },
    "bradley_terry": {
        "gpt_4o": 1152.8129577165919,
        "gpt_3_5": 847.1870422834081
    }
}
```

The `win_rate` shows the percentage of wins of each model.
The `bradley_terry` shows [the Bradley-Terry score](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) of each model.

## What's Next?

* To define your own LLM evaluator, see [Configuration Guide](../configuration_guide.md).
