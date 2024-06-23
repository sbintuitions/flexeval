# FlexEval

![logo](docs/assets/logo.png)

**Flexible evaluation tool for language models. Easy to extend, highly customizable!**

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/sbintuitions/flexeval/blob/main/README_ja.md">日本語</a> |
    </p>
</h4>

With FlexEval, you can evaluate language models with:

* Zero/few-shot in-context learning tasks
* Open-ended text-generation benchmarks such as [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) with automatic evaluation using GPT-4
* Log-probability-based multiple-choice tasks 
* Computing perplexity of text data

For more use cases, see the [documentation](https://sbintuitions.github.io/flexeval/).


## Key Features

* **Flexibility**: `flexeval` is flexible in terms of the evaluation setup and the language model to be evaluated.
* **Modularity**: The core components of `flexeval` are easily extensible and replaceable.
* **Clarity**: The results of evaluation are clear and all the details are saved.
* **Reproducibility**: `flexeval` should be reproducible, with the ability to save and load configurations and results.

## Installation

```bash
pip install flexeval
```

## Quick Start

The following minimal example evaluates the hugging face model `sbintuitions/tiny-lm` with the `commonsense_qa` task.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model "sbintuitions/tiny-lm" \
  --eval_setup "commonsense_qa" \
  --save_dir "results/commonsense_qa"
```
(The model used in the example is solely for debugging purposes and does not perform well. Try switching to your favorite model!)

The results saved in `--saved_dir` contain:

* `config.json`: The configuration of the evaluation, which can be used to replicate the evaluation.
* `metrics.json`: The evaluation metrics.
* `outputs.jsonl`: The outputs of the language model that comes with instance-level metrics.

You can flexibly customize the evaluation by specifying command-line arguments or configuration files.
Besides the [Transformers](https://github.com/huggingface/transformers) model, you can also evaluate models via [OpenAI ChatGPT](https://openai.com/index/openai-api/) and [vLLM](https://github.com/vllm-project/vllm), and other models can be readily added!

## Next Steps
* Run `flexeval_presets` to check the list of off-the-shelf presets in addition to `commonsense_qa`. You can find the details in the [Preset Configs](https://sbintuitions.github.io/flexeval/preset_configs/) section.
* See [Getting Started](https://sbintuitions.github.io/flexeval/getting_started/) to check the tutorial examples for other kinds of tasks.
* See the [Configuration Guide](https://sbintuitions.github.io/flexeval/configuration_guide/) to set up your evaluation.
