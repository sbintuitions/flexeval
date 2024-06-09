# How to configure your own evaluation

## Overview

`flexeval` allows you to evaluate any language model with any task, any prompt, and any metric via the `flexeval_lm` command.
The CLI command is implemented on [jsonargparse](https://github.com/omni-us/jsonargparse), which allows a flexible configuration either by CLI arguments or by a configuration file.

There are many ways to write configuration files, but for now let's see how to define a config for the argument `--eval_setup`.
You can check the configuration for preset setups by running the following command:

```bash
flexeval_presets commonsense_qa
```

This command will show the configuration for the `commonsense_qa` setup.
The content is written in the [jsonnet](https://jsonnet.org/) format, which is a superset of JSON.

!!! tip
    If you want to convert it to JSON, install `jsonnet` command and run `flexeval_presets commonsense_qa | jsonnet -`.

The skeleton of the configuration is as follows:

```json
{
  "class_path": "Generation",
  "init_args": {
    "eval_dataset": {"class_path": "HfGenerationDataset", "init_args": ...},
    "prompt_template": {"class_path": "Jinja2PromptTemplate", "init_args": ...},
    "gen_kwargs": {"max_new_tokens": 32, "stop_sequences": ["」"]},
    "metrics": [{"class_path": "CharF1"}, {"class_path": "ExactMatch"}],
    "batch_size": 4
  }
}
```

The fields `class_path` and `init_args` directly mirror the initialization of the specified class.

At the top level, `"class_path": "Generation"` specifies what kind of [`EvalSetup`](api_reference/EvalSetup.md) to use.
Currently, there are four types of `EvalSetup`: [`Generation`](api_reference/EvalSetup.md#flexeval.scripts.flexeval_lm.Generation), [`ChatResponse`](api_reference/EvalSetup.md#flexeval.scripts.flexeval_lm.ChatResponse), [`MultipleChoice`](api_reference/EvalSetup.md#flexeval.scripts.flexeval_lm.MultipleChoice), and [`Perplexity`](api_reference/EvalSetup.md#flexeval.scripts.flexeval_lm.Perplexity).

Then, `Generation` is composed of the following components:

- `eval_dataset`: The dataset to evaluate. You can choose from concrete classes inheriting [`GenerationDataset`](api_reference/GenerationDataset.md). Most presets use [`HfGenerationDataset`](api_reference/GenerationDataset.md/#flexeval.core.generation_dataset.hf_dataset.HfGenerationDataset), which load datasets from [Hugging Face Hub](https://huggingface.co/docs/datasets/index).
- `prompt_template`: The template to generate prompts fed to the language model. We have [`Jinja2PromptTemplate`](api_reference/PromptTemplate.md/#flexeval.core.prompt_template.jinja2.Jinja2PromptTemplate), which uses [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) to embed the data from [`GenerationDataset`](api_reference/GenerationDataset.md) into the prompt.
- `gen_kwargs`: The keyword arguments passed to [`LanguageModel.batch_complete_text`](api_reference/LanguageModel.md/#flexeval.core.language_model.base.LanguageModel.batch_complete_text). For example, `max_new_tokens` and `stop_sequences` are used to control the generation process. Acceptable arguments depend on the underlying implementation of the generation function (e.g., `generate()` in `transformers`).
- `metrics`: The metrics to compute. You can choose from concrete classes inheriting [`Metric`](api_reference/Metric.md). These modules take the outputs of the language model, the references, and dataset values, and compute the metrics.

Please refer to the [API reference](api_reference/index.md) for available classes and their arguments.

## Customizing the Configuration

Writing a configuration file from scratch is a bit cumbersome, so we recommend starting from the preset configurations and modifying them as needed.

```bash
flexeval_presets commonsense_qa > my_config.jsonnet
```

Then, pass your config file to `--eval_setup` argument.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model_name "sbintuitions/tiny-lm" \
  --eval_setup "my_config.jsonnet"
```

!!! info

    Under the hood, the preset name like `commonsense_qa` is resolved to the corresponding configuration file under `flexeval/preset_configs` in the library.

## Argument Overrides

[jsonargparse](https://github.com/omni-us/jsonargparse) allows you to flexibly combine configuration files and CLI arguments.
You can override the argument values by specifying them in the CLI.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model_name "sbintuitions/tiny-lm" \
  --eval_setup "commonsense_qa" \
  --eval_setup.batch_size 8
```

The value of `--eval_setup.batch_size` overrides the value defined in the config file of `commonsense_qa`.

## What's Next?

- Proceed to [How to](how_to/index.md) to find examples that suit your needs.
- Look at the [API reference](api_reference/index.md) to see the available classes and their arguments.
