# Implement Your Own Module

FlexEval is designed to be highly extensible, allowing you to implement your own modules without changing the core codebase.
Whether you want to add new language models, evaluation setups, prompt templates, or metrics, FlexEval provides a flexible framework for you to do so.

In this guide, we'll walk through the process of implementing your own module in FlexEval.

## Step 1: Identify the Module Type

First, determine which type of module you want to implement.
FlexEval supports the modules listed in the [API Reference](../api_reference/FewShotGenerator.md).

For this guide, we'll focus on implementing a new [`Metric`](../api_reference/Metric.md).

## Step 2: Create Your Module

Create a new Python file and inherit the appropriate base class provided by `flexeval`.

```bash
mkdir custom_modules
touch custom_modules/__init__.py
touch custom_modules/my_custom_metric.py
```

!!! note
    Make sure the module is importable from your program by adding an `__init__.py` file in the directory.

All you need to do is implement the required methods based on the module type.
Let's create a simple custom metric that calculates the length ratio of the generated text to the reference text.

`custom_modules/my_custom_metric.py`:

```python
from flexeval import Metric, MetricResult


class MyCustomMetric(Metric):
    """
    My custom metric implementation.
    This class reports the length ratio of the generated text to the reference text.
    """
    def evaluate(
        self,
        lm_outputs: list[str],
        extra_info_list: list[dict[str, str]],
        references_list: list[list[str]],
    ) -> MetricResult:
        length_ratios = [
            len(lm_output) / len(references[0])  # Assuming a single reference
            for lm_output, references in zip(lm_outputs, references_list)
        ]

        return MetricResult(
            {"length_ratio": sum(length_ratios) / len(length_ratios)},
            instance_details=[{"length_ratio": ratio} for ratio in length_ratios],
        )
```

## Step 3: Specify the Module in the Configuration

Now you can use your custom metric to run evaluations.

It can be specified in the configuration file as follows:

```jsonnet
{
  class_path: 'Generation',
  init_args: {
    metrics: [
      {class_path: "custom_modules.my_custom_metric.MyCustomMetric" }
    ]
  }
}
```

!!! note
    Make sure the `class_path` is a full import path to the module.
    Unlike the core modules, the program cannot locate your custom module without the full path.

Or add it from the command line:

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model "sbintuitions/tiny-lm" \
  --eval_setup "commonsense_qa" \
  --eval_setup.metrics+="custom_modules.my_custom_metric.MyCustomMetric"
```

!!! info
    The argument `--eval_setup.metrics` can take a list of metric classes.
    You can use `+=` to add your custom metric to the existing metrics in the config.

You will see the new metric `length_ratio` in the results.

Now you've successfully implemented your own module 🎉.

If you believe your module would be useful for others, consider contributing it to the official repository.
