# Configure Few-Shot Examples

The logic to configure few-shot examples is implemented in the [`FewShotGenerator`](../api_reference/FewShotGenerator.md) classes.

## Change the number of shots

### Overriding the arguments

Most presets use [`RandomFewShotGenerator`](../api_reference/FewShotGenerator.md/#flexeval.core.few_shot_generator.rand.RandomFewShotGenerator) to generate few-shot examples.
From command line, the number of shots can be changed using the `--eval_setup.few_shot_generator.num_shots` argument.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model "sbintuitions/tiny-lm" \
  --eval_setup "commonsense_qa" \
  --eval_setup.few_shot_generator.num_shots 3
```

### Editing the configuration file

First, save the configuration file to a local file.

```bash
flexeval_presets commonsense_qa > commonsense_qa_custom.jsonnet
```

Then, edit the `num_shots` field in the `few_shot_generator` section.

Finally, run the evaluation with the custom configuration file.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model "sbintuitions/tiny-lm" \
  --eval_setup "commonsense_qa_custom.jsonnet" 
```

## Change the sampling method

Sometime, you may want to change the sampling method for few-shot examples.
In that case, you can change the [`FewShotGenerator`](../api_reference/FewShotGenerator.md) class.

For example, see the config file from `flexeval_presets twitter_sentiment`.
It uses [`BalancedFewShotGenerator`](../api_reference/FewShotGenerator.md/#flexeval.core.few_shot_generator.balanced.BalancedFewShotGenerator) to generate few-shot examples.
This classes samples examples so that the number of labels (the first element of the `references` field) is balanced.

See [API Reference](../api_reference/FewShotGenerator.md) for other available classes.
