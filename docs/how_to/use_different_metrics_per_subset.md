# Use Different Metrics for Different Subsets of a Dataset

Sometimes a single dataset mixes several kinds of instances, and one metric does not fit all of them.
For example, you may want to use `ExactMatch` for the QA questions and an LLM judge for the open-ended ones, or report the metrics per language.

[`QueryRoutedMetric`](../api_reference/Metric.md) does this.
It selects the instances of each subset with a query on `extra_info` and applies the metrics to those instances only.

## Setup

The query is evaluated on the `extra_info` of each instance, so the dataset must provide the key to split on.

All the fields of the original data are available in `extra_info` by default.
For chat datasets, you can also add derived keys with `extra_info_templates`, which maps a key to a Jinja2 template.

For example, consider the following `toydata.jsonl`:

```json
{"prompt": "What is the capital of France?", "answer": "Paris", "task": "qa"}
{"prompt": "Write a short poem about spring.", "answer": "", "task": "open"}
```

```jsonnet
// toytask.jsonnet
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'JsonlChatDataset',
      init_args: {
        path: 'toydata.jsonl',
        input_template: '{{ prompt }}',
        reference_template: '{{ answer }}',
      },
    },
    metrics: [
      {
        class_path: 'QueryRoutedMetric',
        init_args: {
          routes: [
            {
              name: 'qa',
              where: { task: 'qa' },
              metrics: [{ class_path: 'ExactMatch' }],
            },
            {
              name: 'open',
              where: { task: 'open' },
              metrics: [{ class_path: 'OutputLengthStats' }],
            },
          ],
        },
      },
    ],
  },
}
```

Each entry of `routes` is a `MetricRoute` with the following fields:

- `name`: the name of the subset. It must be unique and is used as the prefix of the result keys.
- `where`: the query to select the instances.
- `metrics`: the metrics applied to the selected instances.

## Check the results

The keys in `metrics.json` and `outputs.jsonl` are prefixed with `{name}/`.

```json
{
  "qa/exact_match": 1.0,
  "open/avg_output_length": 28.0
}
```

If the keys are emitted by multiple metrics in a route and collide, flexeval raises an error.
Distinguish them with arguments like `metric_key`, which many metrics provide.

## Write the query

`where` is a dictionary on `extra_info`, and all of its entries must hold (AND).
A key can refer to a nested value with dots (e.g. `"meta.lang"`).

| Value | Meaning | Example |
|---|---|---|
| scalar | equality | `{ task: 'qa' }` |
| list | membership | `{ task: ['qa', 'mt'] }` |
| dict of operators | `eq`, `ne`, `in`, `not_in`, `regex`, `exists` | `{ title: { regex: '^A', ne: 'AB' } }` |

Omit `where` (or set it to `null`) to apply the metrics to all the instances.
This is handy to combine subset-specific metrics with a metric for the whole dataset.

An instance can match multiple routes or none.

## Report scores per group

To apply the same metrics to the whole dataset but also see the scores for each value of a key, use `group_by`.

```jsonnet
{
  name: 'all',
  metrics: [{ class_path: 'ExactMatch' }],
  group_by: 'lang',
}
```

In addition to `all/exact_match`, the mean of the instance-level scores per group is reported as `all/{group_value}/exact_match`, e.g. `all/en/exact_match`.
`group_by` is ignored for metrics that do not provide instance-level scores.

## Handle unexpected instances

- `on_unmatched`: what to do with instances that match no route. `"skip"` (default) ignores them and `"error"` raises an error.
- `on_missing_key`: what to do when a key used in a query does not exist in `extra_info`. `"skip"` (default) treats the instance as not matching and `"error"` raises an error.

Set them to `"error"` if you want to detect typos in the keys or the dataset.
