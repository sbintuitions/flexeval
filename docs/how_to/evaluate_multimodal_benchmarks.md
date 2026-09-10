# Evaluate Multimodal Benchmarks

In this guide, we will walk you through the process of evaluating a Multimodal Language Model (MLM) on a multimodal benchmark using flexeval.
We will take the benchmark DocVQA as an example and evaluate `Sarashina2.2-Vision-3B` on it.

## Defining the Multimodal Benchmark

Flexeval allows you to define a custom benchmark setup easily by configuring the `HFChatDataset` class. In this example, we showcase how to do this through a jsonnet configuration file (`docvqa.jsonnet`) as follows:

```jsonnet
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'HFChatDataset',
      init_args: {
        path: 'lmms-lab/DocVQA',
        split: 'validation',
        subset: 'DocVQA',
        input_template: '[{ "type": "image_url", "image_url": {"url": "{{ image_base64 }}"}}, { "type": "text", "text": """{{ question }}\nAnswer the question using a single word or phrase."""},]',
        reference_list_template: '{{ answers }}',
        parse_input_utterance: "literal_eval",
        preprocessors: [
          {
            class_path: 'ConvertImageToBase64',
            init_args: {
              key: 'image',
            },
          },
        ],
      },
    },
    metrics: [
      { class_path: 'ExactMatch' },
    ],
  },
}
```

Multimodal Language Models generally require structured input. The `HFChatDataset` templates output raw strings by default, and allows specifying the `parse_input_utterance` argument to convert them into these required structures. Accepted values are `literal_eval` (for `ast.literal_eval`), `json_loads` (for `json.loads`), or `None`.

In the configuration above, we use `literal_eval` because the template outputs a Python literal string rather than strict JSON. This safely evaluates the string directly into the Python list the model expects.

### Preprocessors

The `preprocessors` argument accepts a list of `Preprocessor` instances that sequentially transform each dataset item before prompt generation. In the configuration above, the built-in `ConvertImageToBase64` encodes image objects into Base64 data URLs under `image_base64`.

Other built-in preprocessors: `ConvertImageListToBase64` (list of images → `images_base64`) and `EnsureMinSize` (upscale tiny images in place).

If a benchmark needs a transformation not covered by the built-ins, define a custom preprocessor by extending the base `Preprocessor` class and implementing the `__call__` method, then reference it in the config by its import path (made importable via `PYTHONPATH`).

## Running the Benchmark

With the benchmark defined, you can now run the evaluation through `flexeval`.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model "sbintuitions/sarashina2.2-vision-3b" \
  --eval_setup docvqa.jsonnet\
  --save_dir results/sarashina/docvqa
```

This command evaluates the `Sarashina2.2-Vision-3B` model on the DocVQA benchmark and saves the results to `results/sarashina/docvqa`.
