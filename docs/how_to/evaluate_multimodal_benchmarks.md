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
            class_path: 'docvqa.preprocessors.ConvertImageToBase64',
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

The `preprocessors` argument accepts a list of `Preprocessor` instances that sequentially transform each dataset item before prompt generation. In the configuration above, `ConvertImageToBase64` is used to encode image objects into Base64 strings.

For `flexeval` to load this custom preprocessor, create `docvqa/preprocessors.py` and define `ConvertImageToBase64` by extending the base `Preprocessor` class and implementing the `__call__` method:
```python
import base64
from io import BytesIO
from PIL import Image
from flexeval.core.chat_dataset import Preprocessor

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format=image.format or "PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class ConvertImageToBase64(Preprocessor):
    """Convert image to base64 string."""

    key: str

    def __call__(self, data: Data) -> Data:
        image = data[self.key]
        if image is None:
            base64_image = None
        elif isinstance(image, Image.Image):
            base64_image = image_to_base64(image)
        else:
            raise NotImplementedError(f"Unsupported image type: {type(image)}")

        data[f"{self.key}_base64"] = base64_image
        return data
```

## Running the Benchmark
With the benchmark defined and the preprocessor in place, you can now run the evaluation through `flexeval`.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model "sbintuitions/sarashina2.2-vision-3b" \
  --eval_setup docvqa.jsonnet\
  --save_dir results/sarashina/docvqa
```

This command evaluates the `Sarashina2.2-Vision-3B` model on the DocVQA benchmark and saves the results to `results/sarashina/docvqa`.