/*
Mostly Basic Python Problems (MBPP) is a dataset of crowd-sourced programming problems.
This is a evaluation setup for chat LLMs.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/mbpp)
* [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732)
*/
local dataset_base_args = {
  class_path: 'HFChatDataset',
  init_args: {
    path: 'mbpp',
    subset: 'sanitized',
    input_template: std.stripChars(|||
      Generate a Python function that satisfies the following question and test cases.
      ## Question
      {{ prompt }}
      ## Test cases
      ```python
      {{ test_list | join('\n') }}
      ```
    |||, '\n'),
  },
};

{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'test', reference_list_template: '{{ test_list | join("\n") }}' } },
    few_shot_generator: {
      class_path: 'RandomFewShotGenerator',
      init_args: {
        dataset: dataset_base_args { init_args+: { split: 'prompt', reference_template: '```python\n{{ code }}\n```' } },
        num_shots: 3,
      },
    },
    metrics: [
      { class_path: 'CodeEval', init_args: { processor: { class_path: 'RegexExtractor', init_args: { pattern: '```python\n(.*?)\n```' } } } },
    ],
    gen_kwargs: { max_new_tokens: 512 },
    batch_size: 4,
  },
}
