/*
Mostly Basic Python Problems (MBPP) is a dataset of crowd-sourced programming problems.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/mbpp)
* [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'mbpp',
    subset: 'sanitized',
    reference_list_template: '{{ test_list }}',
  },
};

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'test' } },
    few_shot_generator: {
      class_path: 'RandomFewShotGenerator',
      init_args: {
        dataset: dataset_base_args { init_args+: { split: 'prompt' } },
        num_shots: 3,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {% for item in few_shot_data %}
          ## Question
          {{ item.prompt }}
          ## Test cases
          ```python
          {{ item.test_list | join('\n') }}
          ```
          ## Code
          ```python
          {{ item.code }}
          ```
          {% endfor %}
          ## Question
          {{ prompt }}
          ## Test cases
          ```python
          {{ test_list | join('\n') }}
          ```
          ## Code
          ```python
        |||,
      },
    },
    metrics: [
      { class_path: 'CodeEval' },
    ],
    gen_kwargs: { max_new_tokens: 512, stop_sequences: ['```'] },
    batch_size: 4,
  },
}
