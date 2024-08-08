/*
GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems.
The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/gsm8k]
* [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'gsm8k',
    subset: 'main',
    reference_template: '{{ answer | regex_replace("<<.*?>>", "") }}',
  },
};

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'test' } },
    few_shot_generator: {
      class_path: 'RandomFewShotGenerator',
      init_args: {
        dataset: dataset_base_args { init_args+: { split: 'train' } },
        num_shots: 4,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {% for item in few_shot_data %}
          Q: {{ item.question }}
          A: {{ item.references[0] }}
          {% endfor %}
          Q: {{ question }}
        ||| + 'A:',
      },
    },
    metrics: [
      { class_path: 'ExactMatch', init_args: { processor: { class_path: 'RegexExtractor', init_args: { pattern: '-?[0-9.,]+' } } } },
    ],
    gen_kwargs: { max_new_tokens: 256, stop_sequences: ['Q:'] },
  },
}
