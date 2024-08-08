/*
Synthetic question answering dataset with reasoning questions.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/muennighoff/babi)
* [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](https://arxiv.org/abs/1502.05698)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'Muennighoff/babi',
    reference_template: '{{ answer }}',
  },
};

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'validation' } },
    few_shot_generator: {
      class_path: 'RandomFewShotGenerator',
      init_args: {
        dataset: dataset_base_args { init_args+: { split: 'train' } },
        num_shots: 3,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {% for item in few_shot_data %}
          Passage: {{ item.passage | trim }}
          Question: {{ item.question }}
          Answer: "{{ item.references[0] }}"
          {% endfor %}
          Passage: {{ passage | trim }}
          Question: {{ question }}
        ||| + 'Answer: "',
      },
    },
    metrics: [
      { class_path: 'CharF1' },
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 32, stop_sequences: ['"'] },
  },
}
