/*
 The PIQA dataset introduces the task of physical commonsense reasoning and a corresponding benchmark dataset

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/ybisk/piqa)
* [PIQA: Reasoning about Physical Commonsense in Natural Language](https://ojs.aaai.org/index.php/AAAI/article/view/6239)
*/

local dataset_base_args = {
  path: 'ybisk/piqa',
  choices_templates: ['{{ sol1 }}', '{{ sol2 }}'],
  answer_index_template: '{{ label }}',
  whitespace_before_choices: true,
  dataset_kwargs: { trust_remote_code: true },
};

{
  class_path: 'MultipleChoice',
  init_args: {
    eval_dataset: {
      class_path: 'HFMultipleChoiceDataset',
      init_args: dataset_base_args { split: 'validation' },
    },
    few_shot_generator: {
      class_path: 'RandomFewShotGenerator',
      init_args: {
        dataset: {
          class_path: 'HFMultipleChoiceDataset',
          init_args: dataset_base_args { split: 'train' },
        },
        num_shots: 4,
      },
    },
    prompt_template: |||
      {% for item in few_shot_data %}
      {{ item.goal }}{{ item.choices[item.answer_index] }}
      {% endfor %}
    ||| + '{{ goal }}',
  },
}
