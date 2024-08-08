/*
Hellaswag is a dataset for physically situated commonsense reasoning.
The dataset is constructed through adversarial filtering to make it challenging for models to perform well.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/rowan/hellaswag)
* [HellaSwag: Can a Machine Really Finish Your Sentence?](https://aclanthology.org/P19-1472/)
*/

local dataset_base_args = {
  path: 'Rowan/hellaswag',
  choices_templates: ['{{ endings[0] }}', '{{ endings[1] }}', '{{ endings[2] }}', '{{ endings[3] }}'],
  answer_index_template: '{{ label }}',
  whitespace_before_choices: true,
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
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {% for item in few_shot_data %}
          {{ item.ctx }} {{ item.choices[item.answer_index] }}
          {% endfor %}
        ||| + '{{ ctx }}',
      },
    },
  },
}
