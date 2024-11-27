/*
The ARC dataset contains 7,787 genuine grade-school level, multiple-choice science questions,
assembled to encourage research in advanced question-answering.
The dataset is partitioned into a Challenge Set and an Easy Set, and this is the Challenge Set.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/allenai/ai2_arc)
* [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457)
*/
local dataset_base_args = {
  path: 'allenai/ai2_arc',
  subset: 'ARC-Challenge',
  choices_templates: ['{{ choices.text[0] }}', '{{ choices.text[1] }}', '{{ choices.text[2] }}', '{{ choices.text[3] }}'],
  # answerKey is one of A, B, C, D, 1, 2, 3, 4
  answer_index_template: '{% if answerKey == "A" %}0{% elif answerKey == "B" %}1{% elif answerKey == "C" %}2{% elif answerKey == "D" %}3{% else %}{{ answerKey | int - 1 }}{% endif %}',
  whitespace_before_choices: true,
  remove_conditions: {
    # Remove questions with 3 or 5 choices because the size of choices_template is fixed to 4.
    '{{ choices.text | length }}': '3',
    '{{ choices.label | length }}': '5',
  },
};

{
  class_path: 'MultipleChoice',
  init_args: {
    eval_dataset: {
      class_path: 'HFMultipleChoiceDataset',
      init_args: dataset_base_args { split: 'test' },
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
          Question: {{ item.question }}
          Answer:{{ item.choices[item.answer_index] }}
          {% endfor %}
          Question: {{ question }}
        ||| + 'Answer:',
      },
    },
  },
}
