/*
OpenBookQA contains questions that require multi-step reasoning, use of additional common and commonsense knowledge, and rich text comprehension.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/allenai/openbookqa)
* [Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://aclanthology.org/D18-1260/)
*/
local dataset_base_args = {
  path: 'allenai/openbookqa',
  subset: 'main',
  choices_templates: ['{{ choices.text[0] }}', '{{ choices.text[1] }}', '{{ choices.text[2] }}', '{{ choices.text[3] }}'],
  answer_index_template: '{% if answerKey == "A" %}0{% elif answerKey == "B" %}1{% elif answerKey == "C" %}2{% elif answerKey == "D" %}3{% endif %}',
  whitespace_before_choices: true,
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
          Question: {{ item.question_stem }}
          Answer:{{ item.choices[item.answer_index] }}
          {% endfor %}
          Question: {{ question_stem }}
        ||| + 'Answer:',
      },
    },
  },
}
