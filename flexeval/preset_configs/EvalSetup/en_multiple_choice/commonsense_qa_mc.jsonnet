/*
CommonsenseQA is a multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers.
This is a setup for multiple choice where the model chooses the correct answer based on the log-probabilities of the choices.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/tau/commonsense_qa)
* [CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge](https://aclanthology.org/N19-1421/)
*/
local dataset_base_args = {
  path: 'tau/commonsense_qa',
  choices_templates: ['{{ choices.text[0] }}', '{{ choices.text[1] }}', '{{ choices.text[2] }}', '{{ choices.text[3] }}', '{{ choices.text[4] }}'],
  answer_index_template: '{% if answerKey == "A" %}0{% elif answerKey == "B" %}1{% elif answerKey == "C" %}2{% elif answerKey == "D" %}3{% elif answerKey == "E" %}4{% endif %}',
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
          Question: {{ item.question }}
          Answer:{{ item.choices[item.answer_index] }}
          {% endfor %}
          Question: {{ question }}
        ||| + 'Answer:',
      },
    },
  },
}
