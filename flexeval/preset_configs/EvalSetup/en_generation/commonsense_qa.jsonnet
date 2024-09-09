/*
CommonsenseQA is a multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers.
This is a setup for generating answers based on the choices provided.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/tau/commonsense_qa)
* [CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge](https://aclanthology.org/N19-1421/)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'tau/commonsense_qa',
    reference_template: '{% set answer_index = choices.label.index(answerKey) %}{{ choices.text[answer_index] }}',
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
        num_shots: 2,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          Choose the correct answer from the choices.
          {% for item in few_shot_data %}
          Choices:
          0. "{{ item.choices.text[0] }}"
          1. "{{ item.choices.text[1] }}"
          2. "{{ item.choices.text[2] }}"
          3. "{{ item.choices.text[3] }}"
          4. "{{ item.choices.text[4] }}"
          Question: {{ item.question }}
          Answer: "{{ item.references[0] }}"
          {% endfor %}
          Choices:
          0. "{{ choices.text[0] }}"
          1. "{{ choices.text[1] }}"
          2. "{{ choices.text[2] }}"
          3. "{{ choices.text[3] }}"
          4. "{{ choices.text[4] }}"
          Question: {{question}}
        ||| + 'Answer: "',
      },
    },
    metrics: [
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 40, stop_sequences: ['"'] },
  },
}
