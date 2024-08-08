/*
JCommonsenseQA is a Japanese version of CommonsenseQA, which is a multiple-choice question answering dataset that requires commonsense reasoning ability.
The dataset is built using crowdsourcing with seeds extracted from the knowledge base ConceptNet.
This is a setup for multiple choice where the model chooses the correct answer based on the log-probabilities of the choices.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/llm-book/JGLUE)
* [Original Repository](https://github.com/yahoojapan/JGLUE)
* [JGLUE: Japanese General Language Understanding Evaluation](https://aclanthology.org/2022.lrec-1.317)
* [JGLUE: 日本語言語理解ベンチマーク](https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf)
*/
local dataset_base_args = {
  path: 'llm-book/JGLUE',
  subset: 'JCommonsenseQA',
  choices_templates: ['{{ choice0 }}', '{{ choice1 }}', '{{ choice2 }}', '{{ choice3 }}', '{{ choice4 }}'],
  answer_index_template: '{{ label }}',
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
        num_shots: 0,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {% for item in few_shot_data %}
          問題：{{ item.question }}
          回答：「{{ item.choices[item.answer_index] }}」
          {% endfor %}
          問題：{{question}}
        ||| + '回答：「',
      },
    },
  },
}
