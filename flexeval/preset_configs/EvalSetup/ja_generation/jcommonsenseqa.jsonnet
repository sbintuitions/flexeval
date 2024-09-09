/*
JCommonsenseQA is a Japanese version of CommonsenseQA, which is a multiple-choice question answering dataset that requires commonsense reasoning ability.
The dataset is built using crowdsourcing with seeds extracted from the knowledge base ConceptNet.
This is a setup for generating answers based on the choices provided.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/llm-book/JGLUE)
* [Original Repository](https://github.com/yahoojapan/JGLUE)
* [JGLUE: Japanese General Language Understanding Evaluation](https://aclanthology.org/2022.lrec-1.317)
* [JGLUE: 日本語言語理解ベンチマーク](https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'llm-book/JGLUE',
    subset: 'JCommonsenseQA',
    reference_template: '{% set choices = [choice0, choice1, choice2, choice3, choice4] %}{{ choices[label] }}',
    dataset_kwargs: { trust_remote_code: true },
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
          正しい答えは何でしょう？
          {% for item in few_shot_data %}
          0.「{{ item.choice0 }}」
          1.「{{ item.choice1 }}」
          2.「{{ item.choice2 }}」
          3.「{{ item.choice3 }}」
          4.「{{ item.choice4 }}」
          問題：{{ item.question }}
          回答：「{{ item.references[0] }}」
          {% endfor %}
          0.「{{ choice0 }}」
          1.「{{ choice1 }}」
          2.「{{ choice2 }}」
          3.「{{ choice3 }}」
          4.「{{ choice4 }}」
          問題：{{question}}
        ||| + '回答：「',
      },
    },
    metrics: [
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 40, stop_sequences: ['」'] },
  },
}
