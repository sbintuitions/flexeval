/*
JSQuAD is a Japanese version of SQuAD, one of the datasets of reading comprehension.
The passages are extracted from Japanese Wikipedia, and the questions and answers are created by crowd workers.

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
    subset: 'JSQuAD',
    reference_list_template: '{{ answers.text }}',
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
        num_shots: 3,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          質問に対する回答を文章から一言で抜き出してください。
          {% for item in few_shot_data %}
          文章
          {{ item.title }}
          {{ item.context }}
          質問：「{{ item.question }}」
          回答：「{{ item.references[0] }}」
          {% endfor %}
          文章
          {{ title }}
          {{ context }}
          質問：「{{ question }}」
        ||| + '回答：「',
      },
    },
    metrics: [
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 40, stop_sequences: ['」'] },
  },
}
