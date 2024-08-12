/*
JNLI is a Japanese version of the NLI (Natural Language Inference) dataset.
The sentence pairs are extracted from image captions and annotated by crowd workers.

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
    subset: 'JNLI',
    reference_template: "{{ ['\"含意\"', '\"矛盾\"', '\"中立\"'][label] }}",
    dataset_kwargs: { trust_remote_code: true },
  },
};

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'validation' } },
    few_shot_generator: {
      class_path: 'BalancedFewShotGenerator',
      init_args: {
        dataset: dataset_base_args { init_args+: { split: 'train' } },
        num_shots: 3,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          前提と仮説の関係を「中立」、「含意」、「矛盾」の中から回答してください。
          {% for item in few_shot_data %}
          前提：「{{ item.sentence1 }}」
          仮説：「{{ item.sentence2 }}」
          関係：「{{ item.references[0] }}」
          {% endfor %}
          前提：「{{ sentence1 }}」
          仮説：「{{ sentence2 }}」
        ||| + '関係：「',
      },
    },
    metrics: [
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 6, stop_sequences: ['前提', '」'] },
  },
}
