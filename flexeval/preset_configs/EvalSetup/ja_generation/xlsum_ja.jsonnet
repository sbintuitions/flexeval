/*
XLSum is a comprehensive and diverse dataset comprising 1.35 million professionally annotated article-summary pairs from BBC, extracted using a set of carefully designed heuristics.
This is a Japanese subset of the dataset.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/csebuetnlp/xlsum)
* [Original Repository](https://github.com/ids-cv/wrime)
* [XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages](https://aclanthology.org/2021.findings-acl.413)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'csebuetnlp/xlsum',
    subset: 'japanese',
    reference_template: '{{ summary }}',
  },
};

{
  // as we deal with LLMs with short context window, we set max_text_length and max_summary_length
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'validation' } },
    few_shot_generator: {
      class_path: 'BalancedFewShotGenerator',
      init_args: {
        dataset: dataset_base_args { init_args+: { split: 'train' } },
        num_shots: 1,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          文章を１〜３文で要約してください。
          {% for item in few_shot_data %}
          文章: {{ item.text }}
          要約: {{ item.references[0] }}
          {% endfor %}
          文章: {{ text }}
        ||| + '要約:',
      },
    },
    metrics: [
      {
        class_path: 'ROUGE',
        init_args: { tokenizer: { class_path: 'SacreBleuTokenizer', init_args: { name: 'ja-mecab' } } },
      },
    ],
    gen_kwargs: { max_new_tokens: 100, stop_sequences: ['\n'] },
  },
}
