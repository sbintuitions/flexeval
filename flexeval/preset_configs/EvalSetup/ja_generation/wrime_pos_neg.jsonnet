/*
WRIME (dataset of Writers’ and Readers’ Intensities of eMotion for their Estimation) is constructed by annotating Internet posts with both the writer’s subjective emotional intensity and the reader’s objective one.
This setup converts the original dataset into binary sentiment classification.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/llm-book/wrime-sentiment)
* [Original Repository](https://github.com/ids-cv/wrime)
* [WRIME: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations](https://aclanthology.org/2021.naacl-main.169)
* [A Japanese Dataset for Subjective and Objective Sentiment Polarity Classification in Micro Blog Domain](https://aclanthology.org/2022.lrec-1.759/)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'llm-book/wrime-sentiment',
    reference_template: "{{ ['\"ポジティブ\"', '\"ネガティブ\"'][label] }}",
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
        num_shots: 4,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          文の極性について「ポジティブ」か「ネガティブ」かで答えてください。
          {% for item in few_shot_data %}
          文：{{ item.sentence }}
          極性：「{{ item.references[0] }}」
          {% endfor %}
          文：{{sentence}}
        ||| + '極性：「',
      },
    },
    metrics: [
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 8, stop_sequences: ['」'] },
  },
}
