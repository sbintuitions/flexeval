/*
TSATC: Twitter Sentiment Analysis Training Corpus.
This dataset is a preprocessed version of the original dataset.
See the hugging face dataset page for more information.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis)
* [Twitter Sentiment Analysis Training Corpus (Dataset)](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'carblacac/twitter-sentiment-analysis',
    reference_template: "{{ ['Positive', 'Negative'][feeling] }}",
  },
};

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'test' } },
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
          Classify the sentiment of the following tweet.
          {% for item in few_shot_data %}
          Tweet: {{ item.text }}
          Sentiment: `{{ item.references[0] }}`
          {% endfor %}
          Tweet: {{ text }}
        ||| + 'Sentiment: `',
      },
    },
    metrics: [
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 8, stop_sequences: ['`'] },
  },
}
