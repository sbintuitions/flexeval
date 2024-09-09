/*
This dataset is created as a test set for the WMT20 shared task on news translation.
This is Japanese to English translation.

References:

* [Data Source](https://github.com/mjpost/sacrebleu)
* [2020 Fifth Conference on Machine Translation (WMT20)](https://www.statmt.org/wmt20/)
*/
local dataset = {
  class_path: 'SacreBleuDataset',
  init_args: { name: 'wmt20', langpair: 'ja-en' },
};

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset,
    few_shot_generator: {
      class_path: 'RandomFewShotGenerator',
      init_args: {
        // Use the eval dataset for few-shot data,
        // but `RandomFewShotGenerator` will avoid using the same few-shot isntances as the input.
        dataset: dataset,
        num_shots: 4,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {% for item in few_shot_data %}
          Ja: `{{ item.source }}`
          En: `{{ item.references[0] }}`
          {% endfor %}
          Ja: `{{ source }}`
        ||| + 'En: `',
      },
    },
    metrics: [
      { class_path: 'BLEU', init_args: { tokenize_option: 'intl' } },
    ],
    gen_kwargs: { max_new_tokens: 128, stop_sequences: ['`'] },
    batch_size: 4,
  },
}
