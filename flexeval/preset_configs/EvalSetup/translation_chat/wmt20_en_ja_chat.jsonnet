/*
This dataset is created as a test set for the WMT20 shared task on news translation.
This is English to Japanese translation.
This is a evaluation setup for chat LLMs.

References:

* [Data Source](https://github.com/mjpost/sacrebleu)
* [2020 Fifth Conference on Machine Translation (WMT20)](https://www.statmt.org/wmt20/)
*/
local dataset = {
  class_path: 'SacreBleuChatDataset',
  init_args: { name: 'wmt20', langpair: 'en-ja' },
};

{
  class_path: 'ChatResponse',
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
    metrics: [
      { class_path: 'BLEU', init_args: { tokenize_option: 'ja-mecab' } },
    ],
    gen_kwargs: { max_new_tokens: 128, stop_sequences: ['`'] },
    batch_size: 4,
  },
}
