/*
AI王 (AI king) is a Japanese quiz dataset developed for research and competition purposes.
This is a evaluation setup for chat LLMs.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/llm-book/aio)
* [AI王 〜クイズAI日本一決定戦〜](https://sites.google.com/view/project-aio/home)
* [JAQKET: クイズを題材にした日本語 QA データセットの構築](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf)
*/
local dataset_base_args = {
  class_path: 'HFChatDataset',
  init_args: {
    path: 'llm-book/aio',
    input_template: '{{ question }}',
    reference_list_template: '{{ answers }}',
    dataset_kwargs: { trust_remote_code: true },
  },
};

{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'validation' } },
    few_shot_generator: {
      class_path: 'RandomFewShotGenerator',
      init_args: {
        dataset: dataset_base_args { init_args+: { split: 'train' } },
        num_shots: 4,
      },
    },
    metrics: [
      { class_path: 'CharF1', init_args: { processor: { class_path: 'AIONormalizer' } } },
      { class_path: 'ExactMatch', init_args: { processor: { class_path: 'AIONormalizer' } } },
    ],
    gen_kwargs: { max_new_tokens: 32 },
    batch_size: 4,
  },
}
