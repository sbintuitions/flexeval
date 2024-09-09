/*
AI王 (AI king) is a Japanese quiz dataset developed for research and competition purposes.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/llm-book/aio)
* [AI王 〜クイズAI日本一決定戦〜](https://sites.google.com/view/project-aio/home)
* [JAQKET: クイズを題材にした日本語 QA データセットの構築](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'llm-book/aio',
    reference_list_template: '{{ answers }}',
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
        num_shots: 0,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {% for item in few_shot_data -%}
          {{ item.question }}答えは「{{ item.references[0] }}」
          {% endfor -%}
        ||| + '{{ question }}答えは「',
      },
    },
    metrics: [
      { class_path: 'CharF1', init_args: { processor: { class_path: 'AIONormalizer' } } },
      { class_path: 'ExactMatch', init_args: { processor: { class_path: 'AIONormalizer' } } },
    ],
    gen_kwargs: { max_new_tokens: 32, stop_sequences: ['」'] },
    batch_size: 4,
  },
}
