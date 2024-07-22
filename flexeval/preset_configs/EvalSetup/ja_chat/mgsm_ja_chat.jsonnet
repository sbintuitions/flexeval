/*
Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems.
This is a Japanese subset of the benchmark.
This is a evaluation setup for chat LLMs.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/juletxara/mgsm)
* [Language Models are Multilingual Chain-of-Thought Reasoners](https://arxiv.org/abs/2210.03057)
*/
local dataset_base_args = {
  class_path: 'HFChatDataset',
  init_args: {
    path: 'juletxara/mgsm',
    subset: 'ja',
    reference_template: '{{ answer }}',
  },
};

{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'test', input_template: '問題: {{ question }}' } },
    few_shot_generator: {
      class_path: 'RandomFewShotGenerator',
      init_args: {
        dataset: dataset_base_args { init_args+: { split: 'train', input_template: '{{ question }}' } },
        num_shots: 4,
      },
    },
    metrics: [
      { class_path: 'ExactMatch', init_args: { processor: { class_path: 'RegexExtractor', init_args: { pattern: '-?[0-9.,]+' } } } },
    ],
    gen_kwargs: { max_new_tokens: 256 },
  },
}
