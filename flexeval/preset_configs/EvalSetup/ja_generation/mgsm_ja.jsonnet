/*
Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems.
This is a Japanese subset of the benchmark.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/juletxara/mgsm)
* [Language Models are Multilingual Chain-of-Thought Reasoners](https://arxiv.org/abs/2210.03057)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'juletxara/mgsm',
    subset: 'ja',
    reference_template: '{{ answer_number }}',
  },
};

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'test' } },
    few_shot_generator: {
      class_path: 'RandomFewShotGenerator',
      init_args: {
        dataset: dataset_base_args { init_args+: { split: 'train' } },
        num_shots: 4,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {% for item in few_shot_data %}
          {{ item.question }}
          {{ item.answer }}
          {% endfor %}
          問題: {{ question }}
        ||| + 'ステップごとの答え:',
      },
    },
    metrics: [
      { class_path: 'ExactMatch', init_args: { processor: { class_path: 'RegexExtractor', init_args: { pattern: '-?[0-9.,]+' } } } },
    ],
    gen_kwargs: { max_new_tokens: 256, stop_sequences: ['問題:'] },
  },
}
