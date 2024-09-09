/*
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/rajpurkar/squad)
* [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/)
*/

local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'rajpurkar/squad',
    reference_list_template: '{{ answers.text }}',
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
        num_shots: 2,
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {

        template: |||
          {% for item in few_shot_data %}
          Context: {{ item.context | trim }}
          Question: {{ item.question }}
          Answer: "{{ item.references[0] }}"
          {% endfor %}
          Context: {{ context | trim }}
          Question: {{ question }}
        ||| + 'Answer: "',
      },
    },
    metrics: [
      { class_path: 'CharF1' },
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 32, stop_sequences: ['"'] },
  },
}
