/*
TriviaqQA is a reading comprehension dataset containing over 650K question-answer-evidence triples.
TriviaqQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/trivia_qa)
* [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://aclanthology.org/P17-1147/)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'trivia_qa',
    subset: 'rc.nocontext',
    reference_list_template: '{{ answer.aliases }}',
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
          {% for item in few_shot_data %}
          Question: {{ item.question }}
          Answer: "{{ item.references[0] }}"
          {% endfor %}
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
