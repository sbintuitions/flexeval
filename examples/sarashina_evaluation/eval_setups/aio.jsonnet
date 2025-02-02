{
  class_path: 'Generation',
  init_args: {
    eval_dataset: {
      class_path: 'HFGenerationDataset',
      init_args: {
        path: 'sbintuitions/aio-extended-answers',
        reference_list_template: '{{ answers }}',
        split: 'validation',
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: '{{ question }}答えは「',
      },
    },
    metrics: [
      {
        class_path: 'CharF1',
        init_args: {
          lm_output_processor: { class_path: 'AIONormalizer' },
          reference_processor: { class_path: 'AIONormalizer' },
        },
      },
      {
        class_path: 'ExactMatch',
        init_args: {
          lm_output_processor: { class_path: 'AIONormalizer' },
          reference_processor: { class_path: 'AIONormalizer' },
        },
      },
    ],
    gen_kwargs: { max_new_tokens: 64, stop_sequences: ['」'], do_sample: false },
    batch_size: 1,
  },
}
