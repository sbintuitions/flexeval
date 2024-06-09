{
  class_path: 'Generation',
  init_args: {
    eval_dataset: { class_path: 'tests.dummy_modules.DummyGenerationDataset' },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {{ text }}
        |||,
      },
    },
    metrics: [
      { class_path: 'CharF1' },
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 4 },
    batch_size: 1,
  },
}
