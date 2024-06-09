{
  class_path: 'MultipleChoice',
  init_args: {
    eval_dataset: { class_path: 'tests.dummy_modules.DummyMultipleChoiceDataset' },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {{ text }}
        |||,
      },
    },
    batch_size: 1,
  },
}
