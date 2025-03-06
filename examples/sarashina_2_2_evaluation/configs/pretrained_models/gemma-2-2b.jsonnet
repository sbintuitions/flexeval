{
  class_path: 'HuggingFaceLM',
  init_args: {
    model: 'google/gemma-2-2b',
    default_gen_kwargs: { temperature: 0.0 },
    add_special_tokens: true,
  },
}
