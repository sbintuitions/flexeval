{
  class_path: 'HuggingFaceLM',
  init_args: {
    model: 'google/gemma-2-2b-jpn-it',
    default_gen_kwargs: { max_new_tokens: 2048, repetition_penalty: 1.05, temperature: 0.7, top_p: 0.9, disable_compile: true },
  },
}
