{
  class_path: 'HuggingFaceLM',
  init_args: {
    model: 'google/gemma-3-27b-it',
    default_gen_kwargs: {
      max_new_tokens: 2048,
      repetition_penalty: 1.05,
      temperature: 0.7,
      top_p: 0.9,
      do_sample: true,
      stop_sequences: ["<end_of_turn>"],
      disable_compile: true,
    },
    model_limit_tokens: 16384,
    model_kwargs: {
      attn_implementation: "eager",
    },
  },
}
