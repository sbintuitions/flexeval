{
  class_path: 'VLLM',
  init_args: {
    model: 'meta-llama/Llama-3.2-3B',
    default_gen_kwargs: { temperature: 0.0 },
    add_special_tokens: true,
  },
}
