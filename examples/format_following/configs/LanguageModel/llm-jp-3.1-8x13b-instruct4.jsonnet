{
  class_path: 'VLLM',
  init_args: {
    // Please download model and replace tokenizer_config.
    model: '/local/path/to/llm-jp/llm-jp-3.1-8x13b-instruct4',
    default_gen_kwargs: { max_new_tokens: 2048, repetition_penalty: 1.05, temperature: 0.7, top_p: 0.9 },
  },
}
