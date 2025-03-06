{
  class_path: 'VLLM',
  init_args: {
    model: 'SakanaAI/TinySwallow-1.5B-Instruct',
    default_gen_kwargs: { max_new_tokens: 2048, repetition_penalty: 1.05, temperature: 0.7, top_p: 0.9 },
  },
}
