{
  class_path: 'VLLM',
  init_args: {
    model: 'llm-jp/llm-jp-3-3.7b-instruct3',
    default_gen_kwargs: { max_new_tokens: 2048, repetition_penalty: 1.05, temperature: 0.7, top_p: 0.9 },
  },
}
