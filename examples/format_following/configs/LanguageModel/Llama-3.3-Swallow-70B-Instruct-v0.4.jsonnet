{
  class_path: 'VLLM',
  init_args: {
    model: 'tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4',
    default_gen_kwargs: { max_new_tokens: 2048, repetition_penalty: 1.05, temperature: 0.7, top_p: 0.9 },
    model_kwargs: {
      enable_chunked_prefill: false,
      max_model_len: 8196,
    },
  },
}
