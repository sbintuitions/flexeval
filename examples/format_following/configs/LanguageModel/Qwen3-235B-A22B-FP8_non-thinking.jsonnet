{
  class_path: 'VLLM',
  init_args: {
    model: 'Qwen/Qwen3-235B-A22B-FP8',
    model_kwargs: {
      max_model_len: 32768,
      gpu_memory_utilization: 0.90,
      max_num_seqs: 128,  # OOM occurs w/ 256
      enable_expert_parallel: true,
      tensor_parallel_size: 8,
    },
    // The temperature setting follows best practices for non-thinking mode.
    // https://qwenlm.github.io/blog/qwen3/
    default_gen_kwargs: { max_new_tokens: 2048, repetition_penalty: 1.05, temperature: 0.7, top_p: 0.8, top_k: 20},
    chat_template_kwargs: { enable_thinking: false },
  },
}
