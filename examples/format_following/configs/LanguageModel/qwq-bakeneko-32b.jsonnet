{
  class_path: 'VLLM',
  init_args: {
    model: 'rinna/qwq-bakeneko-32b',
    model_kwargs: {
        max_model_len: 32768,
        gpu_memory_utilization: 0.95,
    },
    // The temperature setting follows https://huggingface.co/rinna/qwq-bakeneko-32b.
    default_gen_kwargs: { max_new_tokens: 32768, temperature: 0.6, top_p: 0.95, top_k: 40 },
    string_processors: [
      { class_path: 'RegexExtractor', init_args: { pattern: '^(?:.*</think>\\s*)?(.*)$' } },
    ]
  },
}
