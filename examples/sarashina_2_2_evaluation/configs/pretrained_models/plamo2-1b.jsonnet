{
  class_path: 'HuggingFaceLM',  // VLLM cannot be used with custom models
  init_args: {
    model: 'pfnet/plamo-2-1b',
    model_kwargs: { trust_remote_code: true },
    tokenizer_kwargs: { trust_remote_code: true },
    default_gen_kwargs: { do_sample: false },
    add_special_tokens: true,
  },
}
