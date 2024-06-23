/*
Multi-Turn Benchmark for large language models in Japanese.

References:

* [Data Source](https://github.com/Stability-AI/FastChat/tree/jp-stable/fastchat/llm_judge)
*/
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'ChatbotBench',
      init_args: {
        path_or_name: 'mt-ja',
        ref_path_or_name: 'mt-ja-ref-gpt4',
      },
    },
    metrics: [
      { class_path: 'OutputLengthStats' },
    ],
    gen_kwargs: { max_new_tokens: 1024 },
    batch_size: 4,
  },
}
