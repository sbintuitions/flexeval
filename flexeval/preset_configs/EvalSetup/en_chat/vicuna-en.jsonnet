/*
Vicuna Benchmark for large language models.

References:

* [Data Source](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
*/
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'ChatbotBench',
      init_args: {
        path_or_name: 'vicuna-en',
        ref_path_or_name: 'vicuna-en-ref-gpt4',
      },
    },
    metrics: [
      { class_path: 'OutputLengthStats' },
    ],
    gen_kwargs: { max_new_tokens: 1024 },
    batch_size: 4,
  },
}
