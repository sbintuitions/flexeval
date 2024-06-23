/*
Multi-Turn Benchmark for large language models.

References:

* [Data Source](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
* [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
*/
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'ChatbotBench',
      init_args: {
        path_or_name: 'mt-en',
        ref_path_or_name: 'mt-en-ref-gpt4',
      },
    },
    metrics: [
      { class_path: 'OutputLengthStats' },
    ],
    gen_kwargs: { max_new_tokens: 1024 },
    batch_size: 4,
  },
}
