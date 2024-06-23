/*
Rakuda benckmark concists of a set of 40 questions in Japanese about Japanese-specific topics designed to evaluate the capabilities of AI Assistants in Japanese.

References:

* [Original Repository](https://github.com/yuzu-ai/japanese-llm-ranking)
* [Hugging Face Dataset](https://huggingface.co/datasets/yuzuai/rakuda-questions)
*/
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'ChatbotBench',
      init_args: {
        path_or_name: 'rakuda-v2-ja',
      },
    },
    metrics: [
      { class_path: 'OutputLengthStats' },
    ],
    gen_kwargs: { max_new_tokens: 1024 },
    batch_size: 4,
  },
}
