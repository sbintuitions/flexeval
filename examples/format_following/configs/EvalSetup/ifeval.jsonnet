{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'flexeval.JsonlChatDataset',
      init_args: {
        path: 'examples/format_following/data/ifeval.jsonl',
        input_template: '{{ prompt }}',
      },
    },
    metrics: [
      { class_path: 'OutputLengthStats' },
      { class_path: 'examples.format_following.src.metric.instruction_following_eval.FormatFollowingMetric' },
    ],
    gen_kwargs: { max_new_tokens: 4096 },
    batch_size: 4,
  },
}
