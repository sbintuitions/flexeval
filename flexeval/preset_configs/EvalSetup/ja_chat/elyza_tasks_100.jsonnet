/*
A dataset for evaluating instruction-tuned models developed by ELYZA Inc.

References:

* [Hugging Face Dataset](https://huggingface.co/elyza/ELYZA-tasks-100)
* [公式ブログ](https://note.com/elyza/n/na405acaca130)
*/
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'HFChatDataset',
      init_args: {
        path: 'elyza/ELYZA-tasks-100',
        split: 'test',
        input_template: '{{ input }}',
        reference_template: '{{ output }}',
        extra_info_templates: { eval_aspect: '{{ eval_aspect }}' },
      },
    },
    metrics: [
      { class_path: 'OutputLengthStats' },
    ],
    gen_kwargs: { max_new_tokens: 1024 },
    batch_size: 4,
  },
}
