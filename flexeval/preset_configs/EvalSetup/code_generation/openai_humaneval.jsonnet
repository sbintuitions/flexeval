/*
Zero-shot Python code generation task developed by OpenAI.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/openai_humaneval)
* [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
*/
{
  class_path: 'Generation',
  init_args: {
    eval_dataset: {
      class_path: 'HFGenerationDataset',
      init_args: {
        path: 'openai_humaneval',
        split: 'test',
        reference_template: '{{ test }}\n\ncheck({{ entry_point }})\n',
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: '{{ prompt }}',
      },
    },
    metrics: [
      { class_path: 'CodeEval', init_args: { code_template: '{{ prompt }}{{ lm_output }}' } },
    ],
    gen_kwargs: { max_new_tokens: 512, stop_sequences: ['\nclass', '\ndef', '\n#', '\n@', '\nprint', '\nif', '\n```'] },
    batch_size: 4,
  },
}
