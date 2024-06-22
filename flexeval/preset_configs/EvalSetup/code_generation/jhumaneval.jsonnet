/*
Zero-shot Python code generation task in Japanese.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/kogi-jwu/jhumaneval)
* [LLM は日本語追加学習により言語間知識転移を起こすのか？](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/P10-9.pdf)
*/
{
  class_path: 'Generation',
  init_args: {
    eval_dataset: {
      class_path: 'HFGenerationDataset',
      init_args: {
        dataset_name: 'kogi-jwu/jhumaneval',
        split: 'test',
        references_template: '{{ test }}\n\ncheck({{ entry_point }})\n',
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          {{ prompt }}
        |||,
      },
    },
    metrics: [
      { class_path: 'CodeEval', init_args: { code_prompt_template: '{{ prompt }}' } },
    ],
    gen_kwargs: { max_new_tokens: 512, stop_sequences: ['\nclass', '\ndef', '\n#', '\n@', '\nprint', '\nif', '\n```'] },
    batch_size: 4,
  },
}
