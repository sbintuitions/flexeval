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
      {
        class_path: 'ChatLLMScore',
        init_args: {
          language_model: { class_path: 'OpenAIChatAPI', init_args: { model: 'gpt-4o-2024-08-06' } },
          valid_score_range: [1, 5],
          prompt_template: {
            class_path: 'Jinja2PromptTemplate',
            init_args: {  // https://soysoftware.sakura.ne.jp/archives/3850 よりテンプレートを拝借
              template: std.stripChars(|||
                あなたは言語モデルの採点者です。

                問題, 正解例, 採点基準, 言語モデルが生成した回答が与えられます。

                「採点基準」と「正解例」を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

                # 問題
                {{ messages[-1]["content"] }}

                # 正解例
                {{ references[0] }}

                # 採点基準
                基本的な採点基準
                - 1点: 誤っている、 指示に従えていない
                - 2点: 誤っているが、方向性は合っている
                - 3点: 部分的に誤っている、 部分的に合っている
                - 4点: 合っている
                - 5点: 役に立つ

                基本的な減点項目
                - 不自然な日本語: -1点
                - 部分的に事実と異なる内容を述べている: -1点
                - 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

                問題固有の採点基準
                {{ eval_aspect }}

                # 言語モデルの回答
                {{ lm_output }}
              |||, '\n'),
            },
          },
        },
      },
    ],
    gen_kwargs: { max_new_tokens: 1024 },
    batch_size: 1,
  },
}
