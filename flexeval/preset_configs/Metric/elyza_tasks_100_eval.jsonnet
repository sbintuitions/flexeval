/*
This is a config of the ChatLLMScore class designed to evaluate chat assistants with [ELYZA-tasks-100](https://huggingface.co/datasets/elyza/ELYZA-tasks-100).
The template is adapted from the blog post [ELYZAが公開した日本語LLM「ELYZA-japanese-Llama-2-7b」についての解説 : (2) 評価編](https://zenn.dev/elyza/articles/5e7d9373c32a98).
*/
{
  class_path: 'ChatLLMScore',
  init_args: {
    language_model: { class_path: 'OpenAIChatAPI', init_args: { model: 'gpt-4-turbo-2024-04-09' } },
    valid_score_range: [1, 5],
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: std.stripChars(|||
          あなたは採点者です。

          問題, 正解例, 採点基準, 回答 が与えられます。

          採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

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

          # 回答
          {{ lm_output }}
        |||, '\n'),
      },
    },
  },
}
