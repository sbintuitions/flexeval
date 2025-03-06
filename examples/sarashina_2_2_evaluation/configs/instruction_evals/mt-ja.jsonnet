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
      {
        class_path: 'ChatLLMScore',
        init_args: {
          language_model: { class_path: 'OpenAIChatBatchAPI', init_args: { model: 'gpt-4o-2024-08-06' } },
          valid_score_range: [1, 10],
          prompt_template: {
            class_path: 'Jinja2PromptTemplate',
            init_args: {
              template: std.stripChars(|||
                {% if references|length > 0 -%}
                <|参照回答の開始|>
                ### ユーザ:
                {{ messages[0]["content"] }}

                ### 参照回答:
                {{ references[0] }}

                ### ユーザ:
                {{ messages[2]["content"] }}

                ### 参照回答:
                {{ references[1] }}

                <|参照回答の修了|>
                {% endif -%}

                <|アシスタントAとユーザの対話の開始|>

                ### ユーザ:
                {{ messages[0]["content"] }}

                ### アシスタントA:
                {{ messages[1]["content"] }}

                ### ユーザ:
                {{ messages[2]["content"] }}

                ### アシスタントA:
                {% if messages|length == 3 %}{{ lm_output }}{% else %}{{ messages[3]["content"] }}{% endif %}

                <|アシスタントAとユーザの対話の終了|>
              |||, '\n'),
            },
          },
          system_message: {
            class_path: 'Jinja2PromptTemplate',
            init_args: {
              template: std.stripChars(|||
                {% if references|length > 0 -%}
                以下に表示されるユーザの質問に対するアシスタントの応答の品質を評価してください。評価は正確さと有用性を考慮すべきです。アシスタントの回答の言語は、ユーザが使用している言語と一致しているべきで、そうでない場合は減点されるべきです。参照回答とアシスタントの回答が与えられます。ユーザの２つ目の質問に対するアシスタントの応答の品質について評価してください。あなたの評価は、アシスタントの回答と参照回答を比較することから始めてください。ミスを特定し、訂正してください。できるだけ客観的であること。評価の説明をした後、"[[rating]]"という形式で、1から10までの整数の評価値を出力してください（例 "rating：[[5]]"）。
                {%- else -%}
                以下に表示されるユーザの質問に対するアシスタントの応答の品質を公平に評価してください。評価は、応答の有用性、関連性、正確性、深さ、創造性、詳細度などの要素を考慮すべきです。アシスタントの回答の言語は、ユーザが使用している言語と一致しているべきで、そうでない場合は減点されるべきです。ユーザの２つ目の質問に対するアシスタントの応答の品質について評価してください。評価は短い説明から始めてください。できるだけ客観的であること。評価の説明をした後、"[[rating]]"という形式で、1から10までの整数の評価値を出力してください（例 "rating：[[5]]"）。
                {%- endif %}
              |||, '\n'),
            },
          },
          category_key: 'category',
        },
      },
    ],
    gen_kwargs: { max_new_tokens: 1024 },
    batch_size: 1,
  },
}
