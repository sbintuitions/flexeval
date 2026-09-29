/*
LLM-as-a-judge metric for the `j_multi_challenge` eval setup.

The judge checks whether the last assistant response satisfies the YES/NO check question (`TARGET_QUESTION`),
and `llm_score` is the pass rate. Category-wise pass rates are reported as
`llm_score/AXIS/<AXIS>` and `llm_score/REQUIRES_EXTERNAL_KNOWLEDGE/<True|False>`.

The prompt is a Japanese translation of the official evaluator's prompt:
https://github.com/ekwinox117/multi-challenge/blob/5ccefcca6a39020d66c1383c4e6a809cb07afa33/src/evaluator.py

The default judge is the open-weight model `openai/gpt-oss-120b`, served locally with `VLLMServeLM`
(requires the `vllm` extra and GPUs), so that the results do not depend on the availability of a proprietary API.
To use an already running OpenAI-compatible server instead, override the judge with `OpenAIChatAPI`, e.g.

  --metrics.init_args.language_model OpenAIChatAPI \
  --metrics.init_args.language_model.init_args.model openai/gpt-oss-120b \
  --metrics.init_args.language_model.init_args.api_headers.base_url http://localhost:8000/v1
*/
{
  class_path: 'ChatLLMLabel',
  init_args: {
    language_model: {
      class_path: 'VLLMServeLM',
      init_args: { model: 'openai/gpt-oss-120b', default_gen_kwargs: { temperature: 0.0 } },
    },
    label_names: ['YES', 'NO'],
    label_points: [1.0, 0.0],
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: std.stripChars(|||
          あなたは、モデルの回答が特定の基準を満たしているかどうかを評価する役割を担っています。
          基準は常に「YES」または「NO」の2択評価となります。

          モデルの回答は以下の通りです：
          <MODEL_RESPONSE>
          {{ lm_output }}
          </MODEL_RESPONSE>

          モデルの回答が満たすべき基準は以下の通りです。**厳格に**評価してください：
          <CRITERIA>
          {{ TARGET_QUESTION }}
          </CRITERIA>

          まずあなたの推論（評価理由）を出力し、その後に最終判定として「YES」または「NO」のいずれかを出力してください。
        |||, '\n'),
      },
    },
    category_key: ['AXIS', 'REQUIRES_EXTERNAL_KNOWLEDGE'],
  },
}
