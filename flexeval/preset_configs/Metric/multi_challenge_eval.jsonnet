/*
LLM-as-a-judge metric for the `multi_challenge` eval setup.

The judge checks whether the last assistant response satisfies the YES/NO check question (`target_question`),
and `llm_score` is the pass rate. Category-wise pass rates are reported as `llm_score/axis/<AXIS>`.

The prompt follows the official evaluator:
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
          You are tasked with evaluating a model response to see if it meets a specific criteria.
          The criteria will always be YES/NO evaluation.

          The model response is as follows:
          <MODEL_RESPONSE>
          {{ lm_output }}
          </MODEL_RESPONSE>

          The criteria that the model response must meet is as follows. Be VERY STRICT!:
          <CRITERIA>
          {{ target_question }}
          </CRITERIA>

          Print your reasoning followed by your verdict, either "YES" or "NO".
        |||, '\n'),
      },
    },
    category_key: 'axis',
  },
}
