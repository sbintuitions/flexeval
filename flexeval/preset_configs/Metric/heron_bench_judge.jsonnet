/*
LLM-as-a-judge scoring for Japanese-Heron-Bench.

Scores each response on a 1-5 scale with an OpenAI judge model, given the
image context description, the question, and the bundled reference answer
(gpt-4-0125-preview). Reports per-category mean scores (`category` is one of
complex / conversation / detail).

Requires OPENAI_API_KEY at run time. Judge-based scores depend on the judge
model; report the judge model together with the numbers.

Usage (after running the `heron_bench` EvalSetup preset):

    flexeval_file \
      --eval_file <save_dir>/outputs.jsonl \
      --metrics heron_bench_judge \
      --save_dir <judge_save_dir>
*/
{
  class_path: 'ChatLLMScore',
  init_args: {
    language_model: { class_path: 'OpenAIChatAPI', init_args: { model: 'gpt-4o-2024-08-06' } },
    valid_score_range: [1, 5],
    category_key: 'category',
    system_message: 'You are an expert evaluator.',
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: std.stripChars(|||
          You are an expert evaluator. You are given a set of (Context, Question, Reference, Prediction). Your task is to evaluate the quality of the Prediction as the response to the given Context and Question, compared with the Reference as the baseline response.

          Please assign a score from 1 to 5 based on the following criteria:

          5: Excellent — The Prediction is much more relevant and correct than the Reference.
          4: Good — The Prediction is more relevant and correct than the Reference.
          3: Fair — The Prediction is almost equally relevant and correct compared with the Reference.
          2: Poor — The Prediction is less relevant and correct than the Reference.
          1: Very Poor — The Prediction is much less relevant and correct than the Reference.
          Output only the score (an integer from 1 to 5). Do not add any explanation.

          [Given data]
          Context: {{ context }}
          Question: {{ text }}
          Reference: {{ references[0] }}
          Prediction: {{ lm_output }}

          Your Score:
        |||, '\n'),
      },
    },
  },
}
