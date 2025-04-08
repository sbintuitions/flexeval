/*
Multi-Turn Benchmark for large language models.

References:

* [Data Source](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
* [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
*/
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'ChatbotBench',
      init_args: {
        path_or_name: 'mt-en',
        ref_path_or_name: 'mt-en-ref-gpt4',
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
                <|The Start of Reference Answer|>
                ### User:
                {{ messages[0]["content"] }}

                ### Reference answer:
                {{ references[0] }}

                ### User:
                {{ messages[2]["content"] }}

                ### Reference answer:
                {{ references[1] }}

                <|The End of Reference Answer|>
                {% endif -%}

                <|The Start of Assistant A's Conversation with User|>

                ### User:
                {{ messages[0]["content"] }}

                ### Assistant A:
                {{ messages[1]["content"] }}

                ### User:
                {{ messages[2]["content"] }}

                ### Assistant A:
                {% if messages|length == 3 %}{{ lm_output }}{% else %}{{ messages[3]["content"] }}{% endif %}

                <|The End of Assistant A's Conversation with User|>
              |||, '\n'),
            },
          },
          system_message: {
            class_path: 'Jinja2PromptTemplate',
            init_args: {
              template: std.stripChars(|||
                {% if references|length > 0 -%}
                Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. You evaluation should focus on the assistant's answer to the second question. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
                {%- else -%}
                Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
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
