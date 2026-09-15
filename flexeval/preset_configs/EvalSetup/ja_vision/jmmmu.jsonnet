/*
JMMMU (Japanese MMMU) is a Japanese benchmark of college-level
multiple-choice questions that require reasoning over images, covering
culture-agnostic subjects translated from MMMU and Japanese-culture-specific
subjects (Japanese Art, Japanese Heritage, Japanese History, World History).

This preset evaluates the multiple-choice questions of all 28 subjects on the
`test` split. Scoring is deterministic:
`last_choice_exact_match` extracts the last mentioned option letter from the response
and compares it with the reference letter; responses without an extractable
letter count as wrong.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/JMMMU/JMMMU)
* [JMMMU: A Japanese Massive Multi-discipline Multimodal Understanding Benchmark for Culture-aware Evaluation](https://arxiv.org/abs/2410.17250)
*/
local subjects = [
  'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Basic_Medical_Science', 'Biology',
  'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine',
  'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Japanese_Art',
  'Japanese_Heritage', 'Japanese_History', 'Manage', 'Marketing', 'Materials',
  'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics',
  'Psychology', 'Public_Health', 'World_History',
];

{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'HFChatDataset',
      init_args: {
        path: 'JMMMU/JMMMU',
        subset: subjects,
        split: 'test',
        keep_conditions: {
          '{{ question_type }}': 'multiple-choice',
        },
        input_template: std.stripChars(|||
          {%- set image_list = [
              image_1_base64,
              image_2_base64,
              image_3_base64,
              image_4_base64,
              image_5_base64,
              image_6_base64,
              image_7_base64
          ] -%}
          [
          {%- for image_base64 in image_list if image_base64 %}
          {"type": "image_url", "image_url": {"url": {{ image_base64 | tojson }} }},
          {%- endfor %}
          { "type": "text", "text": r"""{{ question | regex_replace('<image \d>', '<image>') }}
          {%- for option in options | literal_eval %}
          {{ 'ABCDEFGHIJ'[loop.index0] }}. {{ option | regex_replace('<image \d>', '<image>') }}
          {%- endfor %}

          与えられた選択肢の中から最も適切な回答のアルファベットだけを直接記入してください。
          回答："""},
          ]
        |||, '\n'),
        reference_template: '{{ answer }}',
        parse_input_utterance: 'literal_eval',
        preprocessors: [
                        {
                          class_path: 'EnsureMinSize',
                          init_args: { key: 'image_%d' % i, min_size: 30 },
                        }
                        for i in std.range(1, 7)
                      ]
                      +
                      [
                        {
                          class_path: 'ConvertImageToBase64',
                          init_args: { key: 'image_%d' % i },
                        }
                        for i in std.range(1, 7)
                      ],
      },
    },
    metrics: [
      { class_path: 'OutputLengthStats' },
      { class_path: 'ExactMatch' },
      {
        class_path: 'ExactMatch',
        init_args: {
          lm_output_processor: {
            class_path: 'LastChoiceExtractor',
            init_args: {
              lang: 'ja',
              max_num_options: 5,  // JMMMU contains a small number of 5-choice questions.
            },
          },
          metric_key: 'last_choice_exact_match',
          category_key: 'subset',
        },
      },
    ],
    gen_kwargs: { temperature: 0 },
    batch_size: 1,
  },
}
