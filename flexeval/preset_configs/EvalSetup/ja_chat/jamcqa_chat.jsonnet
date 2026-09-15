/*
# JamC-QA (for instruction/chat models)

This benchmark evaluates knowledge specific to Japan through multiple-choice questions.
It covers eight categories: culture, custom, regional_identity, geography, history, government, law, and healthcare.
Achieving high performance requires broad and detailed understanding of Japan across these categories.

Although this benchmark was originally designed for pre-trained models, this configuration is provided for instruction/chat models.
We prompt the model to generate responses in a zero-shot setting and calculate the exact-match rate.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/sbintuitions/JamC-QA)
*/


{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'HFChatDataset',
      init_args: {
        path: 'sbintuitions/JamC-QA',
        input_template: std.stripChars(
          |||
            以下の質問に回答してください。最後に「答え: (E)」のように、「答え: (ここにどの選択肢を選んだか書く)」というフォーマットで回答してください。

            {{question}}

            (A) {{ choice0 }}
            (B) {{ choice1 }}
            (C) {{ choice2 }}
            (D) {{ choice3 }}
          |||, '\n'
        ),
        reference_template: '{% set choices = ["A", "B", "C", "D"] %}{{ choices[answer_index] }}',
        split: 'test',
        subset: 'v1.0'
      },
    },
    metrics: [
      {
        class_path: 'ExactMatch',
        init_args: {
          lm_output_processor: [
            { class_path: 'RegexExtractor', init_args: { pattern: '^(?:.*</think>\\s*)?(.*)$' } },
            { class_path: 'RegexExtractor', init_args: { pattern: '答え: \\((.)\\)' } },
          ],
        },
      },
      { class_path: 'flexeval.core.metric.repetition_n.RepetitionN' },
    ],
    gen_kwargs: { max_new_tokens: 8192 },
    batch_size: 4,
  },
}
