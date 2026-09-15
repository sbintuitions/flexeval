/*
Japanese-Heron-Bench is a Japanese open-ended VQA benchmark of 103 questions
over 21 images (anime, landmarks, documents, ...), each with an image context
description and reference answers from strong closed models.

This preset runs inference and rule-based metrics (BLEU / ROUGE). Heron-Bench
is primarily judged with an LLM: after inference, score the saved outputs with
the companion Metric preset `heron_bench_judge` via `flexeval_file`, e.g.:

    flexeval_file \
      --eval_file <save_dir>/outputs.jsonl \
      --metrics heron_bench_judge \
      --save_dir <judge_save_dir>

The reference answer is the gpt-4-0125-preview response bundled in the dataset,
following the original Heron-Bench evaluation.

Note on image fidelity: the source images are JPEG. `ConvertImageToBase64`
re-encodes the decoded pixels as PNG, which is lossless with respect to the
decoded JPEG but produces different bytes (and larger payloads) than sending
the original files, so scores can differ slightly from pipelines that pass the
original JPEG bytes. Re-encoding with `format: 'JPEG'` instead would compress
lossily a second time and further degrade image quality; keep PNG unless
payload size forces otherwise.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/Silviase/Japanese-Heron-Bench) ([original](https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench))
* [Heron-Bench: A Benchmark for Evaluating Vision Language Models in Japanese](https://arxiv.org/abs/2404.07824)
*/
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'HFChatDataset',
      init_args: {
        path: 'Silviase/Japanese-Heron-Bench',
        split: 'train',
        input_template: '[{ "type": "image_url", "image_url": {"url": "{{ image_base64 }}"}}, { "type": "text", "text": """{{ text }}"""},]',
        reference_template: "{{ answer['gpt-4-0125-preview'] }}",
        parse_input_utterance: 'literal_eval',
        preprocessors: [
          {
            // Source images are JPEG; PNG re-encoding is decoded-pixel-lossless
            // (see the header note on image fidelity).
            class_path: 'ConvertImageToBase64',
            init_args: { key: 'image' },
          },
        ],
      },
    },
    metrics: [
      { class_path: 'BLEU', init_args: { tokenize_option: 'ja-mecab' } },
      {
        class_path: 'ROUGE',
        init_args: {
          tokenizer: { class_path: 'SacreBleuTokenizer', init_args: { name: 'ja-mecab' } },
          max_output_tokens: 1024,
          recursion_limit: 3000,
        },
      },
    ],
    gen_kwargs: { temperature: 0 },
    batch_size: 1,
  },
}
