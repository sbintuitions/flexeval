{
  class_path: 'Generation',
  init_args: {
    eval_dataset: {
      class_path: 'HFGenerationDataset',
      init_args: {
        path: 'sbintuitions/niilc-qa',
        reference_list_template: '{{ answers }}',
        split: 'test',
        keep_conditions: { "{{ answers | length }}": "1" }
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          以下はタスクを説明する指示と、追加の背景情報を提供する入力の組み合わせです。要求を適切に満たす回答を書いてください。
          ### 指示
          質問に対する答えを出力してください。

          ### 入力：
          質問：ワールドカップは何年に一度開催されるの？
          ### 回答：
          4年

          ### 入力：
          質問：サッカーのイエローカードはいつ出るの？
          ### 回答：
          非紳士的行為等を行ったとき

          ### 入力：
          質問：ハリーポッターの著者は誰？
          ### 回答：
          J・K・ローリング

          ### 入力：
          質問：携帯電話の身体に与える影響は？
          ### 回答：
          発がん性のリスクが高まる

          ### 入力：
          質問：{{ question }}
          ### 回答：
        |||,
      },
    },
    metrics: [
      { class_path: 'CharF1', init_args: { processor: { class_path: 'AIONormalizer' } } },
      { class_path: 'ExactMatch', init_args: { processor: { class_path: 'AIONormalizer' } } },
    ],
    gen_kwargs: { max_new_tokens: 256, stop_sequences: ['\n\n'], do_sample: false },
    batch_size: 1,
  },
}
