/*
Japanese question answering dataset for explainable QA systems.

References:

* [Homepage](https://mynlp.is.s.u-tokyo.ac.jp/niilc-qa/)
* [GitHub](https://github.com/mynlp/niilc-qa)
* [百科事典を対象とした質問応答システムの開発](https://www.anlp.jp/proceedings/annual_meeting/2003/pdf_dir/C7-6.pdf)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'sbintuitions/niilc-qa',
    subset: 'v1.2',
    split: 'test',
    keep_conditions: { '{{ D_3 | length }}': '1', '{{ D_3[0] }}': '唯一' },  //https://github.com/mynlp/niilc-qa/blob/master/data/NIILC-ECQA2015_AnnotationDefinition.md
    reference_list_template: '{{ answers }}',
  },
};

local template_ = |||
  以下はタスクを説明する指示と、追加の背景情報を提供する入力の組み合わせです。要求を適切に満たす回答を書いてください。
  指示:: 質問に対する答えを出力してください。

  質問:: ハリーポッターの著者は誰？, 回答:: J・K・ローリング
  質問:: 軽自動車は何cc以下から？, 回答:: 660cc以下
  質問:: 向日葵は何科の植物？, 回答:: キク科
  質問:: 入道雲はどの季節に発生するの？, 回答:: 夏
||| + '質問:: {{ text }}, 回答::';

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args,
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: { template: template_ },
    },
    metrics: [
      {
        class_path: 'CharF1',
        init_args: {
          lm_output_processor: { class_path: 'AIONormalizer' },
          reference_processor: { class_path: 'AIONormalizer' },
        },
      },
      {
        class_path: 'ExactMatch',
        init_args: {
          lm_output_processor: { class_path: 'AIONormalizer' },
          reference_processor: { class_path: 'AIONormalizer' },
        },
      },
    ],
    gen_kwargs: { max_new_tokens: 64, stop_sequences: ['\n'] },
    batch_size: 1,
  },
}
