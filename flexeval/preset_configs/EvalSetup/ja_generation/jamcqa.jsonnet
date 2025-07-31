/*
# JamC-QA

JamC-QA is a multiple-choice (4 options) question-answering benchmark specializing in Japan-specific knowledge such as Japanese culture and customs.
All questions were created from scratch by hand, not translated from existing English benchmarks.
JamC-QA includes 8 categories with 2,341 questions.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/sbintuitions/JamC-QA)
*/

local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'sbintuitions/JamC-QA',
    subset: 'v1.0',
    split: 'test',
    reference_template: '{% set choices = [choice0, choice1, choice2, choice3] %}{{ choices[answer_index] }}',
  },
};

local template_ = |||
  以下はタスクを説明する指示と、追加の背景情報を提供する入力の組み合わせです。要求を適切に満たす回答を書いてください。
  指示:: 質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。回答の他には何も含めないことを厳守してください。
  
  質問:: 西暦1989年1月7日を和暦で書くと何年何月何日?, 選択肢::
   昭和62年1月7日
   昭和63年1月7日
   昭和64年1月7日
   平成元年1月7日, 回答:: 昭和64年1月7日
  質問:: 正月飾りの門松をしまうことを一般的に何と呼ぶか選べ, 選択肢::
   松納め
   門松終い
   松終い
   門納め, 回答:: 松納め
  質問:: 「神戸」を「かんべ」と読む土地は次のうちどれ?, 選択肢::
   愛知県田原市神戸町
   兵庫県神戸市
   群馬県高崎市神戸町
   岡山県津山市神戸, 回答:: 愛知県田原市神戸町
  質問:: 肉食禁止令を出した人は誰か選択肢から選べ, 選択肢::
   弘文天皇
   天武天皇
   昭和天皇
   後光明天皇, 回答:: 天武天皇
  質問:: コンビニエンスストアで発行できない証明書はどれ, 選択肢::
   住民票
   戸籍謄本
   住民票記載事項証明書
   戸籍記載事項証明書, 回答:: 戸籍記載事項証明書
  質問:: 「占有離脱物横領罪」が成立した場合の罰金は何万円以下か選べ, 選択肢::
   3万円
   5万円
   10万円
   20万円, 回答:: 10万円
  質問:: 日本の市町村が予防接種を行う定期接種の対象疾患であるA類疾病でないものはどれか?, 選択肢::
   破傷風
   ポリオ
   結核
   インフルエンザ, 回答:: インフルエンザ
  質問:: {{ question }}, 選択肢::
   {{ choice0 }}
   {{ choice1 }}
   {{ choice2 }}
||| + ' {{ choice3 }}, 回答::';

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args,
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: { template: template_, },
    },
    metrics: [
      { class_path: 'ExactMatch',
        init_args: {
          lm_output_processor: [ 
            { class_path: 'NFKCNormalizer'},
            { class_path: 'StringStrip', },
          ],
          reference_processor: [ 
            { class_path: 'NFKCNormalizer'},
            { class_path: 'StringStrip', },
          ],
        },
      },
    ],
    gen_kwargs: { max_new_tokens: 128, stop_sequences: ['\n'], },
    batch_size: 1,
  },
}
