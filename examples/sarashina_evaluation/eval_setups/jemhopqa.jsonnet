local template_comparison_choice = |||
  以下はタスクを説明する指示と、追加の背景情報を提供する入力の組み合わせです。要求を適切に満たす回答を書いてください。
  ### 指示
  質問を入力とし、回答を出力してください。回答の他には何も含めないことを厳守してください。

  ### 入力：
  質問：『仮面ライダー電王』と『あまちゃん』、放送回数が多いのはどちらでしょう？
  ### 導出：
  (仮面ライダー電王, 放送回数, 49), (あまちゃん, 放送回数, 156)
  ### 回答：
  あまちゃん

  ### 入力：
  質問：清原果耶と森田成一で活動開始が早かったのはどちらですか？
  ### 導出：
  (清原果耶, 活動開始時, 2015年), (森田成一, 活動開始時, 2001年)
  ### 回答：
  森田成一

  ### 入力：
  質問：映画『ダイ・ハード』と『デッドプール2』のうち、公開年が早いほうはどっち？
  ### 導出：
  (ダイ・ハード, 公開年, 1988年), (デッドプール2, 公開年, 2018年)
  ### 回答：
  ダイ・ハード

  ### 入力：
  質問：漫画『テセウスの船』と『35歳の高校生』はどちらの話数が多いでしょうか？
  ### 導出：
  (テセウスの船 (漫画), 話数, 89話), (35歳の高校生, 話数, 11話)
  ### 回答：
  テセウスの船

  ### 入力：
  質問：{{question}}
  ### 導出：
|||;

local template_yes_not = |||
  以下はタスクを説明する指示と、追加の背景情報を提供する入力の組み合わせです。要求を適切に満たす回答を書いてください。
  ### 指示
  質問を入力とし、回答を出力してください。回答の他には何も含めないことを厳守してください。

  ### 入力：
  質問：『ぼくらが旅に出る理由』と『ボーン・トゥ・ラヴ・ユー』はどちらも小沢健二のシングルですか？
  ### 導出：
  (ぼくらが旅に出る理由,　企画・制作,　小沢健二), (ボーン・トゥ・ラヴ・ユー,　企画・制作,　フレディ・マーキュリー)
  ### 回答：
  NO

  ### 入力：
  質問：日立製作所と三菱重工業はどちらも会計監査人は同じですか？
  ### 導出：
  (日立製作所,　会計監査人,　EY新日本有限責任監査法人), (三菱重工業,　会計監査人,　有限責任あずさ監査法人)
  ### 回答：
  NO

  ### 入力：
  質問：島津氏と平氏は、どちらも日本の氏族ですか？
  ### 導出：
  (島津氏,　出自集団,　武家・華族だった日本の氏族), (平氏, 出自集団,　日本の平姓（たいらのかばね）を持つ氏族)
  ### 回答：
  YES

  ### 入力：
  質問：『マツケンサンバ』と『RADIO GA GA』はどちらもミュージックビデオがありますか？
  ### 導出：
  (マツケンサンバ,　ミュージックビデオ,　【公式】松平健「マツケンサンバⅡ」MV-YouTube), (RADIO GA GA,　ミュージックビデオ,　「RADIO GA GA」-YouTube)
  ### 回答：
  YES

  ### 入力：
  質問：{{question}}
  ### 導出：
|||;

local template_compositioal = |||
  以下はタスクを説明する指示と、追加の背景情報を提供する入力の組み合わせです。要求を適切に満たす回答を書いてください。
  ### 指示
  質問を入力とし、回答を出力してください。回答の他には何も含めないことを厳守してください。

  ### 入力：
  質問：孝明天皇が生涯過ごした都に以前の都から遷都があった年は？
  ### 導出：
  (孝明天皇, 生涯を過ごした都, 平安京), (平安京, 遷都された年, 794年)
  ### 回答：
  794年

  ### 入力：
  質問：無名塾の主宰者の誕生日の年月日は？
  ### 導出：
  (無名塾, 主宰者, 仲代達矢), (仲代達矢, 生年月日, 1932年12月13日)
  ### 回答：
  1932年12月13日

  ### 入力：
  質問：川島明が所属する事務所がある都道府県は？
  ### 導出：
  (川島明, 所属事務所, 吉本興業), (吉本興業, 本社所在地, 大阪府大阪市中央区(大阪市)難波千日前)
  ### 回答：
  大阪府

  ### 入力：
  質問：『99.9-刑事専門弁護士-』のSEASON Iのヒロイン役で出演した俳優の所属事務所はどこですか？
  ### 導出：
  (99.9-刑事専門弁護士-, SEASON Iのヒロイン役の出演者, 榮倉奈々), (榮倉奈々, 事務所, 研音)
  ### 回答：
  研音

  ### 入力：
  質問：{{question}}
  ### 導出：
|||;

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: {
      class_path: 'HFGenerationDataset',
      init_args: {
        path: 'sbintuitions/JEMHopQA',
        subset: 'v1.1',
        reference_template: '{{ answer }}',
        split: 'validation',
        keep_conditions: { "{{ time_dependent }}": "False" }
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: "{% if type == 'compositional' %}" + template_compositioal + "{% elif answer in ['YES', 'NO'] %}" + template_yes_not + '{% else %}' + template_comparison_choice + '{% endif %}',
      },
    },
    metrics: [
      {
        class_path: 'CharF1',
        init_args: {
          processor: [{ class_path: 'LastLineExtractor' }, { class_path: 'AIONormalizer' }],
          reference_processor: { class_path: 'AIONormalizer' },
        },
      },
      {
        class_path: 'ExactMatch',
        init_args: {
          processor: [{ class_path: 'LastLineExtractor' }, { class_path: 'AIONormalizer' }],
          reference_processor: { class_path: 'AIONormalizer' },
        },
      },
    ],
    gen_kwargs: {
      max_new_tokens: 256,
      stop_sequences: ['\n\n'],
      do_sample: false,
    },
    batch_size: 1,
  },
}
