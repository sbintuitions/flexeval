/*
JCommonsenseQA is a Japanese version of CommonsenseQA, which is a multiple-choice question answering dataset that requires commonsense reasoning ability.
The dataset is built using crowdsourcing with seeds extracted from the knowledge base ConceptNet.
This is a setup for generating answers based on the choices provided.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/sbintuitions/JCommonsenseQA)
* [Original Repository](https://github.com/yahoojapan/JGLUE)
* [JGLUE: Japanese General Language Understanding Evaluation](https://aclanthology.org/2022.lrec-1.317)
* [JGLUE: 日本語言語理解ベンチマーク](https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf)
*/

local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'sbintuitions/JCommonsenseQA',
    split: 'validation',
    reference_template: '{% set choices = [choice0, choice1, choice2, choice3, choice4] %}{{ choices[label] }}',
  },
};

local template_ = |||
  以下はタスクを説明する指示と、追加の背景情報を提供する入力の組み合わせです。要求を適切に満たす回答を書いてください。
  ### 指示
  質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。回答の他には何も含めないことを厳守してください。

  ### 入力：
  質問：主に子ども向けのもので、イラストのついた物語が書かれているものはどれ？
  選択肢：世界,写真集,絵本,論文,図鑑
  ### 回答：
  絵本

  ### 入力：
  質問：未成年者を監護・教育し，彼らを監督し，彼らの財産上の利益を守る法律上の義務をもつ人は？
  選択肢：浮浪者,保護者,お坊さん,宗教者,預言者
  ### 回答：
  保護者

  ### 入力：
  質問：数字の１を表すときに使う体は？
  選択肢：胸,肉球,背中,人差し指,親指
  ### 回答：
  人差し指

  ### 入力：
  質問：火を起こすとあらわれるもくもくするものは？
  選択肢：歯の変色,ガス,中毒,爆発,煙
  ### 回答：
  煙

  ### 入力：
  質問：{{ question }}
  選択肢：{{ choice0 }},{{ choice1 }},{{ choice2 }},{{ choice3 }},{{ choice4 }}
  ### 回答：
|||;

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args,
    prompt_template: template_,
    metrics: [
      { class_path: 'ExactMatch' },
    ],
    gen_kwargs: { max_new_tokens: 64, stop_sequences: ['\n\n'] },
    batch_size: 1,
  },
}
