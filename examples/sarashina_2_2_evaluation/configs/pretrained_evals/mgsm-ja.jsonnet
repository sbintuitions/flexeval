/*
Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems.
This is a Japanese subset of the benchmark.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/juletxara/mgsm)
* [Language Models are Multilingual Chain-of-Thought Reasoners](https://arxiv.org/abs/2210.03057)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'sbintuitions/MGSM_ja',
    split: 'test',
    subset: 'fixed',
    reference_template: '{{ answer_number|string }}',
  },
};

local template_ = |||
  問題: ロジャーは5個のテニスボールがあります。テニスボールの缶を2つ追加で買います。それぞれの缶には3つのテニスボールが入っています。彼は今いくつのテニスボールがありますか？
  ステップごとの答え: ロジャーは最初5個のボールがありました。テニスボール3個入りの缶が2つあれば、テニスボールは6個あります。5+6=11。答えは11です。

  問題: サーバールームには9台のコンピューターがありました。月曜日から木曜日まで毎日5台のコンピューターをインストールしました。今サーバールームには難題のコンピューターがありますか？
  ステップごとの答え: 月曜から木曜まで4日あります。毎日5台のコンピューターが追加されます。つまり、全部で4*5=20台のコンピューターが追加されました。最初に9台のコンピューターがあったので、今は9+20=29台のコンピューターとなります。答えは29です。

  問題: リアは32個のチョコレートを持っていました、彼女の妹は42個持っていました。彼女達が35個食べたとしたら、全部で何個残っていますか？
  ステップごとの答え: リアは32個のチョコレートを持っており、リアの妹は42個持っていた。つまり、元々32+42=74個のチョコレートがあった。35個が食べられた。だから全部で、彼女たちには74-35=39個のチョコレートが残っている。答えは39個。

  問題: ショーンは5個のおもちゃを持っています。クリスマスに、彼は父と母からそれぞれ2つずつおもちゃをもらいました。今彼はいくつのおもちゃがありますか？
  ステップごとの答え: 彼は5個おもちゃを持っています。母から2つもらった後は、5+2=7個のおもちゃがあります。その後父から2つさらにもらい、全部で彼は、7+2=9個のおもちゃがあります。答えは9個です。

  問題: {{ question }}
||| + 'ステップごとの答え:';

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
        class_path: 'ExactMatch',
        init_args: {
          lm_output_processor: [
            {
              class_path: 'flexeval.core.string_processor.SimpleEvalMGSMProcessor',
              init_args: {
                answer_prefix: '答えは',
              },
            },
          ],
          reference_processor: [{ class_path: 'flexeval.core.string_processor.RemoveCommaProcessor' }],
        },

      },
      { class_path: 'flexeval.core.metric.MathVerify' },
    ],
    gen_kwargs: { max_new_tokens: 512, stop_sequences: ['\n\n'] },
    batch_size: 1,
  },
}
