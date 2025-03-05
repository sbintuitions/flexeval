# Sarashina2.2 の評価

Sarashina2.2 のモデル評価の設定や結果をまとめておくディレクトリです。
以下、事前学習モデルの評価とチューニングモデルの評価について説明します。

## 事前学習モデルの評価

各評価対象のモデルとベンチマークの設定ファイルが、[./configs/pretrained_models](./configs/pretrained_models) と [./configs/pretrained_evals](./configs/pretrained_evals) に配置されています。
それぞれのパスを指定して、`flexeval_lm` を実行することで、評価を行うことができます。

コマンド例

```bash
flexeval_lm \
  --language_model "examples/evaluate_sarashina_2_2/configs/pretrained_models/sarashina2.2-3b.jsonnet" \
  --eval_setup "examples/evaluate_sarashina_2_2/configs/pretrained_evals/mgsm-ja.jsonnet" \
  --save_dir "./results_pretrained/sarashina2.2-3b/mgsm-ja"
```

評価スコアは `--save_dir` で指定したディレクトリに `metrics.json` として保存されます。

> [!TIP]
> `--eval_setup.batch_size 4` と指定することで、バッチサイズを変更することができます。
> デフォルトは、各ベンチマークの config に記載されている通り 1 です。

> [!Note]
> HumanEval と JHumanEval のタスクに関しては、設定ファイルが２種類用意されています。
> プロンプト末尾の改行があるもの（`*humaneval.jsonnet`）とないもの（`*humaneval-trim.jsonnet`）です。
> これは、モデル毎に適切なトークン区切りが異なり、改行の有無によって評価スコアが大きく変わることがあるからです。
> 評価結果の表では、２つの設定のうち、より高いスコアを示すものを採用しています。

### 評価スコア

Sarashina2.2 を含む同パラメータ帯のモデルの内、各ベンチマークで最も高いスコアを太字で示しています。

| Model                                                                      | niilcqa     | jmmlu       | gsm8k       | mgsm-ja     | openai_humaneval | jhumaneval  |
|----------------------------------------------------------------------------|-------------|-------------|-------------|-------------|------------------|-------------|
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)                   | 2.4         | 25.1        | **38.0**        | 8.4         | **28.7**             | **20.1**        |
| **[sarashina2.2-0.5b](https://huggingface.co/sbintuitions/sarashina2.2-0.5b)** | **33.9**    | **28.8**        | 28.4        | **21.6**        | 22.0             | 15.2        |
|                                                                            |             |             |             |             |                  |             |
| [RakutenAI-2.0-mini](https://huggingface.co/Rakuten/RakutenAI-2.0-mini)    | 19.7        | 27.4        | 4.0         | 2.0         | 9.1              | 7.3         |
| [TinySwallow-1.5B](https://huggingface.co/SakanaAI/TinySwallow-1.5B)       | 20.5        | **39.6**        | 39.0        | 24.4        | 31.7             | 26.2        |
| [plamo2-1b](https://huggingface.co/pfnet/plamo-2-1b)                       | 35.4        | 28.2        | 11.0        | 8.0         | 25.6             | 18.9        |
| [llm-jp-3-1.8b](https://huggingface.co/llm-jp/llm-jp-3-1.8b)               | 28.3        | 27.4        | 2.8         | 3.2         | 0.6              | 0.0         |
| [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)             | 7.9         | 24.9        | 8.3         | 4.4         | 18.9             | 15.2        |
| [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)                   | 9.4         | 39.2        | **61.7**        | 30.4        | **39.0**             | **31.1**        |
| **[sarashina2.2-1b](https://huggingface.co/sbintuitions/sarashina2.2-1b)**     | **47.2**    | 38.2        | 37.8        | **39.6**        | 33.5             | 20.7        |
|                                                                            |             |             |             |             |                  |             |
| [llm-jp-3-3.7b](https://huggingface.co/llm-jp/llm-jp-3-3.7b)               | 45.7        | 30.1        | 5.5         | 6.4         | 1.8              | 0.0         |
| [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)             | 16.5        | 34.3        | 29.9        | 16.8        | 25.6             | 23.8        |
| [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)                       | 13.4        | 52.1        | **77.9**        | 49.2        | 38.4             | **42.7**        |
| **[sarashina2.2-3b](https://huggingface.co/sbintuitions/sarashina2.2-3b)**   | **63.0**    | **52.7**        | 61.7        | **63.6**        | **52.4**             | 39.0        |
|                                                                            |             |             |             |             |                  |             |
| [sarashina2-7b](https://huggingface.co/sbintuitions/sarashina2-7b)         | 61.4        | 42.5        | 12.8        | 8.4         | 15.2             | 12.8        |
| [sarashina2-70b](https://huggingface.co/sbintuitions/sarashina2-70b)       | 65.4        | 62.7        | 71.7        | 54.0        | 28.7             | 22.0        |

（スコア算出時の flexeval バージョンは v0.10.1）

> [!Note]
> スコアは Greedy Decoding (temperature=0) によるもので、本来は実行する度に同じスコアが得られるべきです。
> ですが、モデルの推論に低精度演算（bfloat16 など）を用いており、ハードウェアやバッチサイズが異なると、スコアも変わる現象が見られます。
> これは、内部における数値計算の順序が変わることで、数値計算誤差が蓄積され、モデルの出力が変わるためです。


## チューニングモデルの評価

各評価対象のモデルとベンチマークの設定ファイルが、[./configs/instruction_models](./configs/instruction_models) と [./configs/instruction_evals](./configs/instruction_evals) に配置されています。
それぞれのパスを指定して、`flexeval_lm` を実行することで、評価を行うことができます。

OpenAI API を利用した LLM-as-a-Judge による評価を行うため、`OPENAI_API_KEY` を設定してください。

コマンド例

```bash
export OPENAI_API_KEY=sk-...

flexeval_lm \
  --language_model "examples/evaluate_sarashina_2_2/configs/instruction_models/sarashina2.2-3b-instruct-v0.1.jsonnet" \
  --language_model.default_gen_kwargs.seed 0 \
  --eval_setup "examples/evaluate_sarashina_2_2/configs/instruction_evals/elyza_tasks_100.jsonnet" \
  --save_dir "./results_instruction/sarashina2.2-3b-instruct-v0.1/elyza_tasks_100"
```

### 評価スコア

各評価値は、9回の評価（モデルに3回生成させ、各生成結果に対して3回の評価）の平均値です。
Sarashina2.2 を含む同パラメータ帯のモデルの内、各ベンチマークで最も高いスコアを太字で示しています。

| Model                                                                                                         | Elyza-tasks-100 | Japanese MT Bench | English MT Bench  |
|---------------------------------------------------------------------------------------------------------------|-----------------|-------------------|-------------------|
| [Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)                                    | 1.53            | 2.95              | 4.98              |
| **[sarashina2.2-0.5B-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-0.5b-instruct-v0.1)**    | **2.38**        | **4.55**          | **5.09**          |
|                                                                                                               |                 |                   |                   |
| [RakutenAI-2.0-mini-instruct](https://huggingface.co/Rakuten/RakutenAI-2.0-mini-instruct)                     | 2.41            | 4.49              | 5.13              |
| [TinySwallow-1.5B-Instruct](https://huggingface.co/SakanaAI/TinySwallow-1.5B-Instruct)                        | 2.81            | **5.24**          | 6.31              |
| [Qwen2.5-1.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)                                    | 2.28            | 4.06              | **6.99**          |
| [llm-jp-3-1.8b-instruct3](https://huggingface.co/llm-jp/llm-jp-3-1.8b-instruct3)                              | 2.53            | 4.62              | 4.83              |
| **[sarashina2.2-1B-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-1b-instruct-v0.1)**        | **2.88**        | 5.09              | 6.46              |
|                                                                                                               |                 |                   |                   |
| [gemma-2-2b-jpn-it](https://huggingface.co/google/gemma-2-2b-jpn-it)                                          | 3.02            | 5.19              | 7.56              |
| [Qwen2.5-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)                                        | 2.99            | 5.68              | **7.88**          |
| [llm-jp-3-3.7b-instruct3](https://huggingface.co/llm-jp/llm-jp-3-3.7b-instruct3)                              | 2.79            | 4.98              | 5.44              |
| **[sarashina2.2-3B-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-3b-instruct-v0.1)**        | **3.75**        | **6.51**          | 7.71              |
|                                                                                                               |                 |                   |                   |
| [Qwen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)                                        | 3.53            | 6.57              | 7.24              |
| [llm-jp-3-7.2b-instruct3](https://huggingface.co/llm-jp/llm-jp-3-7.2b-instruct3)                              | 3.24            | 5.57              | 6.02              |
| [Llama-3.1-Swallow-8B-Instruct-v0.3](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3) | 3.72            | 6.55              | 6.65              |

（スコア算出時の flexeval バージョンは v0.10.1）
