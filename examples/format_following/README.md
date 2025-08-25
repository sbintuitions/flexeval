# ユーザープロンプトとシステムプロンプトにおける指示追従性の比較評価

本ディレクトリには [IFEval](https://huggingface.co/datasets/google/IFEval), [M-IFEval](https://github.com/lightblue-tech/M-IFEval) をベースにシステムプロンプトに制約の指示を埋め込むように変換したデータセットが含まれています。
指示をユーザープロンプトに埋め込む場合とシステムプロンプトに埋め込む場合の比較が可能です。\
データは[./data](./data) に、評価の実行に必要な実装・設定ファイルはそれぞれ[./src](./src),[./configs](./configs/)に格納されています。

## 評価結果

色々なモデルの Accuracy の全カテゴリにわたるマイクロ平均を記載します。

|                Model                | ifeval | ifeval_for_system_prompt | m_ifeval | m_ifeval_for_system_prompt |
| :---------------------------------- | -----: | -----------------------: | -------: | -------------------------: |
| sarashina2.2-3b-instruct-v0.1       |   0.48 |                     0.41 |     0.33 |                       0.29 |
| llm-jp-3.1-8x13b-instruct4          |   0.56 |                     0.29 |      0.4 |                       0.31 |
| Stockmark-2-100B-instruct-beta      |   0.56 |                     0.54 |     0.47 |                       0.44 |
| Llama-3.1-Swallow-8B-Instruct-v0.5  |   0.65 |                     0.52 |     0.49 |                       0.38 |
| Llama-3.3-Swallow-70B-Instruct-v0.4 |    0.8 |                     0.54 |     0.59 |                        0.4 |
| qwq-bakeneko-32b                    |   0.83 |                     0.81 |     0.63 |                        0.6 |
| RakutenAI-2.0-8x7B-instruct         |   0.85 |                     0.44 |     0.52 |                       0.22 |
| Qwen3-235B-A22B-FP8_non-thinking    |   0.88 |                     0.85 |     0.67 |                       0.66 |
| gemma-3-27b-it                      |   0.88 |                      0.9 |     0.62 |                       0.65 |
| gemini-2.5-pro                      |   **0.91** |                     **0.91** |     0.77 |                       0.78 |
| gpt-4.1-2025-04-14                  |   **0.91** |                     0.89 |     **0.82** |                       **0.81** |

結果のCSVファイルは [./results](./results) に格納されています。

## 使い方

まず、依存をインストールします

```bash
poetry install --with format_following --extras vllm
```

評価スクリプトを実行します。

```bash
poetry run bash examples/format_following/scripts/eval.sh
poetry run bash examples/format_following/scripts/eval_proprietary.sh
```

## NOTE

### `llm-jp-3.1-8x13b-instruct4` の評価にはカスタムチャットテンプレートが必要です

`llm-jp-3.1-8x13b-instruct4` のチャットテンプレートは、システムがロールとなっているメッセージを無視して予め決められたシステムプロンプトを埋め込む仕様となっています。
これを避けるため、システムプロンプトを扱えるようなチャットテンプレートを `examples/format_following/resource/llm-jp-3.1-8x13b-instruct4_tokenizer_config.json` に用意しました。
該当モデルを評価する際はこの設定ファイルを使用してください。

## Reference

```text
@misc{zhou2023instructionfollowingevaluationlargelanguage,
    title={Instruction-Following Evaluation for Large Language Models},
    author={Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
    year={2023},
    eprint={2311.07911},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2311.07911},
}
```

```text
@article{Dussolle2025MIFEval,
    title={M-IFEval: Multilingual Instruction-Following Evaluation},
    author={Antoine Dussolle and Andrea Cardena Díaz and Shota Sato and Peter Devine},
    year={2025},
    journal={arXiv preprint},
    volume={arXiv:2502.04688},
    eprint={2502.04688},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2502.04688}
}
```
