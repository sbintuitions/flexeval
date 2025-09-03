# FlexEval

![logo](docs/assets/logo.png)

**Flexible evaluation tool for language models. Easy to extend, highly customizable!**

<h4 align="center">
    <p>
        <a href="https://github.com/sbintuitions/flexeval/">English</a> |
        <b>日本語</b> |
    </p>
</h4>

FlexEval は言語モデル評価のためのツールです。以下のような評価を行うことができます。

* Zero/few-shot の文脈内学習タスク
* チャットボット用の応答生成ベンチマーク（例：[Japanese MT-Bench](https://github.com/Stability-AI/FastChat/tree/jp-stable/fastchat/llm_judge)）の GPT-4 による自動評価
* 対数尤度に基づく多肢選択タスク
* テキストのパープレキシティの計算

その他のユースケースについては、[ドキュメント（英語）](https://sbintuitions.github.io/flexeval/)を参照してください。

## 主な特徴

* **柔軟性**: `flexeval` は柔軟に評価時の設定や評価対象の言語モデルを指定することができます。
* **モジュール性**: `flexeval` 内の主要モジュールは容易に拡張・置換可能です。
* **明瞭性**: 評価結果は明瞭に、全ての詳細が保存されます。
* **再現性**: `flexeval` は再現性を重視しており、評価スクリプトによって保存された設定ファイルから、同じ評価を再現することができます。

## インストール

```bash
pip install flexeval
```

## クイックスタート

以下の例では、`aio` （[AI王](https://sites.google.com/view/project-aio/home)）というクイズに回答するタスクを使って、hugging face モデル `sbintuitions/sarashina2.2-0.5b` を評価します。

### 実行コマンド

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model "sbintuitions/sarashina2.2-0.5b" \
  --eval_setup "aio" \
  --save_dir "results/aio"
```

### 実行結果
```
...
2025-09-03 16:18:12.101 | INFO     | flexeval.core.evaluate_generation:evaluate_generation:92 - {'char_f1': 0.4991599999999999, 'exact_match': 0.38, 'finish_reason_ratio-stop': 0.999, 'finish_reason_ratio-length': 0.001, 'avg_output_length': 4.986, 'max_output_length': 70, 'min_output_length': 1}
...
```

`--saved_dir` に結果が保存され、ファイル毎に以下の内容が含まれます。

* `config.json`: 評価の設定ファイルです。評価設定の確認、結果の再現に使用することができます。
* `metrics.json`: 評価値を含むファイルです。
* `outputs.jsonl`: 評価データセット内の各事例の入出力や、事例単位の評価値が含まれます。

コマンドライン引数や設定ファイルを指定することで、柔軟に評価設定を指定することができます。

[Transformers](https://github.com/huggingface/transformers) のモデルの他にも、[OpenAI ChatGPT](https://openai.com/index/openai-api/) や [vLLM](https://github.com/vllm-project/vllm) などの評価もサポートしています。
また、実装されていないモデルも容易に追加することができるはずです！

## 次のステップ
* `flexeval_presets` のコマンドで、`aio` 以外の利用可能な評価設定のプリセットを確認できます。一覧は [Preset Configs](https://sbintuitions.github.io/flexeval/preset_configs/) で確認できます。
* その他のタイプの評価設定に関しては [Getting Started](https://sbintuitions.github.io/flexeval/getting_started/) をご覧ください。
* 評価設定の詳細な指定方法については [Configuration Guide](https://sbintuitions.github.io/flexeval/configuration_guide/) を参照してください。
* [Sarashina の評価](./examples/sarashina_evaluation)で、実際の設定ファイルの例を確認できます。
