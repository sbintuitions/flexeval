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

以下の例では、`aio` （[AI王](https://sites.google.com/view/project-aio/home)）というクイズに回答するタスクを使って、hugging face モデル `sbintuitions/tiny-lm` を評価します。

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model_name "sbintuitions/tiny-lm" \
  --eval_setup "aio" \
  --save_dir "results/aio"
```
（上に挙げているモデルはあくまでデバッグ用であり、性能は全く期待できません。お好きなモデルに切り替えて試してみてください！）

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
