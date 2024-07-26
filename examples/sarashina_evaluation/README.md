# Sarashina の評価

[SB Intuitions のテックブログで行った評価](https://www.sbintuitions.co.jp/blog/entry/2024/07/26/150005)を再現するための設定ファイルをまとめたディレクトリです。

モデルと評価設定の設定ファイルを以下のように設定して `flexeval_lm` のコマンドを実行すると、評価が実行されます。
```bash
flexeval_lm \
  --language_model "examples/sarashina_evaluation/models/sarashina2-7b.jsonnet" \
  --eval_setup "examples/sarashina_evaluation/eval_setups/aio.jsonnet" \
  --save_dir "results/sarashina2-7b/aio"
```

### 各モデル × データセットの評価値

| モデル            | aio (exact_match) | jcomqa (exact_match) | jemhopqa (exact_match) | jsquad (exact_match) | niilcqa (exact_match) |
|----------------|-------------------|----------------------|------------------------|----------------------|-----------------------|
| sarashina1-7B  | 0.6970            | 0.6953               | 0.4530                 | 0.7778               | 0.4444                |
| sarashina2-7B  | 0.7330            | 0.8606               | 0.5556                 | 0.8566               | 0.5000                |
| sarashina1-13B | 0.7710            | 0.7712               | 0.4444                 | 0.7816               | 0.4877                |
| sarashina2-13B | 0.8080            | 0.8990               | 0.6496                 | 0.8856               | 0.5679                |
| sarashina1-65B | 0.8710            | 0.8409               | 0.6068                 | 0.8521               | 0.5370                |



> [!WARNING]
> 設定ファイルを使用することで評価設定を揃えても、上記とわずかに異なるスコアが得られることがあります。これは行列計算の順序が外部環境依存であり、低精度数値表現 (e.g., bfloat16) においてはその順序が出力数値の差となって現れる場合があるためです（[参考](https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535)）.
