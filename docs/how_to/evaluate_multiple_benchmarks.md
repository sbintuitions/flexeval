# Evaluate Multiple Benchmarks

With `--eval_setups`, you can evaluate a model on multiple benchmarks (we call each of them a *run*) in one command.

```bash
flexeval_lm \
  --language_model HuggingFaceLM \
  --language_model.model "sbintuitions/sarashina2.2-0.5b-instruct-v0.1" \
  --eval_setups.aio aio_chat \
  --eval_setups.mgsm mgsm_ja_chat \
  --save_dir "results/sarashina"
```

The key after `--eval_setups.` (`aio` and `mgsm` above) is the name of the run, and the results are saved in `results/sarashina/<name>/`.
You can override the parameters of a run by adding the run name, e.g., `--eval_setups.aio.max_instances 100`.

If you add `--num_repeats 3`, each run is repeated three times and the results are saved in `results/sarashina/<name>/run0`, `run1`, and `run2`.

## Resume Failed Runs

If some runs fail, re-run the same command with `--retry true`.
Runs whose `metrics.json` already exists are skipped, and only the rest are executed.

## Distribute Runs across Multiple Processes

If you add `--lock true` and launch the same command multiple times with the same `--save_dir`, the processes share the runs.
Each process takes a run that is neither finished nor taken by another process, so no run is executed twice.

```bash
# Run this command in several terminals, or on several machines sharing `--save_dir`
flexeval_lm \
  --language_model OpenAIChatAPI \
  --language_model.model "gpt-4o-mini" \
  --eval_setups.aio aio_chat \
  --eval_setups.mgsm mgsm_ja_chat \
  --save_dir "results/gpt-4o-mini" \
  --lock true
```

`--lock true` also skips finished runs like `--retry true`, so you do not need to specify both.

While a run is executing, a `.lock` file is placed in its output directory and updated periodically.
If a process dies, its lock is taken over by others after `--lock_timeout` seconds (default: 300).
When a process fails, the run it had taken is not picked up by the other running processes, so re-run the same command to execute the remaining runs.
