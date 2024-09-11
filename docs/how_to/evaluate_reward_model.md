# Evaluate Reward model

```bash
export OPENAI_API_KEY="YOUR_API_KEY"

flexeval_reward \
  --language_model OpenAIChatAPI \
  --language_model.model "gpt-3.5-turbo" \
  --save_dir "results/reward_model_gpt3.5-turbo"
```