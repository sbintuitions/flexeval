/*
GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems.
The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/gsm8k]
* [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
*/

local template = |||
  Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
  A: Natalia sold 48/2 = 24 clips in May.
  Natalia sold 48+24 = 72 clips altogether in April and May.
  #### 72

  Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
  A: Weng earns 12/60 = $0.2 per minute.
  Working 50 minutes, she earned 0.2 x 50 = $10.
  #### 10

  Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
  A: In the beginning, Betty has only 100 / 2 = $50.
  Betty's grandparents gave her 15 * 2 = $30.
  This means, Betty needs 100 - 50 - 30 - 15 = $5 more.
  #### 5

  Q: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
  A: Maila read 12 x 2 = 24 pages today.
  So she was able to read a total of 12 + 24 = 36 pages since yesterday.
  There are 120 - 36 = 84 pages left to be read.
  Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.
  #### 42

  Q: {{ question }}
||| + 'A:';

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: {
      class_path: 'HFGenerationDataset',
      init_args: {
        path: 'openai/gsm8k',
        subset: 'main',
        split: 'test',
        reference_template: "{{ answer.split('#### ')[1] | trim }}",
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: template,
      },
    },
    metrics: [
      {
        class_path: 'ExactMatch',
        init_args: {
          lm_output_processor: [{ class_path: 'flexeval.core.string_processor.SimpleEvalMGSMProcessor' }],
          reference_processor: [{ class_path: 'flexeval.core.string_processor.RemoveCommaProcessor' }],
        },
      },
      { class_path: 'flexeval.core.metric.MathVerify' },
    ],
    gen_kwargs: { max_new_tokens: 512, stop_sequences: ['Q:'] },
    batch_size: 1,
  },
}
