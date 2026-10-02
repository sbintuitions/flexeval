/*
MultiChallenge: a multi-turn conversation benchmark for LLMs (English).

Each instance is a conversation with three or more turns, where the assistant has to respond to the last user turn.
The conversations are categorized into four axes:

* Inference Memory: recall and use the information given by the user in earlier turns
* Instruction Retention: keep following the instructions given in earlier turns
* Reliable Version Editing: iteratively edit a piece of work through the conversation
* Self-Coherence: stay consistent with the assistant's own earlier responses

Each conversation has a YES/NO check question (`target_question`) for the last assistant response.
Use the `multi_challenge_eval` metric preset with `flexeval_file` to let an LLM judge the responses.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/ScaleAI/MultiChallenge)
* [Data Source](https://github.com/ekwinox117/multi-challenge)
* [MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs](https://aclanthology.org/2025.findings-acl.958/)
*/
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'OpenAIMessagesDataset',
      init_args: {
        path: 'ScaleAI/MultiChallenge',
        split: 'test',
        message_key: 'conversation',
      },
    },
    metrics: [
      { class_path: 'OutputLengthStats' },
      { class_path: 'flexeval.core.metric.repetition_n.RepetitionN' },
    ],
    batch_size: 4,
  },
}
