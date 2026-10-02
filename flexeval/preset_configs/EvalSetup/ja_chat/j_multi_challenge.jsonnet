/*
J-MultiChallenge: a Japanese adaptation of MultiChallenge, a multi-turn conversation benchmark for LLMs.

Each instance is a conversation with three or more turns, where the assistant has to respond to the last user turn.
The conversations are categorized into four axes (`AXIS`):

* INFERENCE_MEMORY: recall and use the information given by the user in earlier turns
* INSTRUCTION_RETENTION: keep following the instructions given in earlier turns
* RELIABLE_VERSION_EDITING: iteratively edit a piece of work through the conversation
* SELF_COHERENCE: stay consistent with the assistant's own earlier responses

Each conversation has a YES/NO check question (`TARGET_QUESTION`) for the last assistant response.
Use the `j_multi_challenge_eval` metric preset with `flexeval_file` to let an LLM judge the responses.

In addition to the original data, instances that require external world knowledge to be answered and judged
are flagged with `REQUIRES_EXTERNAL_KNOWLEDGE`, so that the pure multi-turn ability can be measured separately.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/sbintuitions/J-MultiChallenge)
* [Original Data Source](https://github.com/ekwinox117/multi-challenge)
* [MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs](https://aclanthology.org/2025.findings-acl.958/)
*/
{
  class_path: 'ChatResponse',
  init_args: {
    eval_dataset: {
      class_path: 'OpenAIMessagesDataset',
      init_args: {
        path: 'sbintuitions/J-MultiChallenge',
        split: 'test',
        message_key: 'CONVERSATION',
      },
    },
    metrics: [
      { class_path: 'OutputLengthStats' },
      { class_path: 'flexeval.core.metric.repetition_n.RepetitionN' },
    ],
    batch_size: 4,
  },
}
