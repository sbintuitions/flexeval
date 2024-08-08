/*
XWinograd is a multilingual collection of Winograd Schemas in six languages that can be used for evaluation of cross-lingual commonsense reasoning capabilities.
This is an Japanese subset of the dataset.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/Muennighoff/xwinograd)
* [It’s All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning](https://aclanthology.org/2021.findings-acl.310/)
*/
{
  class_path: 'MultipleChoice',
  init_args: {
    eval_dataset: {
      class_path: 'HFMultipleChoiceDataset',
      init_args: {
        path: 'Muennighoff/xwinograd',
        subset: 'jp',
        split: 'test',
        choices_templates: [
          '{{ option1 }}{{ sentence.split("_")[1] }}',
          '{{ option2 }}{{ sentence.split("_")[1] }}',
        ],
        answer_index_template: '{{ answer | int - 1 }}',
        input_templates: { context: '{{ sentence.split("_")[0] }}' },
      },
    },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: '{{ context }}',
      },
    },
  },
}
