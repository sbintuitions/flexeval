/*
Zero-shot Python code generation task in Japanese.

This is a version of jhumaneval preprocessed to replace indentation spaces with tabs.
Some models (e.g., Llama) seems to have trouble with spaces in the prompt.
*/
local original_config = import './jhumaneval.jsonnet';

original_config {
  init_args+: {
    eval_dataset+: {
      init_args+: {
        reference_template: '{{ test | replace("    ", "\t") }}\n\ncheck({{ entry_point }})\n',
      },
    },
    prompt_template+: {
      init_args+: {
        template: "{{ prompt | replace('    ', '\t') }}",
      },
    },
    metrics: [
      { class_path: 'CodeEval', init_args: { code_template: '{{ prompt | replace("    ", "\t") }}{{ lm_output }}' } },
    ],
  },
}
