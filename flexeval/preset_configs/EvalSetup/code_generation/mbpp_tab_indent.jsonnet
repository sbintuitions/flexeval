/*
Mostly Basic Python Problems (MBPP) is a dataset of crowd-sourced programming problems.

This is a version of openai_humaneval preprocessed to replace indentation spaces with tabs.
Some models (e.g., Llama) seems to have trouble with spaces in the prompt.
*/
local original_config = import './mbpp.jsonnet';

original_config {
  init_args+: {
    prompt_template+: {
      init_args+: {
        template: |||
          {% for item in few_shot_data %}
          ## Question
          {{ item.prompt }}
          ## Test cases
          ```python
          {{ item.test_list | join('\n') }}
          ```
          ## Code
          ```python
          {{ item.code | replace('    ', '\t') }}
          ```
          {% endfor %}
          ## Question
          {{ prompt }}
          ## Test cases
          ```python
          {{ test_list | join('\n') }}
          ```
          ## Code
          ```python
        |||,
      },
    },
  },
}
