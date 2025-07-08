# IFEval Constraints

This directory contains the constraints implemented in [the original IFEval repository](https://github.com/google-research/google-research/tree/master/instruction_following_eval).
Intended to use with [the original English dataset](https://huggingface.co/datasets/google/IFEval).

Here is the constraints and its args used in the original IFEval dataset.

| Instruction Name                         | Args                                             |
|------------------------------------------|--------------------------------------------------|
| change_case:capital_word_frequency       | {'capital_relation', 'capital_frequency'}       |
| change_case:english_capital              | {}                                              |
| change_case:english_lowercase            | {}                                              |
| combination:repeat_prompt                | {'prompt_to_repeat'}                            |
| combination:two_responses                | {}                                              |
| detectable_content:number_placeholders   | {'num_placeholders'}                            |
| detectable_content:postscript            | {'postscript_marker'}                           |
| detectable_format:constrained_response   | {}                                              |
| detectable_format:json_format            | {}                                              |
| detectable_format:multiple_sections      | {'section_spliter', 'num_sections'}             |
| detectable_format:number_bullet_lists    | {'num_bullets'}                                 |
| detectable_format:number_highlighted_sections | {'num_highlights'}                            |
| detectable_format:title                  | {}                                              |
| keywords:existence                       | {'keywords'}                                    |
| keywords:forbidden_words                 | {'forbidden_words'}                             |
| keywords:frequency                       | {'relation', 'frequency', 'keyword'}           |
| keywords:letter_frequency                | {'letter', 'let_frequency', 'let_relation'}     |
| language:response_language               | {'language'}                                    |
| length_constraints:nth_paragraph_first_word | {'num_paragraphs', 'nth_paragraph', 'first_word'} |
| length_constraints:number_paragraphs     | {'num_paragraphs'}                              |
| length_constraints:number_sentences      | {'relation', 'num_sentences'}                   |
| length_constraints:number_words          | {'num_words', 'relation'}                       |
| punctuation:no_comma                     | {}                                              |
| startend:end_checker                     | {'end_phrase'}                                  |
| startend:quotation                       | {}                                              |

Implementation Notes:

- We change the name of the class into a camel case version of the constraint name in the dataset.
  - This is different from the original implementation. E.g., `CapitalWordFrequencyChecker` -> `CapitalWordFrequency`.
