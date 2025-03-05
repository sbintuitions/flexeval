/*
Mostly Basic Python Problems (MBPP) is a dataset of crowd-sourced programming problems.

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/mbpp)
* [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732)
*/
local dataset_base_args = {
  class_path: 'HFGenerationDataset',
  init_args: {
    path: 'mbpp',
    subset: 'sanitized',
    reference_list_template: '{{ test_list }}',
  },
};

{
  class_path: 'Generation',
  init_args: {
    eval_dataset: dataset_base_args { init_args+: { split: 'test' } },
    prompt_template: {
      class_path: 'Jinja2PromptTemplate',
      init_args: {
        template: |||
          ## Question
          Write a function to find the shared elements from the given two lists.
          ## Test cases
          ```python
          assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
          assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))
          assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))
          ```
          ## Code
          ```python
          def similar_elements(test_tup1, test_tup2):
            res = tuple(set(test_tup1) & set(test_tup2))
            return (res)
          ```

          ## Question
          Write a python function to identify non-prime numbers.
          ## Test cases
          ```python
          assert is_not_prime(2) == False
          assert is_not_prime(10) == True
          assert is_not_prime(35) == True
          assert is_not_prime(37) == False
          ```
          ## Code
          ```python
          import math
          def is_not_prime(n):
              result = False
              for i in range(2,int(math.sqrt(n)) + 1):
                  if n % i == 0:
                      result = True
              return result
          ```

          ## Question
          Write a function to find the n largest integers from a given list of numbers, returned in descending order.
          ## Test cases
          ```python
          assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
          assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
          assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
          ```
          ## Code
          ```python
          import heapq as hq
          def heap_queue_largest(nums,n):
            largest_nums = hq.nlargest(n, nums)
            return largest_nums
          ```

          ## Question
          Write a python function to check whether the two numbers differ at one bit position only or not.
          ## Test cases
          ```python
          assert differ_At_One_Bit_Pos(13,9) == True
          assert differ_At_One_Bit_Pos(15,8) == False
          assert differ_At_One_Bit_Pos(2,4) == False
          assert differ_At_One_Bit_Pos(2, 3) == True
          assert differ_At_One_Bit_Pos(5, 1) == True
          assert differ_At_One_Bit_Pos(1, 5) == True
          ```
          ## Code
          ```python
          def is_Power_Of_Two (x):
              return x and (not(x & (x - 1)))
          def differ_At_One_Bit_Pos(a,b):
              return is_Power_Of_Two(a ^ b)
          ```

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
    metrics: [
      {
        class_path: 'CodeEval',
        init_args: {
          evaluate_module: 'ktakuya/safe_code_eval',
        },
      },
    ],
    gen_kwargs: { max_new_tokens: 512, stop_sequences: ['```'] },
    batch_size: 1,
  },
}
