/*
40,000 lines of Shakespeare from a variety of Shakespeare's plays.
Featured in Andrej Karpathy's blog post ['The Unreasonable Effectiveness of Recurrent Neural Networks'](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

References:

* [Hugging Face Dataset](https://huggingface.co/datasets/karpathy/tiny_shakespeare)
*/

{
  class_path: 'Perplexity',
  init_args: {
    eval_dataset: {
      class_path: 'HFTextDataset',
      init_args: {
        path: 'karpathy/tiny_shakespeare',
        split: 'test',
        text_template: '{{ text }}',
      },
    },
  },
}
