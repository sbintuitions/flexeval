# Design Principle

`flexeval` is designed according to the following principles:

* **Flexibility**: `flexeval` should be flexible in terms of the evaluation setup and the language model to be evaluated.
* **Modularity**: The core components of `flexeval` should be easily extensible and replaceable.
* **Clarity**: The results of evaluation should be clear and easy to understand its configuration.
* **Reproducibility**: `flexeval` should be reproducible, with the ability to save and load configurations and results.

To achieve flexibility and modularity, the core logic is implemented with abstract interfaces, and the concrete implementations are provided when running each CLI command.

Thanks to [jsonargparse](https://github.com/omni-us/jsonargparse), we can transparently specify the configuration of every component either via CLI arguments or jsonnet config files.
Thus, when you want to use your own module, all you have to do is implement a concrete class inheriting the right interface and specify it in the configuration, without modifying the existing code.

To achieve clarity and reproducibility, `flexeval` saves the configuration and the evaluation results in a directory specified by `--save_dir`.
The resulting `config.json` file contains everything needed to replicate the evaluation, configuration of all modules, the version of `flexeval` and the installed packages.

It is often a case that a small preprocessing in the data affects the evaluation results significantly.
We would like to the config file tells us what preprocessing is done without we need to dig into the code.
Thus we recommend loading datasets using a generic class such as `HFGenerationDataset` or `JsonlGenerationDataset` and specifying a preprocessing using their parameters or [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) templates in the configuration file.
