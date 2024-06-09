# Installation

`flexeval` is tested on Python 3.8+.

## Install from pip

```bash
pip install flexeval
```

Extras dependencies can be installed via pip install -e ".[NAME]".

| Name | Description                                                                 |
|------|-----------------------------------------------------------------------------|
| vllm | To load language models using [vLLM](https://github.com/vllm-project/vllm). |

## Install from source

```bash
git clone https://github.com/sbintuitions/flexeval
cd flexeval
pip install -e .
```

## Install with Docker

```bash
git clone https://github.com/sbintuitions/flexeval
cd flexeval
docker build -t flexeval .
```
