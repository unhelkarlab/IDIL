# IDIL

> [!WARNING]  
> This code is not yet officially released, and it is still undergoing cleanup. Once the final version is uploaded, this warning will be removed.

## Installation
We recommend you use `conda` environment with python 3.8. 
```conda create -n idil python=3.8```
Then, please install with the following command:
```pip install -e .```


## Execution
You can run IDIL with the following command:
```bash train_dnn/scripts/idil_run.sh```

## Development

### Style Guide
This project uses a custom Python style guide, which differs from [PEP 8](https://www.python.org/dev/peps/pep-0008/) in the following ways:
- Use two-space indentation instead of four-space indentation.
- 80 character line limits rather than 79.

You can format your code according to the style guide using linters (e.g., [flake8](https://pypi.org/project/flake8/)) and autoformatters (e.g., [yapf](https://github.com/google/yapf)).
