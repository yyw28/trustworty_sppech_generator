# Trustworthy speech study

This repository contains a variety of models for recognizing and synthesizing trustworthy speech.

## Trustworthy speech recognition

This task fine-tuning a HuBERT model to determine trustworthiness of speech, either through binary classification (trustworthy/not-trustworthy) or regression (predicting a trustworthiness score). It is currently configred to work with two datasets: the [TIS corpus](https://osf.io/45d8j/) and/or a synthetic dataset of speech samples rated for trustworthiness (contained in this project).

To fine-tune the HuBERT model, run the following command:

```
tspeech-hubert fit --config <path to a HuBERT config file>
```

If you are training in a multi-GPU environment, you must give `--trainer.strategy ddp_find_unused_parameters_true` as an additional argument.

Sample config files are provided in the `config/` directory.