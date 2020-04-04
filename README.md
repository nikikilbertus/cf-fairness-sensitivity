# The Sensitivity of Counterfactual Fairness to Unmeasured Confounding

This repo contains the code for the paper

> The Sensitivity of Counterfactual Fairness to Unmeasured Confounding

published at UAI 2019 [arxiv](https://arxiv.org/abs/1907.01040)

All code lives in the `src` folder with experiment runs fully specified by the `json` config files in `src/experiments`. Provide the path to the config file to the main `run.py`. For example, navigate to the `src` directory and run

```sh
python3 run.py --path experiments/config_lawschool.json
```

The results will be written to the path specified in the config file (`data/results/` by default).