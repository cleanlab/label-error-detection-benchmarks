# Label Error Detection Benchmarks

This repo contains code to reproduce results in the paper "Model-Agnostic Label Quality Scoring to Detect Real-World Label Errors" (TODO: insert link).

Full detail of results can be found [here](https://docs.google.com/spreadsheets/d/1EvdeGOtLW7z4C7Edg3FIg0Q-Su_AqtsmRzVv5_uuPO4/edit?usp=sharing).

## Download Datasets

|     | Dataset                                                                                                                                                                                                                                                                  | Links                                                                                                                                                                                                                                                                           |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [roman-numeral](https://github.com/JohnsonKuan/label-error-detection-benchmarks/tree/main/src/experiments/roman-numeral)                                                                                                                                                 | Dataset: [Codalab](https://worksheets.codalab.org/bundles/0x497f5d7096724783aa1eb78b85aa321f)<br />Verified Labels: [andrew-ng-dcai-comp-2021-manual-review-for-label-errors.xlsx](andrew-ng-dcai-comp-2021-manual-review-for-label-errors.xlsx)                                |
| 2   | [food-101n](https://github.com/JohnsonKuan/label-error-detection-benchmarks/tree/main/src/experiments/food-101n)                                                                                                                                                         | Dataset and Verified Labels: https://kuanghuei.github.io/Food-101N/ <br />Download file: `Food-101N_release.zip` <br /> Training dataset: `./Food-101N_release/train`<br />Verified training labels (subset of training dataset): `./Food-101N_release/meta/verified_train.tsv` |
| 3   | [cifar-10n-aggregate](https://github.com/JohnsonKuan/label-error-detection-benchmarks/tree/main/src/experiments/cifar-10n-aggregate) <br /> [cifar-10n-worst](https://github.com/JohnsonKuan/label-error-detection-benchmarks/tree/main/src/experiments/cifar-10n-worst) | https://github.com/UCSC-REAL/cifar-10-100n <br /> http://ucsc-real.soe.ucsc.edu:1995/Home.html                                                                                                                                                                                  |

## (Optional) Run cross-validation for each dataset to generate predicted probabilities

Running cross-validation is optional because we've conveniently provided pre-computed out-of-sample predicted probabilities for each dataset and model.

**Prerequisite**

- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker): allows us to properly utilize our NVIDIA GPUs inside docker environments

#### 1. Run docker-compose to build the docker image and run the container

Clone this repo and run below commands:

```bash
sudo docker-compose build
sudo docker-compose run --rm --service-port dcai
```

#### 2. Start Jupyter Lab

```bash
make jupyter-lab
```

#### 3. Run notebook for each dataset

Each dataset will have its own folder in `./src/eperiments` with a notebook to:

1. `1_Run_Cross_Val_Noisy_Labels.ipynb`: for each model, run k-fold cross-validation with noisy labels to generated out-of-sample predicted probabilities
2. `2_Save_Cross_Val_Results_To_Numpy.ipynb`: for each model, save predicted probabilities to a Numpy file

## Evaluate Label Quality Scores

All pre-computed predicted probabilities are available in the `/src/experiments` folder except for Food-101n (due to large file size.)

For Food-101n, download the pre-computed predicted probabilities (`pred_probs.npy`) [here](https://drive.google.com/file/d/1DV45bpazRIeLGV_wJD7fDuz4AzuVzhq9/view?usp=sharing).

Once we have the out-of-sample predicted probabilities for all datasets and models, we can evaluate them with a single notebook located below:

[src/experiments/Evaluate_All_Experiments.ipynb](https://github.com/JohnsonKuan/label-error-detection-benchmarks/blob/main/src/experiments/Evaluate_All_Experiments.ipynb)
