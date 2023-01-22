# Label Error Detection Benchmarks

Code to reproduce results from the paper:

[**Model-Agnostic Label Quality Scoring to Detect Real-World Label Errors**](https://people.csail.mit.edu/jonasmueller/info/LabelQuality_icml.pdf). *ICML DataPerf Workshop 2022*

This repository is only for intended for scientific purposes. 
To find label issues in your own classification data, you should instead use the official [cleanlab](https://github.com/cleanlab/cleanlab) library.


## Download Datasets

|     | Dataset                                                                                                                                                                                                                                                            | Links                                                                                                                                                                                                                                                                                                                                                    |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [roman-numeral](https://github.com/cleanlab/label-error-detection-benchmarks/tree/main/src/experiments/roman-numeral)                                                                                                                                           | Dataset: [Codalab](https://worksheets.codalab.org/bundles/0x497f5d7096724783aa1eb78b85aa321f)<br />Verified Labels: [andrew-ng-dcai-comp-2021-manual-review-for-label-errors.xlsx](https://github.com/cleanlab/label-error-detection-benchmarks/blob/main/src/experiments/roman-numeral/andrew-ng-dcai-comp-2021-manual-review-for-label-errors.xlsx) |
| 2   | [food-101n](https://github.com/cleanlab/label-error-detection-benchmarks/tree/main/src/experiments/food-101n)                                                                                                                                                   | Dataset and Verified Labels: https://kuanghuei.github.io/Food-101N/ <br /> File: `Food-101N_release.zip` <br /> Training dataset: `./Food-101N_release/train`<br />Verified training labels (subset of training dataset): `./Food-101N_release/meta/verified_train.tsv`                                                                                  |
| 3   | [cifar-10n-agg](https://github.com/cleanlab/label-error-detection-benchmarks/tree/main/src/experiments/cifar-10n-aggregate) <br /> [cifar-10n-worst](https://github.com/cleanlab/label-error-detection-benchmarks/tree/main/src/experiments/cifar-10n-worst) | https://github.com/UCSC-REAL/cifar-10-100n <br /> http://ucsc-real.soe.ucsc.edu:1995/Home.html                                                                                                                                                                                                                                                           |
| 4   | [cifar-10s](https://github.com/cleanlab/label-error-detection-benchmarks/tree/main/src/experiments/cifar-10)                                                                                                                                                    | Dataset: [Download Cifar as PNG files](https://github.com/knjcode/cifar2png)<br /> Noisy Labels: [cifar10_train_dataset_noise_amount_0.2_sparsity_0.4_20220326055753.csv](https://github.com/cleanlab/label-error-detection-benchmarks/blob/main/src/experiments/cifar-10/cifar10_train_dataset_noise_amount_0.2_sparsity_0.4_20220326055753.csv)     |

The roman-numeral dataset contain duplicate images (exact same image with different file names). We use the following script to dedupe: `src/preprocess/remove_dupes.py`

## (Optional) Run cross-validation for each dataset to train models and generate predicted probabilities

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

#### 3. Run training notebooks for each dataset

Each dataset will have its own folder in `./src/eperiments` with a notebook to:

1. `1_Run_Cross_Val_Noisy_Labels.ipynb`: For each model, run k-fold cross-validation with noisy labels to generated out-of-sample predicted probabilities.
2. `2_Save_Cross_Val_Results_To_Numpy.ipynb`: For each model, save predicted probabilities to a Numpy file.

## Evaluate Label Quality Scores

The above step is optional because pre-computed predicted probabilities from all of our models are available for you to utilize in the `/src/experiments` folder (except for Food-101n, due to large file size). For Food-101n, download the pre-computed predicted probabilities (`pred_probs.npy`) [here](https://drive.google.com/file/d/1DV45bpazRIeLGV_wJD7fDuz4AzuVzhq9/view?usp=sharing).

Once we have the out-of-sample predicted probabilities for all datasets and models, we evaluate their performance for detecting label errors using the following notebook:

[src/experiments/Evaluate_All_Experiments.ipynb](https://github.com/cleanlab/label-error-detection-benchmarks/blob/main/src/experiments/Evaluate_All_Experiments.ipynb)

Raw tables of all performance numbers for each method+dataset can be found in [this Google sheet](https://docs.google.com/spreadsheets/d/1EvdeGOtLW7z4C7Edg3FIg0Q-Su_AqtsmRzVv5_uuPO4/edit?usp=sharing).
