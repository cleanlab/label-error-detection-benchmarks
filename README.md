# Label Error Detection Benchmarks

This repo contains code to reproduce results in the paper "Model-Agnostic Label Quality Scoring to Detect Real-World Label Errors" (TODO: insert link).

## Dataset Download Instructions

#### **roman-numeral**

Link to download dataset: https://worksheets.codalab.org/bundles/0x497f5d7096724783aa1eb78b85aa321f
Link to download annotations with verified labels: https://docs.google.com/spreadsheets/d/17tjqB23cUkW7XseWgL51hRBfQnbvfHB0puHIRwoH6OM/edit?usp=sharing

#### **Food-101n**

Link to download dataset: https://kuanghuei.github.io/Food-101N/

Download file: `Food-101N_release.zip`
Training dataset: `./Food-101N_release/train`
Verified training labels (subset of training dataset): `./Food-101N_release/meta/verified_train.tsv`

---

#### **Cifar-10n**

Link to download dataset: https://github.com/UCSC-REAL/cifar-10-100n

See link for details on this dataset: http://ucsc-real.soe.ucsc.edu:1995/Home.html

## Instructions to run cross-validation for each dataset to generate predicted probabilities

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
