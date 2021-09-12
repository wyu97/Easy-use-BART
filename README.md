## Easy-to-use BART baseline for PermGen

## Introduction of BART
BART is proposed by Facebook in Oct. 2019. Full paper is [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) on arXiv. Authors: Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer

> We present BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. It uses a standard Tranformer-based neural machine translation architecture which, despite its simplicity, can be seen as generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder), and many other more recent pretraining schemes.

## Model Usage

### Step 1: Set up the enviornment

1. python >= 3.6.0
```
conda create -n bart python=3.6
conda activate bart
```
2. install necessary packages
```
pip install transformers==3.3.1
pip install torch==1.7.0
pip install -r requirements.txt
```

### Step 2: Download the datasets

Download the datasets from following links and put them into the `dataset` folder.

\[[ROCStory](https://drive.google.com/drive/folders/1hQ4OMdJZCe9DhzLpv5Wkg-2rufVFePpE?usp=sharing)\] \[[AGENDA](https://drive.google.com/drive/folders/1ydkQSBuHlkteGN07Ul57_Qdz64N2zTJu?usp=sharing)\] \[[DailyMail](https://drive.google.com/drive/folders/1GXColf7nfNAC5E0NCGBHgijzwR8wQqUj?usp=sharing)\]

### Step 3: Pre-process the datasets

```
python dataset/preprocessing_baseline.py
```

### Step 4: Train the model 
```
bash scripts/train_agenda.sh
bash scripts/train_dailymail.sh
bash scripts/train_rocstory.sh
```

## Step 5: Test with saved checkpoints

```
bash scripts/test_agenda.sh
bash scripts/test_dailymail.sh
bash scripts/test_rocstory.sh
```