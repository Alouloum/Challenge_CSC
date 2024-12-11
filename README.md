# Challenge_CSC: Sub-event Detection in Twitter Streams

## Team: Chicha Learning  
**Members:**  
- Aloulou Mohamed¹  
- Jemmali Fadi¹  
- Nicolas Elie¹  

¹École Polytechnique (X22)

This project is part of the **CSC_51054_EP - Machine and Deep Learning (2024-2025)** course.

---

## Overview

The goal of this project is to classify the presence or absence of specific sub-events in tweets posted during football matches from the 2010 and 2014 FIFA World Cups. Using annotated datasets, we aim to build an accurate binary classification model capable of predicting in-game sub-events based on tweet content.



## How to Run the Project

### Step 1:
Navigate to the folder Challenge_CSC
```bash
cd Challenge_CSC
```
### Step 2: Copy the `challenge_data` folder
If you're using the terminal, run the following command (replace `/path/to/challenge_data` with the actual path to your `challenge_data` folder):
```bash
cp -r /path/to/challenge_data .
```

### Optional Step: Set Up the Environment
We recommend creating a new virtual environment. For example, using Conda:
```bash
conda create -n my_env_name python=3.11
conda activate my_env_name
```

Then, install all required libraries listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```


## Training models
To train the LSTM or BERT model, use the following command, replacing `{model_name}` with either `lstm` or `bert`:
```bash
python scripts/train_{model_name}.py
```

## Evaluate models
To generate the predictions for the validation set, use the following command, replacing `{model_name}` with either `lstm` or `bert`:
```bash
python scripts/eval_{model_name}.py
```
The predictions will be saved as a csv in the `./results directory.`

# Generating final predictions with the Meta-Model
## Step 1: Generate Bert probabilities
Run the following script to generate BERT probabilities of both train
```bash
python scripts/generate_proba_bert.py
```

