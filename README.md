# Cluster-Guided Fairness Testing for AI Classification Models

This project presents a cluster-guided approach for detecting individual discrimination in machine learning classifiers. The method improves over traditional random testing by focusing on regions of the input space where discriminatory behaviour is more likely to occur.

---

## Overview

Fairness testing evaluates whether machine learning models treat individuals consistently across sensitive attributes such as age, race, and gender.

This project uses the concept of **Individual Discriminatory Instances (IDIs)**, where two inputs differ only in sensitive attributes but produce significantly different predictions.

Three testing strategies are implemented:
- **Random Search** (baseline)
- **Cluster-Guided Search** (proposed method)
- **Adaptive Cluster-Guided Search** (extension)

---

## Method

The proposed approach improves testing efficiency through structured exploration:

1. **Clustering**  
   K-Means++ is applied on non-sensitive features to group similar inputs.

2. **Pilot Phase**  
   A subset of the testing budget is used to estimate discrimination density in each cluster.

3. **Budget Allocation**  
   Remaining resources are allocated proportionally to clusters with higher discrimination scores.

4. **Local Exploitation**  
   Test generation is refined within high-risk clusters to improve detection.

---

## Datasets

The experiments are conducted on two benchmark datasets:

- **ADULT Dataset**  
  UCI Census dataset for income classification

- **COMPAS Dataset**  
  Recidivism prediction dataset widely used in fairness research

---

## Results

The cluster-guided method significantly improves fairness testing effectiveness:

- **ADULT:** +55.3% improvement over random search  
- **COMPAS:** +270.0% improvement over random search  

The results demonstrate that discriminatory behaviour is concentrated in specific regions of the input space, which can be efficiently explored using clustering.

---

## Project Structure
├── fairness_tool.py # Core fairness testing logic
├── train_model.py # Model training script
├── experiments.py # Experimental evaluation
├── dataset/ # Preprocessed datasets
├── DNN/ # Trained models and scalers
├── results/ # Output results (CSV, figures)
├── manual.pdf # User manual
├── replication.pdf # Reproducibility instructions
├── requirements.txt # Dependencies



---

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```
---

## Usage

### 1. Train Models
```bash
python train_model.py
```

### 2. Run Experiments
```bash
python experiments.py
```

Results will be saved in the `results/` directory.

---

## Reproducibility

All experiments in the report can be reproduced using the provided scripts, datasets, and trained models. Detailed instructions are included in `replication.pdf`.

---

## Author

**Indhu Shree Prakash**  
MSc AdvancedComputer Science  
University of Birmingham
