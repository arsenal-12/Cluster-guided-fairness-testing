# Cluster-guided-fairness-testing
Cluster-Guided Fairness Testing for AI Models 
# Cluster-Guided Fairness Testing for AI Models

This project presents a cluster-guided approach for detecting individual discrimination in machine learning models. The method improves over traditional random testing by focusing on regions of the input space where bias is more likely to occur.

## Overview

The system evaluates fairness using Individual Discriminatory Instances (IDIs), where two inputs differ only in sensitive attributes but produce significantly different predictions.

Three methods are implemented:
- Random Search (baseline)
- Cluster-Guided Search
- Adaptive Cluster-Guided Search

## Datasets

- ADULT (UCI Census dataset)
- COMPAS (recidivism prediction dataset)

## Method

The approach consists of:
1. Clustering input space using K-Means++
2. Pilot phase to estimate discrimination density
3. Budget allocation based on cluster scores
4. Local exploitation to refine search

## Results

- ADULT: +55.3% improvement over random search
- COMPAS: +270.0% improvement over random search

## Project Structure
# Cluster-Guided Fairness Testing for AI Models

This project presents a cluster-guided approach for detecting individual discrimination in machine learning models. The method improves over traditional random testing by focusing on regions of the input space where bias is more likely to occur.

## Overview

The system evaluates fairness using Individual Discriminatory Instances (IDIs), where two inputs differ only in sensitive attributes but produce significantly different predictions.

Three methods are implemented:
- Random Search (baseline)
- Cluster-Guided Search
- Adaptive Cluster-Guided Search

## Datasets

- ADULT (UCI Census dataset)
- COMPAS (recidivism prediction dataset)

## Method

The approach consists of:
1. Clustering input space using K-Means++
2. Pilot phase to estimate discrimination density
3. Budget allocation based on cluster scores
4. Local exploitation to refine search

## Results

- ADULT: +55.3% improvement over random search
- COMPAS: +270.0% improvement over random search

## Project Structure
├── fairness_tool.py
├── train_model.py
├── experiments.py
├── dataset/
├── DNN/
├── results/
├── manual.pdf
├── replication.pdf
├── requirements.pdf
## How to Run

1. Install dependencies:

pip install -r requirements.txt


2. Train models:

python train_model.py


3. Run experiments:

python experiments.py


## Reproducibility

All experiments can be reproduced using the provided scripts and datasets. Results will be saved in the `results/` folder.

## Author

Indhu Shree Prakash  
MSc Computer Science  
University of Birmingham
