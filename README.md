# Yacht Booking

## Project Overview

This repository contains the source code for a joint diploma engineering project focused on developing a hybrid recommender system for an online yacht charter platform. The project was created collaboratively by Oleksandra Tkachuk and Yevhen Nesvit as part of their Master of Science in Computer Science program.

The system combines content-based, collaborative filtering, and rule-based SQL aggregation approaches to deliver personalized yacht recommendations, effectively address the cold-start problem for both new users and new yachts, and increase user engagement and platform conversion rates.

### The solution includes:

 - a content-based module for detecting similar yachts using vectorized features and k-Nearest Neighbors;

 - a personalized recommendation module based on weighted explicit interactions and KNNBaseline;

 - a cold-start engine leveraging user context, business logic, and aggregated platform data;

 - a modular architecture spanning React (frontend), Node.js (backend), PostgreSQL (database), and Python-based ML core.

This repository demonstrates a full end-to-end pipeline: data collection, preprocessing, synthetic user interaction generation, model training and evaluation, prediction serving, and system integration.

## To run api locally:

- Navigate into `api` folder and install all dependencies:

```
npm install
```

- Check `.env` file, it should contain all variables from `.env.example` file

- Run from `api` folder:

```
npm run dev
```

## To run web app locally:

- Navigate into `client` folder and install all dependencies:

```
npm install
```

- Check `.env` file, it should contain all variables from `.env.example` file

- Run from `client` folder:

```
npm run dev
```

## To train model locally move to `models` folder

- create virtual environment

```
python3 -m venv .venv
```

- activate virtual environment

```
source .venv/bin/activate
```

- install dependencies

```
pip install -r ./requirements.txt
```

- to train model (run from project root)

```
python -m models.recommendation_model
```

- to test different models and find the best parameters with the help of normal and advanced grid search you can use model/train_reco_model_from_csv.ipynb file, that uses .csv files uploaded from project database