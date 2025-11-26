# Yacht Booking

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