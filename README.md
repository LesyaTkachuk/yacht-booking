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

## To train model locally

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

- to train model 

```
python3 train_reco_model.py
```