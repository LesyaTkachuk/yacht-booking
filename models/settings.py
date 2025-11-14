import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment & DB connection ------------------------------------------------
DATABASE_USERNAME = os.environ.get('DATABASE_USERNAME')
DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD')
DATABASE_HOST = os.environ.get('DATABASE_HOST')
DATABASE_PORT = os.environ.get('DATABASE_PORT')
DATABASE_NAME = os.environ.get('DATABASE_NAME')

# Build Postgres DSN 
PG_DSN = f"postgresql+psycopg2://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# --- Aggregation parameters 
HALF_LIFE_DAYS = 90          # time-decay half-life in days (fresh events weigh more)
VIEW_DEBOUNCE_SECONDS = 30   # debounce window for 'view' events from same user on same yacht
LOOKBACK_DAYS = 365 * 2      # history window for events (e.g., last 2 years)
DAY_CAP = 20.0               # per-(user,yacht,day) cap to prevent single session flooding

MIN_USER_INTERACTIONS=2
MIN_ITEM_INTERACTIONS=1

# Model Config
ART_DIR = os.getenv("ART_DIR", "./artifacts_surprise")
ALGO_NAME = os.getenv("ALGO", "SVD")            # 'SVD' or 'SVDpp'
FACTORS   = int(os.getenv("FACTORS", "64"))
EPOCHS    = int(os.getenv("EPOCHS", "40"))
LR_ALL    = float(os.getenv("LR_ALL", "0.005"))
REG_ALL   = float(os.getenv("REG_ALL", "0.02"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))# used if no CUTOFF
K_TOP     = int(os.getenv("TOPK", "15"))        # for top-K metrics
POS_THRESHOLD    = float(os.getenv("POS_THRESHOLD", "3.5"))  # rating >= POS_TH is positive in test

# Recommendation Config
CANDIDATES_N = 30
BUSINESS_LIMIT = 12
BUSINESS_STRATEGY = "high_price_then_sort_by_rating"