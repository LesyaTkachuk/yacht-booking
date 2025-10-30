import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from dotenv import load_dotenv
from surprise import Dataset, Reader, SVD, SVDpp, accuracy
from surprise.model_selection import train_test_split

load_dotenv()

# --- Environment & DB connection ------------------------------------------------
DATABASE_USERNAME = os.environ.get('DATABASE_USERNAME')
DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD')
DATABASE_HOST = os.environ.get('DATABASE_HOST')
DATABASE_PORT = os.environ.get('DATABASE_PORT')
DATABASE_NAME = os.environ.get('DATABASE_NAME')

# Build Postgres DSN and create a SQLAlchemy engine
PG_DSN = f"postgresql+psycopg2://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
engine = create_engine(PG_DSN, pool_pre_ping=True)

# --- Aggregation parameters (keep in sync with your MV if you use one) ----------
HALF_LIFE_DAYS = 30          # time-decay half-life in days (fresh events weigh more)
VIEW_DEBOUNCE_SECONDS = 30   # debounce window for 'view' events from same user on same yacht
LOOKBACK_DAYS = 365 * 2      # history window for events (e.g., last 2 years)
DAY_CAP = 20.0               # per-(user,yacht,day) cap to prevent single session flooding


# -----------------------------
# --- DATA PREPARATION --------
# -----------------------------

# --- 1) Load base tables --------------------------------------------------------
# Users: only need id and role to keep 'lessee' events
df_users = pd.read_sql('SELECT id, role FROM users;', engine)

# Yachts: (owner_id used to drop self-interactions)
df_yachts = pd.read_sql('SELECT id as yacht_id, "userId" AS owner_id FROM yachts;', engine)

# Events within the lookback window (parameterized to avoid string formatting in SQL)
events_query = text("""
    SELECT id,
           "userId"   AS user_id,
           "yachtId"  AS yacht_id,
           type,
           weight,
           "createdAt" AS ts
    FROM events
    WHERE "createdAt" >= NOW() - (:days || ' days')::interval
""")
df_events = pd.read_sql(events_query, engine, params={'days': str(LOOKBACK_DAYS)})

# Optional guard: if there are no events, stop early to avoid errors downstream
if df_events.empty:
    raise ValueError("No events found within the lookback window. Adjust LOOKBACK_DAYS or check data.")

# --- 2) Keep only 'lessee' events ----------------------------------------------
# Join user role and filter to renter role (events from admins/owners are not demand signals)
df_events = df_events.merge(df_users, left_on="user_id", right_on="id", how="left", suffixes=("", "_u"))
df_events = df_events[df_events["role"] == "lessee"].drop(columns=["id_u", "role"])

# --- 3) Remove interactions with user's own yachts ------------------------------
# (Owners interacting with their listings would bias the model)
df_events = df_events.merge(df_yachts, on="yacht_id", how="left")
df_events = df_events[df_events["owner_id"].astype(str) != df_events["user_id"].astype(str)]

# --- 4) Debounce frequent 'view' events ----------------------------------------
# Sort and drop near-duplicate 'view' events per (user, yacht) inside the debounce window
df_events = df_events.sort_values(["user_id", "yacht_id", "type", "ts"])
mask = ~(
    (df_events["type"] == "view") &
    (df_events.groupby(["user_id", "yacht_id", "type"])["ts"]
        .diff()
        .dt.total_seconds()
        .between(0, VIEW_DEBOUNCE_SECONDS, inclusive="both"))
)
df_events = df_events[mask]

# Optional guard: if everything got filtered out, stop early
if df_events.empty:
    raise ValueError("All events were filtered out by role/self-ownership/debounce. Check filters.")

# --- 5) Time-decay weighting ----------------------------------------------------
# Compute exponentially decayed weight: eff_w = weight * exp(-days_ago / half_life)
now = pd.Timestamp.utcnow()
days_ago = (now - pd.to_datetime(df_events["ts"], utc=True)).dt.total_seconds() / 86400.0
decay = np.exp(-days_ago / HALF_LIFE_DAYS)
df_events["eff_w"] = df_events["weight"].astype(float) * decay

# --- 6) Daily cap to limit flood from single sessions ---------------------------
# Aggregate by day, cap the daily contribution, then sum across days
df_events["d"] = pd.to_datetime(df_events["ts"], utc=True).dt.floor("D")
daily = (
    df_events
    .groupby(["user_id", "yacht_id", "d"], as_index=False)["eff_w"]
    .sum()
)
daily["day_sum"] = daily["eff_w"].clip(upper=DAY_CAP)

scores = (
    daily
    .groupby(["user_id", "yacht_id"], as_index=False)
    .agg(score=("day_sum", "sum"), latest_ts=("d", "max"))
)

# --- 7) Per-user min-max scaling to [1..5] -------------------------------------
# Convert implicit scores into pseudo-ratings for Surprise (stable range per user)
def scale_1_5(g: pd.DataFrame) -> pd.DataFrame:
    mn, mx = g["score"].min(), g["score"].max()
    if mx - mn < 1e-9:
        # If user has uniform scores, assign a neutral-but-positive rating
        g["rating"] = 4.0
    else:
        g["rating"] = 1 + 4 * (g["score"] - mn) / (mx - mn)
    return g

def ensure_str_ids(df):
    df["user_id"] = df["user_id"].astype(str)
    df["yacht_id"] = df["yacht_id"].astype(str)
    return df

ratings = scores.groupby("user_id", group_keys=False).apply(scale_1_5)
ratings = ratings[["user_id", "yacht_id", "rating", "latest_ts"]].reset_index(drop=True)
ratings = ensure_str_ids(ratings)

print("Sample ratings:\n", ratings.head())

# --- 8) Build Surprise dataset --------------------------------------------------
# Surprise expects explicit ratings with a known scale; we mapped to [1..5]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["user_id", "yacht_id", "rating"]], reader)

# -----------------------------
# --- TRAIN / TEST MODEL --------
# -----------------------------

# Config
ART_DIR = os.getenv("ART_DIR", "./artifacts_surprise")
ALGO_NAME = os.getenv("ALGO", "SVD")            # 'SVD' or 'SVDpp'
FACTORS   = int(os.getenv("FACTORS", "64"))
EPOCHS    = int(os.getenv("EPOCHS", "40"))
LR_ALL    = float(os.getenv("LR_ALL", "0.005"))
REG_ALL   = float(os.getenv("REG_ALL", "0.02"))
CUTOFF    = os.getenv("CUTOFF_DATE", "")        # e.g. "2025-06-01" for time-based split
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))# used if no CUTOFF
K_TOP     = int(os.getenv("TOPK", "10"))        # for top-K metrics
POS_TH    = float(os.getenv("POS_THRESHOLD", "4.0"))  # rating >= POS_TH is positive in test

os.makedirs(ART_DIR, exist_ok=True)


# ==== 9) Build Surprise train/test
def build_train_test_from_ratings(ratings_df: pd.DataFrame, cutoff_date: str = "", test_size: float = 0.2):
    """Create Surprise trainset/testset either by time split or random split."""
    reader = Reader(rating_scale=(1, 5))
    if cutoff_date:
        T = pd.to_datetime(cutoff_date)
        train_df = ratings_df[ratings_df["latest_ts"] < T]
        test_df  = ratings_df[ratings_df["latest_ts"] >= T]
        if train_df.empty or test_df.empty:
            raise ValueError("Time-based split produced empty train or test; adjust CUTOFF_DATE.")
        train_data = Dataset.load_from_df(train_df[["user_id","yacht_id","rating"]], reader)
        trainset = train_data.build_full_trainset()
        testset  = list(test_df[["user_id","yacht_id","rating"]].itertuples(index=False, name=None))
        return trainset, testset, train_df, test_df
    else:
        data = Dataset.load_from_df(ratings_df[["user_id","yacht_id","rating"]], reader)
        trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
        # For convenience, reconstruct DataFrames
        train_df = pd.DataFrame(trainset.build_testset(), columns=["user_id","yacht_id","rating"])
        test_df  = pd.DataFrame(testset, columns=["user_id","yacht_id","rating"])
        return trainset, testset, train_df, test_df

trainset, testset, train_df, test_df = build_train_test_from_ratings(ratings, CUTOFF, TEST_SIZE)
train_df = ensure_str_ids(train_df)
test_df = ensure_str_ids(test_df)


# ==== 10 Train model
def make_algo(name: str):
    if name == "SVDpp":
        return SVDpp(n_factors=FACTORS, n_epochs=EPOCHS, random_state=42)
    # default SVD
    return SVD(n_factors=FACTORS, n_epochs=EPOCHS, lr_all=LR_ALL, reg_all=REG_ALL, random_state=42)

algo = make_algo(ALGO_NAME)
algo.fit(trainset)


# ==== 11 Evaluate: RMSE/MAE (pointwise)
preds = algo.test(testset)
rmse = accuracy.rmse(preds, verbose=False)
mae  = accuracy.mae(preds, verbose=False)
print(f"[METRIC] RMSE={rmse:.4f}  MAE={mae:.4f}")


# ==== 12  Evaluate: Top-K ranking metrics
def precision_recall_ndcg_at_k(algo, train_df, test_df, K=10, pos_threshold=4.0):
    # positives in test
    test_pos = test_df[test_df["rating"] >= pos_threshold]
    user_pos = test_pos.groupby("user_id")["yacht_id"].apply(set).to_dict()
    # candidates = items seen in train universe
    item_set = set(train_df["yacht_id"].unique())
    # items seen in train per user
    train_seen = train_df.groupby("user_id")["yacht_id"].apply(set).to_dict()

    precs, recs, ndcgs = [], [], []

    for u, pos_items in user_pos.items():
        candidates = list(item_set - train_seen.get(u, set()))
        if not candidates:
            continue
        est = [(iid, algo.predict(u, iid).est) for iid in candidates]
        est.sort(key=lambda x: x[1], reverse=True)
        top = [iid for iid,_ in est[:K]]

        # precision/recall
        hits = len(set(top) & pos_items)
        precs.append(hits / K)
        recs.append(hits / max(1, len(pos_items)))

        # nDCG
        # relevance is 1 if item in pos_items else 0
        gains = [1.0 if iid in pos_items else 0.0 for iid in top]
        dcg = 0.0
        for i, g in enumerate(gains, start=1):
            dcg += g / np.log2(i + 1)
        # ideal DCG: first min(K, |pos|) are 1
        ideal_hits = min(K, len(pos_items))
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {
        "precision_at_k": float(np.mean(precs)) if precs else 0.0,
        "recall_at_k": float(np.mean(recs)) if recs else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users_evaluated": int(len(precs)),
    }
    return metrics

rank_metrics = precision_recall_ndcg_at_k(algo, train_df, test_df, K=K_TOP, pos_threshold=POS_TH)
print(f"[METRIC] P@{K_TOP}={rank_metrics['precision_at_k']:.4f}  "
      f"R@{K_TOP}={rank_metrics['recall_at_k']:.4f}  "
      f"nDCG@{K_TOP}={rank_metrics['ndcg_at_k']:.4f}  "
      f"(users={rank_metrics['users_evaluated']})")


# ==== 13 Interpretation (“why recommended?”)
# -----------------------------
# For SVD/SVDpp we can use:
#  - global/user/item biases (SVD): algo.bi (item bias), algo.bu (user bias)
#  - cosine similarity to user's recently liked items (post-hoc explanation)
#
# We'll build a tiny helper that, for a given (user, candidate),
# shows top-3 "most similar liked items" by cosine in latent space.

def latent_vectors(algo):
    # item latent matrix (inner item id -> vector)
    Qi = algo.qi
    # mapping inner <-> raw ids
    inner_to_raw_iid = {i: trainset.to_raw_iid(i) for i in range(trainset.n_items)}
    raw_to_inner_iid = {raw: inner for inner, raw in inner_to_raw_iid.items()}
    return Qi, raw_to_inner_iid, inner_to_raw_iid

Qi, raw2inner, inner2raw = latent_vectors(algo)

def cosine(a, b, eps=1e-12):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def explain_reco(user_id: str, candidate_iid: str, top_m=3):
    """Return explanation: similar previously liked items for this user."""
    # 1) map user to inner id
    try:
        u_in = trainset.to_inner_uid(user_id)
    except ValueError:
        return {"note": "user not in train"}

    # 2) items the user interacted with in TRAIN (inner ids)
    user_items_inner = [i for (i, _rating) in trainset.ur[u_in]]

    # 3) convert inner -> raw item ids
    user_items_raw = [trainset.to_raw_iid(i) for i in user_items_inner]

    # 4) pick user's positives (rating >= POS_TH) from TRAIN df
    liked_raw = set(
        train_df.loc[
            (train_df.user_id.astype(str) == str(user_id)) &
            (train_df.rating >= POS_TH),
            "yacht_id"
        ].astype(str)
    )

    liked = [iid for iid in user_items_raw if iid in liked_raw]
    if not liked:
        return {"note": "no liked items in train"}

    # 5) candidate latent vector
    if candidate_iid not in raw2inner:
        return {"note": "candidate not in train"}
    cand_vec = Qi[raw2inner[candidate_iid]]

    # 6) compute cosine similarity with liked items
    sims = []
    for iid in liked:
        if iid not in raw2inner:
            continue
        sims.append((iid, cosine(cand_vec, Qi[raw2inner[iid]])))

    sims.sort(key=lambda x: x[1], reverse=True)
    return {"similar_liked": sims[:top_m]}

# ==== 14  Save artifacts & sample outputs
model_path = os.path.join(ART_DIR, f"surprise_{ALGO_NAME.lower()}.pkl")
meta_path  = os.path.join(ART_DIR, f"surprise_meta_{ALGO_NAME.lower()}.json")
metrics_path = os.path.join(ART_DIR, f"metrics_{ALGO_NAME.lower()}.json")
samples_path = os.path.join(ART_DIR, f"sample_recommendations_{ALGO_NAME.lower()}.csv")

# ==== 15 Save model
with open(model_path, "wb") as f:
    pickle.dump(algo, f)

# Save meta (train universe for inference & filtering)
meta = {
    "algo": ALGO_NAME,
    "trained_utc": datetime.now(timezone.utc).isoformat(),
    "params": {"factors": FACTORS, "epochs": EPOCHS, "lr_all": LR_ALL, "reg_all": REG_ALL},
    "num_users": int(train_df.user_id.nunique()),
    "num_items": int(train_df.yacht_id.nunique()),
    "users": sorted(str(uid) for uid in train_df.user_id.unique().tolist()),
    "items": sorted(str(iid) for iid in train_df.yacht_id.unique().tolist()),
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

# Save metrics
all_metrics = {
    "rmse": rmse,
    "mae": mae,
    "precision_at_k": rank_metrics["precision_at_k"],
    "recall_at_k": rank_metrics["recall_at_k"],
    "ndcg_at_k": rank_metrics["ndcg_at_k"],
    "users_evaluated": rank_metrics["users_evaluated"],
    "k": K_TOP,
    "pos_threshold": POS_TH,
    "cutoff_date": CUTOFF or None,
    "test_size": TEST_SIZE if not CUTOFF else None,
}
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"[OK] Saved model → {model_path}")
print(f"[OK] Saved meta  → {meta_path}")
print(f"[OK] Saved metrics → {metrics_path}")

# ==== 16 Generate sample top-N for first N users (for QA / quick sanity check)
def recommend_for_user(user_id: str, limit=10):
    # candidate universe = items seen in train
    item_set = set(train_df["yacht_id"].unique())
    # filter out items seen by user in train to avoid echo
    try:
        seen = set(train_df[train_df.user_id == user_id]["yacht_id"].unique())
    except KeyError:
        seen = set()
    candidates = list(item_set - seen)
    if not candidates:
        return []
    est = [(iid, algo.predict(user_id, iid).est) for iid in candidates]
    est.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid,_ in est[:limit]]

sample_users = train_df["user_id"].drop_duplicates().head(20).tolist()
rows = []
for u in sample_users:
    top10 = recommend_for_user(u, limit=10)
    # add a tiny explanation for the first item
    exp = explain_reco(u, top10[0]) if top10 else {}
    rows.append({
        "user_id": u,
        "top10": ",".join(top10),
        "explanation_first": json.dumps(exp)
    })
pd.DataFrame(rows).to_csv(samples_path, index=False)
print(f"[OK] Saved sample recommendations → {samples_path}")

# -----------------------------
# --- STORE RECOMMENDATIONS IN DB ---------
# -----------------------------
# ==== Save per-user recommendations into users.recommendations (uuid[]) =========

CANDIDATES_N = 30
BUSINESS_LIMIT = 12
BUSINESS_STRATEGY = "high_price_then_sort_by_rating"

ITEM_UNIVERSE = set(train_df["yacht_id"].unique())
SEEN_BY_USER = train_df.groupby("user_id")["yacht_id"].apply(set).to_dict()

# ==== 17 Ensure column exists (uuid[])
with engine.begin() as conn:
    conn.execute(text("""
    ALTER TABLE users
    ADD COLUMN IF NOT EXISTS "recommendations" uuid[];
    """))


HYDRATE_SQL = text("""
  SELECT id, "summerLowSeasonPrice" AS price, rating
  FROM yachts
  WHERE id IN (SELECT UNNEST(CAST(:ids AS uuid[])))
""")

# We pass a Python list of (string) UUIDs, cast to uuid[] safely via UNNEST → array_agg
UPDATE_USER_RECS = text("""
  UPDATE users
  SET "recommendations" = :yacht_ids
  WHERE id = :user_id
""").bindparams(
    bindparam("yacht_ids", type_=ARRAY(UUID)),
    bindparam("user_id",   type_=UUID),
)

# ==== 18 create helpers
def recommend_ids_for_user(user_id: str, limit=CANDIDATES_N):
    """Top-N candidate yacht IDs for a user (excluding items seen in train)."""
    candidates = list(ITEM_UNIVERSE - SEEN_BY_USER.get(user_id, set()))
    if not candidates:
        return []
    est = [(iid, algo.predict(user_id, iid).est) for iid in candidates]
    est.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in est[:limit]]


def hydrate_yachts(ids):
    """Fetch price/rating for given yacht IDs preserving the input order."""
    if not ids:
        return pd.DataFrame(columns=["id","price","rating"])
    df = pd.read_sql(HYDRATE_SQL, engine, params={"ids": ids})
    order = {yid: i for i, yid in enumerate(ids)}
    df["__ord"] = df["id"].map(order)
    return df.sort_values("__ord").drop(columns="__ord")

def business_select_and_sort(df_y):
    """Pick 12 highest-price, then sort for return by rating desc (tie: price desc)."""
    if df_y.empty:
        return []
    top_by_price = df_y.sort_values("price", ascending=False).head(BUSINESS_LIMIT)
    top_by_price["rating"] = pd.to_numeric(top_by_price["rating"], errors="coerce")
    top_by_price["__r"] = top_by_price["rating"].fillna(-1)
    final_sorted = top_by_price.sort_values(["__r","price"], ascending=[False, False]).drop(columns="__r")
    return final_sorted["id"].tolist()


# ==== 18 Generate & save for all users from train (you can filter to "active" users if needed)
all_users = sorted(train_df["user_id"].unique().tolist())
print(f"[INFO] Generating & saving recommendations into users.recommendations for {len(all_users)} users...")

processed = 0
for user_id in all_users:
    cand_ids = recommend_ids_for_user(user_id, limit=CANDIDATES_N)
    if not cand_ids:
        continue
    ydf = hydrate_yachts(cand_ids)
    final_ids = business_select_and_sort(ydf)  # list of yacht UUIDs (as strings)

    if not final_ids:
        # Optionally set NULL when nothing to recommend:
        with engine.begin() as conn:
            conn.execute(text('UPDATE users SET "recommendations" = NULL WHERE id = :user_id::uuid'),
                      {"user_id": user_id})
        continue

    # Write uuid[] array into users.recommendations
    with engine.begin() as conn:
        conn.execute(UPDATE_USER_RECS, {"user_id": user_id, "yacht_ids": final_ids})

    processed += 1
    if processed % 200 == 0:
        print(f"[INFO] processed {processed} users...")

print(f"[OK] Saved recommendations for {processed} users into users.recommendations.")