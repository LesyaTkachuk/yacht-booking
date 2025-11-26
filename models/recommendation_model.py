from __future__ import annotations
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import time


from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from surprise import Dataset, Reader, SVD, SVDpp, NMF, KNNBaseline, BaselineOnly, CoClustering, accuracy
from surprise.model_selection import train_test_split

from .settings import PG_DSN, HALF_LIFE_DAYS, VIEW_DEBOUNCE_SECONDS, LOOKBACK_DAYS, DAY_CAP, ALGO_NAME, FACTORS, EPOCHS, LR_ALL, REG_ALL, TEST_SIZE, K_TOP, POS_THRESHOLD, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS,ART_DIR,CANDIDATES_N, COUNTRY_CANDIDATES_N, FALLBACK_MIN_N, FALLBACK_GLOBAL_N

logger = logging.getLogger(__name__)


class RecommendationModel:
    def __init__(self) -> None:
        self.algo: Optional[object] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.engine: Optional[Engine] = None
        self.df_users: Optional[pd.DataFrame] = None
        self.df_yachts: Optional[pd.DataFrame] = None
        self.df_events: Optional[pd.DataFrame] = None
        self.ratings: Optional[pd.DataFrame] = None

    # create a SQLAlchemy engine
    def create_engine(self) -> Engine:
        """Create (or reuse) a SQLAlchemy engine with safe defaults."""
        if self.engine is not None:
            return self.engine

        self.engine = create_engine(PG_DSN, pool_pre_ping=True, future=True)
        logger.debug("SQLAlchemy engine created.")
        return self.engine

    def dispose_engine(self) -> None:
        """Dispose the engine to free DB connections."""
        if self.engine is not None:
            self.engine.dispose()
            logger.debug("SQLAlchemy engine disposed.")
            self.engine = None

    # load data about yachts, users, events from database
    def load_data(self) -> None:
        """Load users, yachts, and recent events into dataframes."""
        engine = self.create_engine()

        try:
            # --- 1) Load base tables --------------------------------------------------------
            # Users: only need id and role to keep 'lessee' events
            self.df_users = pd.read_sql_query(
                sql=text("SELECT id, role FROM users;"),
                con=engine,
            )
            # Use pandas' dedicated string dtype 
            self.df_users = self.df_users.astype({"id": "string", "role": "string"})

            # Yachts: (owner_id used to drop self-interactions)
            self.df_yachts = pd.read_sql_query(
                sql=text('SELECT id AS yacht_id, "userId" AS owner_id FROM yachts;'),
                con=engine,
            )
            self.df_yachts = self.df_yachts.astype({"yacht_id": "string", "owner_id": "string"})

            # --- 2) Load events in the lookback window -------------------------------------
            events_query = text(
                """
                SELECT
                    id,
                    "userId"   AS user_id,
                    "yachtId"  AS yacht_id,
                    type,
                    weight,
                    "createdAt" AT TIME ZONE 'UTC' AS ts
                FROM events
                WHERE "createdAt" >= (NOW() AT TIME ZONE 'UTC') - (:days || ' days')::interval
                """
            )

            self.df_events = pd.read_sql_query(
                sql=events_query,
                con=engine,
                params={"days": str(LOOKBACK_DAYS)},
            )

            # Ensure proper dtypes
            if not self.df_events.empty:
                self.df_events["ts"] = pd.to_datetime(self.df_events["ts"], utc=True)
                self.df_events = self.df_events.astype(
                    {"user_id": "string", "yacht_id": "string", "type": "string"}
                )

            # Optional guard: if there are no events, stop early to avoid errors downstream
            if self.df_events.empty:
                logger.warning(
                    "No events found within the lookback window (LOOKBACK_DAYS=%s).",
                    LOOKBACK_DAYS,
                )
                raise ValueError(
                    "No events found within the lookback window. "
                    "Adjust LOOKBACK_DAYS or check data."
                )

            logger.info(
                "Loaded %d users, %d yachts, %d events (last %s days).",
                len(self.df_users),
                len(self.df_yachts),
                len(self.df_events),
                LOOKBACK_DAYS,
            )

        except Exception as exc:
            logger.exception("Failed to load data: %s", exc)
            raise RuntimeError("Failed to load data from database.") from exc

    # ---------- Utility: Data preparation ----------
    # Convert implicit scores into pseudo-ratings for Surprise (per-user min-max scaling to [1..5])
    def scale_1_5(self, g: pd.DataFrame) -> pd.DataFrame:
        """
        Scale 'score' column to [1, 5] per user. If a user has a constant score,
        assign a stable default (4.0) to avoid NaNs and zero-variance issues.
        """
        # Use to_numpy for speed and avoid chained assignment pitfalls
        s = g["score"].to_numpy(dtype=float, copy=False)
        mn, mx = float(np.min(s)), float(np.max(s))
        if not np.isfinite(mn) or not np.isfinite(mx):
            # Defensive: if scores are invalid, fall back to neutral rating
            g = g.copy()
            g["rating"] = 4.0
            return g

        g = g.copy()
        if mx - mn < 1e-9:
            g["rating"] = 4.0
        else:
            g["rating"] = 1.0 + 4.0 * (g["score"] - mn) / (mx - mn)
        return g
    
    # Data preparation
    def prepare_data(self) -> None:
        """
        Prepare interactions for model consumption:
        1) keep only 'lessee' events;
        2) remove owner self-interactions;
        3) debounce frequent 'view' events;
        4) enforce lookback window in UTC;
        5) time-decay weighting;
        6) daily cap aggregation;
        7) per-user scaling to [1..5].
        Produces self.ratings with columns [user_id, yacht_id, rating, latest_ts].
        """
        if self.df_events is None or self.df_users is None or self.df_yachts is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        df = self.df_events.copy()

        # --- 1) Keep only 'lessee' events ----------------------------------------------
        df = df.merge(
            self.df_users[["id", "role"]],
            left_on="user_id", right_on="id",
            how="left", suffixes=("", "_u")
        )
        df = df[df["role"] == "lessee"].drop(columns=["id_u", "role"])

        # --- 2) Remove interactions with user's own yachts ------------------------------
        df = df.merge(self.df_yachts[["yacht_id", "owner_id"]], on="yacht_id", how="left")
        # Ensure consistent string comparison
        df["owner_id"] = df["owner_id"].astype("string")
        df["user_id"] = df["user_id"].astype("string")
        df = df[df["owner_id"].ne(df["user_id"])].drop(columns=["owner_id"])

        # --- 3) Debounce frequent 'view' events ----------------------------------------
        # Sort once and compute diffs per (user, yacht, type)
        df = df.sort_values(["user_id", "yacht_id", "type", "ts"])
        if VIEW_DEBOUNCE_SECONDS and VIEW_DEBOUNCE_SECONDS > 0:
            delta = (
                df.groupby(["user_id", "yacht_id", "type"])["ts"]
                  .diff()
                  .dt.total_seconds()
            )
            # Keep all non-views; for 'view' keep only those outside debounce window
            keep_mask = (df["type"] != "view") | (~delta.between(0, VIEW_DEBOUNCE_SECONDS))
            df = df[keep_mask]

        if df.empty:
            raise ValueError("All events were filtered out by role/self-ownership/debounce. Check filters.")

        # --- 4) Enforce lookback window (UTC-stable) -----------------------------------
        # Even if load_data already applied it, keep a defensive clamp here.
        cutoff_utc = pd.Timestamp.utcnow() - pd.Timedelta(days=int(LOOKBACK_DAYS))
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df[df["ts"] >= cutoff_utc]

        if df.empty:
            raise ValueError("No events remain after enforcing lookback window.")

        # --- 5) Time-decay weighting ----------------------------------------------------
        # eff_w = weight * exp(-days_ago / HALF_LIFE_DAYS)
        if HALF_LIFE_DAYS is None or HALF_LIFE_DAYS <= 0:
            logger.warning("HALF_LIFE_DAYS <= 0; disabling time decay.")
            decay = 1.0
            eff = df["weight"].astype(float)
        else:
            now_utc = pd.Timestamp.utcnow()
            days_ago = (now_utc - df["ts"]).dt.total_seconds().to_numpy() / 86400.0
            decay = np.exp(-days_ago / float(HALF_LIFE_DAYS))
            eff = df["weight"].astype(float).to_numpy() * decay

        df = df.assign(eff_w=eff)

        # --- 6) Daily cap to limit flood from single sessions ---------------------------
        df = df.assign(d=df["ts"].dt.floor("D"))
        daily = (
            df.groupby(["user_id", "yacht_id", "d"], as_index=False)["eff_w"]
              .sum()
              .rename(columns={"eff_w": "day_sum"})
        )

        cap = float(DAY_CAP) if DAY_CAP is not None else np.inf
        if cap <= 0:
            logger.warning("DAY_CAP <= 0; disabling daily cap.")
            cap = np.inf
        daily["day_sum"] = daily["day_sum"].clip(upper=cap)

        scores = (
            daily.groupby(["user_id", "yacht_id"], as_index=False)
                 .agg(score=("day_sum", "sum"), latest_ts=("d", "max"))
        )

        if scores.empty:
            raise ValueError("No (user, yacht) scores after daily aggregation.")

        # --- 7) Per-user scaling to [1..5] ----------------------------------------------
        ratings = scores.groupby("user_id", group_keys=False).apply(self.scale_1_5)
        ratings = ratings[["user_id", "yacht_id", "rating", "latest_ts"]].reset_index(drop=True)
        ratings["user_id"] = ratings["user_id"].astype("string")
        ratings["yacht_id"] = ratings["yacht_id"].astype("string")

        self.ratings = ratings

        logger.info(
            "Prepared ratings: users=%d, pairs=%d (median rating=%.2f).",
            ratings["user_id"].nunique(),
            len(ratings),
            float(ratings["rating"].median()) if not ratings.empty else float("nan"),
        )
        # Optional peek for dev runs
        logger.debug("Sample ratings:\n%s", ratings.head(10).to_string(index=False))

    # ---------- Utility: filter sparsity ------------------------------------
    def _filter_min_interactions(
        self,
        df: pd.DataFrame,
        min_user_inter: int = 2,
        min_item_inter: int = 2
    ) -> pd.DataFrame:
        """
        Iteratively drop users/items with fewer than required interactions.
        Helps avoid degenerate splits and cold-start in evaluation.
        """
        prev_len = -1
        cur = df.copy()
        while len(cur) != prev_len:
            prev_len = len(cur)
            if min_user_inter > 0:
                users_ok = cur["user_id"].value_counts()
                keep_users = users_ok[users_ok >= min_user_inter].index
                cur = cur[cur["user_id"].isin(keep_users)]
            if min_item_inter > 0:
                items_ok = cur["yacht_id"].value_counts()
                keep_items = items_ok[items_ok >= min_item_inter].index
                cur = cur[cur["yacht_id"].isin(keep_items)]
        return cur

    # ---------- Top-K metrics -----------------------------------------------
    def precision_recall_ndcg_at_k(self) -> dict:
        """
        Compute Precision@K, Recall@K, nDCG@K using the model's predictions
        over the candidate set = all train items unseen by the user.
        Assumes: self.algo, self.train_df, self.test_df are set.
        """
        if self.algo is None or self.train_df is None or self.test_df is None:
            raise RuntimeError("Model and splits are not ready. Call train_test_model() first.")

        # positives in test
        test_pos = self.test_df[self.test_df["rating"] >= float(POS_THRESHOLD)]
        if test_pos.empty:
            logger.warning("No positive items in test set (rating >= POS_THRESHOLD).")
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "ndcg_at_k": 0.0,
                "users_evaluated": 0,
            }

        user_pos = test_pos.groupby("user_id")["yacht_id"].apply(set).to_dict()

        # candidate universe = items in train
        item_set = set(self.train_df["yacht_id"].unique())
        # items seen in train per user
        train_seen = self.train_df.groupby("user_id")["yacht_id"].apply(set).to_dict()

        precs, recs, ndcgs = [], [], []

        for u, pos_items in user_pos.items():
            # Users with no positives will be absent here; we evaluate only those with >=1 positive.
            candidates = list(item_set - train_seen.get(u, set()))
            if not candidates:
                # If user saw all train items, skip (nothing to rank)
                continue

            # Predict scores for candidate items (raw ids are ok for Surprise)
            est = [(iid, self.algo.predict(str(u), str(iid), verbose=False).est) for iid in candidates]
            est.sort(key=lambda x: x[1], reverse=True)
            top = [iid for iid, _ in est[:int(K_TOP)]]

            # Precision/Recall
            hits = len(set(top) & pos_items)
            precs.append(hits / float(K_TOP))
            recs.append(hits / max(1.0, float(len(pos_items))))

            # nDCG@K (binary relevance)
            gains = [1.0 if iid in pos_items else 0.0 for iid in top]
            dcg = 0.0
            for i, g in enumerate(gains, start=1):
                dcg += g / np.log2(i + 1)
            ideal_hits = min(int(K_TOP), len(pos_items))
            idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
            ndcgs.append(dcg / idcg if idcg > 1e-12 else 0.0)

        metrics = {
            "precision_at_k": float(np.mean(precs)) if precs else 0.0,
            "recall_at_k": float(np.mean(recs)) if recs else 0.0,
            "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "users_evaluated": int(len(precs)),
        }
        return metrics

    # ---------- Build train/test from ratings --------------------------------
    def build_train_test_from_ratings(self) -> Tuple[object, list, pd.DataFrame, pd.DataFrame]:
        """
        Create Surprise trainset/testset by random split.
        Expects self.ratings with columns: [user_id, yacht_id, rating, latest_ts].
        Returns (trainset, testset, train_df, test_df).
        """
        if getattr(self, "ratings", None) is None or self.ratings.empty:
            raise RuntimeError("ratings are empty. Run prepare_data() first.")

        rs = self.ratings.copy()

        # Optional sparsity filtering (configurable via settings)
        rs = self._filter_min_interactions(
            rs,
            min_user_inter=int(MIN_USER_INTERACTIONS),
            min_item_inter=int(MIN_ITEM_INTERACTIONS),
        )
        
        if rs.empty:
            raise ValueError("All interactions were filtered out by min_interactions thresholds.")

        reader = Reader(rating_scale=(1, 5))

        # Random split (default)
        data = Dataset.load_from_df(rs[["user_id", "yacht_id", "rating"]], reader)
        tr_set, te_set = train_test_split(
            data,
            test_size=TEST_SIZE,
            random_state=42
        )
        # Reconstruct DataFrames for convenience
        train_df = pd.DataFrame(tr_set.build_testset(), columns=["user_id", "yacht_id", "rating"])
        test_df  = pd.DataFrame(te_set, columns=["user_id", "yacht_id", "rating"])
        logger.info("🏋️ Random split: train=%d rows, test=%d rows.", len(train_df), len(test_df))
        return tr_set, te_set, train_df, test_df

    # ---------- Algo factory --------------------------------------------------
    def make_algo(self, name: str):
        """
        Factory for Surprise algorithms with project defaults.
        """
        name = (name or "").strip().lower()
        if name == "svdpp":
            return SVDpp(n_factors=FACTORS, n_epochs=EPOCHS, random_state=42)
        if name == "nmf":
            return NMF(n_factors=FACTORS, n_epochs=EPOCHS, random_state=42)
        if name == "knnbaseline":
            sim_options = {"name": "pearson_baseline", "user_based": True}
            return KNNBaseline(sim_options=sim_options)
        if name == "baselineonly":
            return BaselineOnly()
        if name == "coclustering":
            return CoClustering()
        # default — SVD
        return SVD(n_factors=FACTORS, n_epochs=EPOCHS, lr_all=LR_ALL, reg_all=REG_ALL, random_state=42)

    # ---------- Train & evaluate ---------------------------------------------
    def train_test_model(
        self,
        algo_name: Optional[str] = None
    ) -> dict:
        """
        Train the chosen algorithm and evaluate with pointwise RMSE/MAE and Top-K metrics.
        Returns a metrics dict.
        """
        trainset, testset, train_df, test_df = self.build_train_test_from_ratings()
        self.train_df, self.test_df = train_df, test_df

        self.algo = self.make_algo(algo_name or ALGO_NAME)
        self.algo.fit(trainset)

        # Pointwise metrics
        preds = self.algo.test(testset)
        rmse = accuracy.rmse(preds, verbose=False)
        mae  = accuracy.mae(preds, verbose=False)
        logger.info("📦 Model: %s", ALGO_NAME)
        logger.info("📊 RMSE=%.4f  MAE=%.4f", rmse, mae)

        # Top-K metrics
        rank_metrics = self.precision_recall_ndcg_at_k()
        logger.info(
            "📊 [Ranking] P@%d=%.4f  R@%d=%.4f  nDCG@%d=%.4f  (users=%d)",
            K_TOP, rank_metrics["precision_at_k"],
            K_TOP, rank_metrics["recall_at_k"],
            K_TOP, rank_metrics["ndcg_at_k"],
            rank_metrics["users_evaluated"],
        )

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            **rank_metrics,
        }
    
    # ---------- Model artifacts storage ----------------------------------------------
    def store_model_artifacts(self, metrics: dict):
        """Save model (pickle), metadata (json), metrics (json)"""
        os.makedirs(ART_DIR, exist_ok=True)

        algo_tag = (ALGO_NAME or "svd").lower()
        model_path   = os.path.join(ART_DIR, f"surprise_{algo_tag}.pkl")
        meta_path    = os.path.join(ART_DIR, f"surprise_meta_{algo_tag}.json")
        metrics_path = os.path.join(ART_DIR, f"metrics_{algo_tag}.json")
        samples_path = os.path.join(ART_DIR, f"sample_recommendations_{algo_tag}.csv")

        # 1) Save model
        with open(model_path, "wb") as f:
            pickle.dump(self.algo, f)
        logger.info("Saved model → %s", model_path)

        # 2) Save meta (basic training universe)
        meta = {
            "algo": ALGO_NAME,
            "trained_utc": datetime.now(timezone.utc).isoformat(),
            "params": {"factors": FACTORS, "epochs": EPOCHS, "lr_all": LR_ALL, "reg_all": REG_ALL},
            "num_users": int(self.train_df["user_id"].nunique()) if self.train_df is not None else 0,
            "num_items": int(self.train_df["yacht_id"].nunique()) if self.train_df is not None else 0,
            "users": sorted(map(str, self.train_df["user_id"].unique().tolist())) if self.train_df is not None else [],
            "items": sorted(map(str, self.train_df["yacht_id"].unique().tolist())) if self.train_df is not None else [],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info("Saved meta  → %s", meta_path)

        if (metrics):
                
            # 3) Save metrics (treat as dict, not object)
            all_metrics = {
                "rmse": float(metrics.get("rmse", 0.0)),
                "mae": float(metrics.get("mae", 0.0)),
                "precision_at_k": float(metrics.get("precision_at_k", 0.0)),
                "recall_at_k": float(metrics.get("recall_at_k", 0.0)),
                "ndcg_at_k": float(metrics.get("ndcg_at_k", 0.0)),
                "users_evaluated": int(metrics.get("users_evaluated", 0)),
                "k": int(K_TOP),
                "pos_threshold": float(POS_THRESHOLD),
                "test_size": float(TEST_SIZE),
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            logger.info("Saved metrics → %s", metrics_path)

    def hydrate_yachts(self, user_id: str) -> Dict[str, pd.DataFrame]:
        """
        For a given user_id:
        Returns two DataFrames:
           - 'by_country_and_budget' — yachts filtered by user's country AND [budgetMin; budgetMax * 1.2].
           - 'by_budget' — yachts filtered ONLY by budget.
        """

        # 1. Fetch user's country and budget range
        user_sql = text("""
            SELECT country, "budgetMin", "budgetMax"
            FROM users
            WHERE id = :user_id
        """)

        user_df = pd.read_sql(user_sql, self.engine, params={"user_id": user_id})

        if user_df.empty:
            # User not found — return empty lists
            empty = pd.DataFrame(columns=["id", "price", "rating"])
            return {
                "by_country_and_budget": empty.copy(),
                "by_budget": empty.copy(),
            }

        country = user_df.loc[0, "country"]
        budget_min = user_df.loc[0, "budgetMin"]
        budget_max = user_df.loc[0, "budgetMax"]

        # 2. Compute extended budget range (+20% for max)
        price_min = budget_min if pd.notnull(budget_min) else None
        price_max = budget_max * 1.2 if pd.notnull(budget_max) else None

        price_min = float(price_min) if price_min is not None else None
        price_max = float(price_max) if price_max is not None else None

        # 3. Fetch yachts filtered by country AND budget
        yachts_country_budget_sql = text("""
            SELECT id,
                   "summerLowSeasonPrice" AS price,
                   rating
            FROM yachts
            WHERE
                (:country IS NULL OR country = :country)
                AND (:price_min IS NULL OR "summerLowSeasonPrice" >= :price_min)
                AND (:price_max IS NULL OR "summerLowSeasonPrice" <= :price_max)
        """)

        df_country_budget = pd.read_sql(
            yachts_country_budget_sql,
            self.engine,
            params={
                "country": country,
                "price_min": price_min,
                "price_max": price_max,
            },
        )

        # 4. Fetch yachts filtered ONLY by budget (ignore country)
        yachts_budget_sql = text("""
            SELECT id,
                   "summerLowSeasonPrice" AS price,
                   rating
            FROM yachts
            WHERE
                (:price_min IS NULL OR "summerLowSeasonPrice" >= :price_min)
                AND (:price_max IS NULL OR "summerLowSeasonPrice" <= :price_max)
        """)

        df_budget = pd.read_sql(
            yachts_budget_sql,
            self.engine,
            params={
                "price_min": price_min,
                "price_max": price_max,
            },
        )

        # 5. Normalize returned structure in case DataFrames are empty
        base_cols = ["id", "price", "rating"]

        if df_country_budget.empty:
            df_country_budget = pd.DataFrame(columns=base_cols)
        else:
            df_country_budget = df_country_budget[base_cols]

        if df_budget.empty:
            df_budget = pd.DataFrame(columns=base_cols)
        else:
            df_budget = df_budget[base_cols]

        return {
            "by_country_and_budget": df_country_budget,
            "by_budget": df_budget,
        }

    def _predict_for_candidates(
            self, user_id: str, candidates: List[str]
        ) -> List[Tuple[str, float]]:
            """
            Helper: predict scores for a list of candidate item IDs for a given user.
            Returns list of (item_id, predicted_score), sorted by score desc.
            """
            estimates: List[Tuple[str, float]] = []

            for iid in candidates:
                # Surprise expects string IDs; convert defensively
                uid_str = str(user_id)
                iid_str = str(iid)
                try:
                    pred = self.algo.predict(uid_str, iid_str, verbose=False)
                    estimates.append((iid_str, pred.est))
                except Exception:
                    # If the model cannot handle some item (e.g. cold-start), skip it
                    continue

            # Sort candidates by predicted score in descending order
            estimates.sort(key=lambda x: x[1], reverse=True)
            return estimates

    def recommend_ids_for_user(self, user_id: str) -> List[str]:
        """
        Recommend top-N item IDs for a given user using the trained Surprise model.
        """

        user_id_str = str(user_id)

        # 1. Precompute item universe and "seen" sets from the training data.
        #    This could be cached across calls if needed.
        item_universe = set(self.train_df["yacht_id"].astype(str).unique())
        seen_by_user_dict = (
            self.train_df.groupby("user_id")["yacht_id"]
            .apply(lambda s: set(map(str, s)))
            .to_dict()
        )
        seen_by_user = seen_by_user_dict.get(user_id_str, set())

        # 2. Get candidate DataFrames from hydrate_yachts
        yachts_data = self.hydrate_yachts(user_id_str)
        df_country_budget = yachts_data.get("by_country_and_budget", pd.DataFrame())
        df_budget = yachts_data.get("by_budget", pd.DataFrame())

        # 3. Filter out items already seen by the user
        if not df_country_budget.empty:
            df_country_budget = df_country_budget[
                ~df_country_budget["id"].astype(str).isin(seen_by_user)
            ]

        if not df_budget.empty:
            df_budget = df_budget[
                ~df_budget["id"].astype(str).isin(seen_by_user)
            ]

        # Convert DataFrames to candidate ID lists
        candidates_country_budget = (
            df_country_budget["id"].astype(str).tolist()
            if not df_country_budget.empty
            else []
        )
        candidates_budget = (
            df_budget["id"].astype(str).tolist()
            if not df_budget.empty
            else []
        )

        # If both candidate sets are empty, we will rely entirely on the fallback later.
        # 4. Predict scores for each candidate list (if not empty)
        est_country = (
            self._predict_for_candidates(user_id_str, candidates_country_budget)
            if candidates_country_budget
            else []
        )
        est_budget = (
            self._predict_for_candidates(user_id_str, candidates_budget)
            if candidates_budget
            else []
        )

        # 5. Take top-N from each list
        top_country = est_country[:COUNTRY_CANDIDATES_N]
        top_budget = est_budget[:CANDIDATES_N]

        # 6. Merge and deduplicate, keeping the highest score per item
        merged_scores: Dict[str, float] = {}
        for iid, score in top_country + top_budget:
            if iid not in merged_scores or score > merged_scores[iid]:
                merged_scores[iid] = score

        merged_list = sorted(
            merged_scores.items(), key=lambda x: x[1], reverse=True
        )
        final_ids = [iid for iid, _ in merged_list]

        # 7. Fallback strategy if too few recommendations
        #    Example: if we got < FALLBACK_MIN_N (e.g., < 6) candidates,
        #    then use all unseen items from the training universe.
        if len(final_ids) < FALLBACK_MIN_N:
            # logger.info("Fallback case for User %s: %s items", user_id, len(final_ids))

            # Global candidates: all items from train universe that user has not seen
            global_candidates = list(item_universe - seen_by_user)

            if not global_candidates:
                # User has seen everything; return whatever we have (possibly empty)
                return final_ids

            global_est = self._predict_for_candidates(user_id_str, global_candidates)
            # Take top FALLBACK_GLOBAL_N (e.g. 12) globally
            final_ids_global = [iid for iid, _ in global_est[:(FALLBACK_GLOBAL_N-len(final_ids))]]

            return final_ids + final_ids_global

        return final_ids
    
    def generate_and_save_recommendations(self) -> None:
        """
        Generate recommendations for all users in the train set and store them
        into users.recommendations (uuid[]) column, using raw_connection()
        for faster bulk updates.
        """
        if self.engine is None:
            raise RuntimeError("DB engine is not initialized. Call load_data() first.")
        if self.train_df is None or self.train_df.empty:
            raise RuntimeError("train_df is empty. Train/split before generating recommendations.")

        all_users = sorted(map(str, self.train_df["user_id"].astype(str).unique().tolist()))
        logger.info(
            "🚀 STARTING generating & saving recommendations into users.recommendations for %d users...",
            len(all_users),
        )

        # Ensure the target column exists
        with self.engine.begin() as conn:
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS "recommendations" uuid[];
            """))

        # 2. Generate recommendations for each user
        prepared_updates_with_recs = []  # (recommendations_list, user_id)
        prepared_updates_null = []       # (user_id,)
        processed = 0

        gen_start = time.perf_counter()
        for uid in all_users:
            cand_ids = self.recommend_ids_for_user(uid)

            if cand_ids:
                prepared_updates_with_recs.append((cand_ids, uid))
            else:
                prepared_updates_null.append((uid,))

            processed += 1
            if processed % 200 == 0:
                logger.info("[INFO] prepared recommendations for %d users...", processed)

        gen_time = time.perf_counter() - gen_start
        logger.info("⏱ TIMER: Recommendation generation for %d users took %.2f seconds", processed, gen_time)

        # 3. Fast write into DB with raw_connection()
        db_start = time.perf_counter()
        conn = self.engine.raw_connection()
        cursor = None
        try:
            cursor = conn.cursor()

            # update with recommendations
            if prepared_updates_with_recs:
                cursor.executemany(
                    'UPDATE users SET "recommendations" = %s::uuid[] WHERE id = %s::uuid',
                    prepared_updates_with_recs
                )

            # update without recommendations -> NULL
            if prepared_updates_null:
                cursor.executemany(
                    'UPDATE users SET "recommendations" = NULL WHERE id = %s::uuid',
                    prepared_updates_null
                )

            conn.commit()
        finally:
            if cursor is not None:
                cursor.close()
            conn.close()

        db_time = time.perf_counter() - db_start
        logger.info(
            "⏱ TIMER: 🗄 DB: Saved recommendations for %d users into users.recommendations in %.2f seconds "
            "(DB write only).",
            len(all_users),
            db_time,
        )

    def test_recommendations(self):
        logger.info("Start testing recommendations...")
        # Test recommendation 
        first_5_users = sorted(self.train_df["user_id"].unique().tolist())[:10]

        for user_id in first_5_users:
            cand_ids = self.recommend_ids_for_user(user_id)
            if not cand_ids:
                continue
            logger.info("User %s: %s", user_id, cand_ids[:5])

        



def main():
    logging.basicConfig(
        level=logging.INFO,  # або DEBUG для детальніших логів
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    model = RecommendationModel()
    model.load_data()
    model.prepare_data()
    metrics=model.train_test_model()
    model.store_model_artifacts(metrics)

    # store recommendations to database
    model.generate_and_save_recommendations()

    # test recommendations
    model.test_recommendations()

if __name__ == "__main__":
    main()