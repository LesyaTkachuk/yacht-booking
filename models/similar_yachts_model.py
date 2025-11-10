import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

class YachtRecommender:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_matrix = None
        self.knn_model = None
        self.scaler = None
        self.yacht_id_to_idx = {}
        self.idx_to_yacht_id = {}
        
    def prepare_features(self):
        df = self.df.copy()
        
        numerical_features = ['guests', 'cabins', 'crew', 'length', 'year', 'rating']

        for col in numerical_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
        
        price_cols = ['summerLowSeasonPrice', 'summerHighSeasonPrice', 'winterLowSeasonPrice', 'winterHighSeasonPrice']
        for col in price_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['avg_price'] = df[price_cols].mean(axis=1, skipna=True)
        df['avg_price'] = df['avg_price'].fillna(df['avg_price'].median())
        df['log_price'] = np.log1p(df['avg_price'])
        
        type_dummies = pd.get_dummies(df['type'], prefix='type').astype(int)
        
        top_marinas = df['baseMarina'].value_counts().head(50).index 
        df['marina_grouped'] = df['baseMarina'].apply(
            lambda x: x if x in top_marinas else 'Other'
        )
        marina_dummies = pd.get_dummies(df['marina_grouped'], prefix='marina').astype(int)
        
        country_dummies = pd.get_dummies(df['country'], prefix='country').astype(int)
        
        feature_cols = numerical_features + ['log_price']
        
        numerical_df = df[feature_cols]
        
        feature_matrix = pd.concat([
            numerical_df,
            type_dummies,
            marina_dummies,
            country_dummies
        ], axis=1)
        
        feature_matrix = feature_matrix.fillna(0)
        
        print(f"✅ Feature matrix created: {feature_matrix.shape}")
        print(f"   Numerical features: {len(feature_cols)}")
        print(f"   Type dummies: {type_dummies.shape[1]}")
        print(f"   Marina dummies: {marina_dummies.shape[1]}")
        print(f"   Country dummies: {country_dummies.shape[1]}")
        
        return feature_matrix
    
    def fit(self, n_neighbors=11, metric='cosine'):
        print(f"\n🔧 Training KNN model (n_neighbors={n_neighbors}, metric={metric})...")
        
        self.feature_matrix = self.prepare_features()
        
        self.scaler = StandardScaler()
        feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix.values)
        
        if 'id' in self.df.columns:
            self.yacht_id_to_idx = {yacht_id: idx for idx, yacht_id in enumerate(self.df['id'])}
            self.idx_to_yacht_id = {idx: yacht_id for yacht_id, idx in self.yacht_id_to_idx.items()}
        else:
            self.yacht_id_to_idx = {idx: idx for idx in range(len(self.df))}
            self.idx_to_yacht_id = self.yacht_id_to_idx
        
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm='auto',
            n_jobs=-1
        )
        
        self.knn_model.fit(feature_matrix_scaled)
        
        print(f"✅ KNN model trained!")
        
        return self
    
    def recommend(self, yacht_id, top_k=10, filters=None):
        if self.knn_model is None:
            raise ValueError("Model is not trained! Call .fit() before")
        
        if yacht_id not in self.yacht_id_to_idx:
            raise ValueError(f"Yacht ID {yacht_id} not found in dataset")
        
        yacht_idx = self.yacht_id_to_idx[yacht_id]
        
        yacht_feature_series = self.feature_matrix.iloc[yacht_idx]
        yacht_features = self.scaler.transform(yacht_feature_series.values.reshape(1, -1))
        
        distances, indices = self.knn_model.kneighbors(yacht_features)
        
        distances = distances[0][1:]
        indices = indices[0][1:]
        
        if self.knn_model.metric == 'cosine':
            similarities = 1 - distances
        else:
            max_dist = distances.max() if distances.max() > 0 else 1
            similarities = 1 - (distances / max_dist)
        
        recommendations = []
        for idx, similarity in zip(indices, similarities):
            yacht_data = self.df.iloc[idx].copy()
            yacht_data['similarity_score'] = similarity
            recommendations.append(yacht_data)
        
        recommendations_df = pd.DataFrame(recommendations)
        
        return recommendations_df.head(top_k)
    
    def save_model(self, filepath='similar_yachts.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'knn_model': self.knn_model,
                'feature_matrix': self.feature_matrix,
                'scaler': self.scaler,
                'yacht_id_to_idx': self.yacht_id_to_idx,
                'idx_to_yacht_id': self.idx_to_yacht_id,
                'df': self.df
            }, f)
        print(f"✅ Model saved in {filepath}")
    
    @classmethod
    def load_model(cls, filepath='similar_yachts.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls(data['df'])
        recommender.knn_model = data['knn_model']
        recommender.feature_matrix = data['feature_matrix']
        recommender.scaler = data['scaler']
        recommender.yacht_id_to_idx = data['yacht_id_to_idx']
        recommender.idx_to_yacht_id = data['idx_to_yacht_id']
        
        print(f"✅ Model uploaded from {filepath}")
        return recommender


if __name__ == "__main__":
    
    print("--- [Sim Yachts] Starting update 'similarYachts' ---")
    print(os.getenv("DATABASE_USERNAME"))

    db_string = "postgresql+psycopg2://" + os.getenv("DATABASE_USERNAME") + ":" + os.getenv("DATABASE_PASSWORD") + "@" + os.getenv("DATABASE_HOST") + ":" + os.getenv("DATABASE_PORT") + "/" + os.getenv("DATABASE_NAME")
    if not db_string:
        print("❌ ERROR: Environment variable DB_STRING not set.")
        exit(1)
        
    engine = create_engine(db_string)
    query = "SELECT * FROM yachts;"
    
    try:
        raw_conn = engine.raw_connection()
        df = pd.read_sql(query, raw_conn)
    finally:
        raw_conn.close()
    
    print(f"Uploaded {len(df)} yachts from database.")
    
    recommender = YachtRecommender(df)
    recommender.fit(n_neighbors=13, metric='cosine')
    
    recommender.save_model('similar_yachts.pkl')
    
    print("\n🚀 Start generating similar yachts...")
    
    all_recommendations_data = []
    all_yacht_ids = df['id'].unique()
    
    for yacht_id in tqdm(all_yacht_ids, desc="Generating recommendations"):
        try:
            recs_df = recommender.recommend(yacht_id, top_k=11)
            recs_ids = recs_df['id'].tolist()
            
            all_recommendations_data.append({
                'yacht_id': yacht_id,
                'similar_yachts': recs_ids
            })
        except ValueError as e:
            print(f"Error for yacht_id {yacht_id}: {e}")
            all_recommendations_data.append({
                'yacht_id': yacht_id,
                'similar_yachts': []
            })

    print(f"\n✅ Successfull generation for {len(all_recommendations_data)} yachts.")
    
    print(f"📤 Updating 'similarYachts' in 'yachts' table...")
    
    update_query = text("""
        UPDATE yachts
        SET "similarYachts" = :recs
        WHERE id = :id
    """).bindparams(
        bindparam("recs", type_=ARRAY(UUID)),
        bindparam("id",   type_=UUID),
    )
    
    try:
        with engine.begin() as conn:
            for item in tqdm(all_recommendations_data, desc="Database update"):
                yacht_id = str(item['yacht_id'])
                recs = [str(r) for r in item['similar_yachts']]

                conn.execute(update_query, {
                    "id": yacht_id,
                    "recs": recs
                })

        print(f"✅ Data successfully uploaded in 'yachts' table!")
        
    except Exception as e:
        print(f"❌ Error during update 'yachts' in database: {e}")

    print("--- [Sim Yachts] Update 'similarYachts' finished ---")
