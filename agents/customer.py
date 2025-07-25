import pandas as pd
import umap.umap_ as umap
from sklearn.cluster import KMeans
import numpy as np

def run(state: dict, full_df: pd.DataFrame) -> dict:
    sub = full_df[full_df['product_id'] == state['product_id']]
    if sub.empty:
        return state
    feats = sub[['price','promotion_flag','competitor_price','units_sold']].values
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    emb = reducer.fit_transform(feats)
    emb = np.asarray(emb)
    kmeans = KMeans(n_clusters=3, random_state=42)
    segs = kmeans.fit_predict(emb)
    sub = sub.copy()
    sub['segment'] = segs
    state['segment_info'] = sub.groupby('segment')['units_sold'].mean().to_dict()
    state['segments'] = segs
    state['umap_x']   = emb[:,0]
    state['umap_y']   = emb[:,1]
    return state 