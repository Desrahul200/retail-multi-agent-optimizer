import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('dh_demand.csv', parse_dates=['date'])

# Feature engineering for customer behavior (simulate segments)
# We'll use price, promotion_flag, competitor_price, units_sold as behavioral features
features = df[['price', 'promotion_flag', 'competitor_price', 'units_sold']].values

# Dimensionality reduction with UMAP
reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
embedding = reducer.fit_transform(features)

# Clustering with KMeans
n_segments = 3
kmeans = KMeans(n_clusters=n_segments, random_state=42)
segments = kmeans.fit_predict(embedding)
df['segment'] = segments

# Visualize segments
plt.figure(figsize=(8, 5))
for i in range(n_segments):
    plt.scatter(embedding[segments == i, 0], embedding[segments == i, 1], label=f'Segment {i}')
plt.title('Customer Segments (UMAP + KMeans)')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.legend()
plt.tight_layout()
plt.show()

# Estimate likelihood of buying at current price and reacting to promo
# For each segment, calculate mean units_sold with/without promo
print("\nSegment behavior summary:")
for i in range(n_segments):
    seg = df[df['segment'] == i]
    mean_buy = seg['units_sold'].mean()
    mean_promo = seg[seg['promotion_flag'] == 1]['units_sold'].mean()
    mean_no_promo = seg[seg['promotion_flag'] == 0]['units_sold'].mean()
    print(f"Segment {i}: Avg units_sold={mean_buy:.2f}, With promo={mean_promo:.2f}, Without promo={mean_no_promo:.2f}") 