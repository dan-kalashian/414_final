import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# Load data
df = pd.read_csv('calculated_stocks.csv')

# Select and drop NaNs
cluster_data = df[['Vol', 'VV']].copy().dropna()

# Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# MiniBatch KMeans
kmeans = MiniBatchKMeans(n_clusters=4, random_state=0, batch_size=10000)
cluster_labels = kmeans.fit_predict(scaled_data)

# Save scaled centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Vol', 'VV'])
centroids.to_csv('vol_cluster_means.csv', index=True)

# Assign back
cluster_data['Cluster'] = cluster_labels
df.loc[cluster_data.index, 'Cluster'] = cluster_data['Cluster']

# Optional label map
label_map = {
    0: 'Low Volatility',
    1: 'Stable Low-Risk',
    2: 'Above Average Volatility',
    3: 'Unstable High-Risk'
}
df.loc[cluster_data.index, 'Cluster_Label'] = cluster_data['Cluster'].map(label_map)

# Save final
df.to_csv('clustered_stocks.csv', index=False)
