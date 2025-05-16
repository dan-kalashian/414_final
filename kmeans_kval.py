from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

import gc
gc.collect()

df=pd.read_csv('calculated_stocks.csv')
df=df[['Vol', 'VV']].copy()


inertia = []
k_values = range(1, 15)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()