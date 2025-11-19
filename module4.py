import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv("events.csv", nrows=100000)
df.head()

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.sort_values(by=['visitorid', 'timestamp']).reset_index(drop=True)
df.head()

buyers = df[df['event'] == 'transaction']['visitorid'].unique()
df_buyers = df[df['visitorid'].isin(buyers)].copy()

user_sequences = (
    df_buyers.groupby('visitorid')['event']
    .apply(list)
    .reset_index()
    .rename(columns={'event': 'sequence'})
)
user_sequences.head()

user_sequences['seq_str'] = user_sequences['sequence'].apply(lambda x: " ".join(x))
user_sequences.head()

vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X = vectorizer.fit_transform(user_sequences['seq_str'])

k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
user_sequences['cluster'] = kmeans.fit_predict(X)

cluster_examples = (
    user_sequences.groupby('cluster')['sequence']
    .apply(lambda x: x.head(5))
)
print(cluster_examples)

plt.figure(figsize=(8,6))
user_sequences['cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.xlabel("Cluster ID")
plt.ylabel("Number of Users")
plt.title("Distribution of User Journey Clusters")
plt.tight_layout()
plt.savefig("cluster_distribution.png")
print("Plot saved as cluster_distribution.png")