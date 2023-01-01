# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load the Data
# For this example, we'll use the Mall Customers dataset.
# Make sure you have a CSV file named 'Mall_Customers.csv' in your working directory.
data = pd.read_csv('Mall_Customers.csv')
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Data Exploration & Preprocessing
# Let's check for missing values and get an overview of the dataset.
print("\nDataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# For clustering, we will select numerical features.
# Here, we choose 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)'.
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Optional: If you want to include 'Gender', you can convert it to numeric using one-hot encoding:
# gender_dummies = pd.get_dummies(data['Gender'], drop_first=True)
# features = pd.concat([features, gender_dummies], axis=1)

# Step 4: Feature Scaling
# Scaling helps ensure that each feature contributes equally to the clustering algorithm.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Determine the Optimal Number of Clusters (Elbow Method)
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow graph, suppose the optimal number of clusters is 5.
optimal_clusters = 5

# Step 6: Apply K-Means Clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
data['Cluster'] = clusters  # Append the cluster assignments to the original dataframe

print("\nCluster counts:")
print(data['Cluster'].value_counts())

# Step 7: Visualize the Clusters
# We'll create a scatter plot for Annual Income vs Spending Score.
# Each cluster is shown in a different color.
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']  # One color per cluster

for i in range(optimal_clusters):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'],
                cluster_data['Spending Score (1-100)'],
                s=50,
                c=colors[i],
                label=f'Cluster {i}')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

# Plot the centroids
# Note: centroids are in scaled space, so we inverse transform them back to original scale.
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 1], centers[:, 2], marker='*', s=200, c='black', label='Centroids')
plt.legend()
plt.show()

# Step 8: Analyzing the Clusters
# You can further explore each segment by summarizing the demographic details.
cluster_summary = data.groupby('Cluster').mean()
print("\nCluster Summary (Mean values):")
print(cluster_summary)
