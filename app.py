# app.py

from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

app = Flask(__name__)

@app.route('/')
def index():
    # Load the dataset (ensure Mall_Customers.csv is in your project folder)
    data = pd.read_csv('Mall_Customers.csv')
    
    # Select relevant features for clustering
    features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply K‑Means clustering (assume optimal clusters = 5 for this example)
    optimal_clusters = 5
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    data['Cluster'] = clusters
    
    # ----- Create Graphs using Plotly -----
    
    # Graph 1: Scatter Plot of Annual Income vs Spending Score by Cluster
    fig_scatter = px.scatter(
        data, x='Annual Income (k$)', y='Spending Score (1-100)',
        color='Cluster', title='Customer Segments: Income vs Spending Score'
    )
    scatter_div = fig_scatter.to_html(full_html=False)
    
    # Graph 2: Bar Chart for Number of Customers per Cluster
    cluster_counts = data['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig_bar = px.bar(
        cluster_counts, x='Cluster', y='Count',
        title='Number of Customers per Cluster'
    )
    bar_div = fig_bar.to_html(full_html=False)
    
    # Graph 3: Histogram for Age Distribution by Cluster
    fig_hist = px.histogram(
        data, x='Age', color='Cluster', barmode='overlay',
        title='Age Distribution by Cluster'
    )
    hist_div = fig_hist.to_html(full_html=False)
    
    # Graph 4: Box Plot for Spending Score by Cluster
    fig_box = px.box(
        data, x='Cluster', y='Spending Score (1-100)',
        title='Spending Score Distribution by Cluster'
    )
    box_div = fig_box.to_html(full_html=False)
    
    return render_template(
        'index.html',
        scatter_div=scatter_div,
        bar_div=bar_div,
        hist_div=hist_div,
        box_div=box_div
    )

if __name__ == '__main__':
    app.run(debug=True)
