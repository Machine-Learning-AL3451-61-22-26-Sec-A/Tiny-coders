import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.title("KMeans Clustering")

    # Load the dataset
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        df1 = pd.DataFrame(data)
        st.write(df1)

        # Extract features
        f1 = df1['Distance_Feature'].values
        f2 = df1['Speeding_Feature'].values
        X = np.matrix(list(zip(f1, f2)))

        # Plot the dataset
        st.write("Dataset")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(f1, f2)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 50])
        ax.set_title('Dataset')
        ax.set_xlabel('Distance_Feature')
        ax.set_ylabel('Speeding_Feature')
        st.pyplot(fig)

        # Perform KMeans clustering
        kmeans_model = KMeans(n_clusters=3).fit(X)

        # Plot the clustered data
        st.write("KMeans Clustering")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['b', 'g', 'r']
        markers = ['o', 'v', 's']
        for i, label in enumerate(kmeans_model.labels_):
            ax.scatter(f1[i], f2[i], color=colors[label], marker=markers[label], edgecolor='k')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 50])
        ax.set_title('KMeans Clustering')
        ax.set_xlabel('Distance_Feature')
        ax.set_ylabel('Speeding_Feature')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
