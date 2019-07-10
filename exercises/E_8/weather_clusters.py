import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer



def transform(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    # X.values = pd.to_numeric(X.values, downcast='float')
    return scaler.transform(X)


def get_pca(X):
    """
    Transform data to 2D points for plotting. Should return an array with shape (n, 2).
    """
    flatten_model = make_pipeline(
        FunctionTransformer(transform, validate=False),
        PCA(2)
    )
    X2 = flatten_model.fit_transform(X)
    assert X2.shape == (X.shape[0], 2)
    return X2


def get_clusters(X):
    """
    Find clusters of the weather data.
    """
    model = make_pipeline(
        FunctionTransformer(transform, validate=False),
        KMeans(10)
    )
    model.fit(X)
    return model.predict(X)


def main():
    data = pd.read_csv(sys.argv[1])
    #data = pd.read_csv("monthly-data-labelled.csv")

    X = data.drop(columns="city")
    y = data["city"]

    X2 = get_pca(X)
    clusters = get_clusters(X)
    plt.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap='Set1', edgecolor='k', s=20)
    plt.savefig('clusters.png')

    df = pd.DataFrame({
        'cluster': clusters,
        'city': y,
    })
    counts = pd.crosstab(df['city'], df['cluster'])
    print(counts)


if __name__ == '__main__':
    main()