from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

from dataset import FIFA20

pd.options.mode.chained_assignment = None  # default='warn'


def plot_elbow_method(data):
    print("Creating plot for elbow method")
    wcss = []
    r = range(1, 20)
    for k in r:
        model = KMeans(n_clusters=k)
        model.fit(data)
        wcss.append(model.inertia_)
    plt.plot(r, wcss, marker='.')
    plt.title("Elbow method")
    plt.xlabel("Number of clusters")
    plt.xticks(r)
    plt.ylabel("WCSS")
    plt.savefig('images/elbow.png')


def calculate_means_cluster(data, label, sensitivity=0.15):
    att_list = data.columns[:-1]
    
    atts_to_plot = []
    means_to_plot = []

    for i, att in enumerate(att_list):
        all_mean = data[att].mean()

        cluster_df = data[data['cluster'] == label]
        cluster_mean = cluster_df[att].mean()

        standardized_mean = cluster_mean / all_mean

        if standardized_mean > (1+sensitivity):
            atts_to_plot.append(att)
            means_to_plot.append(standardized_mean)

    mean_df = pd.DataFrame({
        'att_list': atts_to_plot,
        'means_list': means_to_plot
        })
    mean_df.sort_values(by=['means_list'], ascending=False)

    return mean_df


def plot_barchart_cluster(data, label):
    mean_df = calculate_means_cluster(data, label)
    if not mean_df.empty:
        plt.gcf().subplots_adjust(left=0.25)
        plt.title('Cluster {}'.format(label))
        plt.barh(mean_df['att_list'], mean_df['means_list'])
        plt.xlabel('Standardized mean')
        plt.ylabel('Attributes')
        plt.axvline(x=1)
        plt.show()
        plt.savefig('images/fifa_{}.png'.format(label))


def perform_k_means(data, k):
    model = KMeans(n_clusters=k)
    pred = model.fit_predict(data)
    return pred


def perform_hc(data, n, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage=linkage)
    pred = model.fit_predict(data)
    return pred


def perform_em(data, n, cov='full'):
    model = GaussianMixture(n_components=n, covariance_type=cov, n_init=10)
    pred = model.fit_predict(data)
    return pred


def main(): 
    data = FIFA20.get_data()

    plot_elbow_method(data)

    # 4 clusters chosen using elbow method
    k = 5
    pred = perform_k_means(data, k)

    fig, ax = plt.subplots()
    
    data['cluster'] = pred

    for i in range(k):
        plot_barchart_cluster(data, i)

if __name__ == "__main__":
    main()