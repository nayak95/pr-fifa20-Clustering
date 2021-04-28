from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

from dataset import S_Sets, Spiral, FIFA20



def calculate_accuracy(labels, pred_labels):
    label_map = np.argmax(contingency_matrix(labels, pred_labels), axis=1).tolist()
    #print("argmax ", np.argmax(contingency_matrix(true_labels, pred_labels), axis=1))

    def map_labels(x):
        try:
            return label_map.index(x) + 1
        except ValueError:
            return 0

    mapped_pred_labels = list(map(map_labels, pred_labels))
    return accuracy_score(labels, mapped_pred_labels)
    

def perform_k_means(n, data):
    kmeans = KMeans(n_clusters=n)
    pred_labels = kmeans.fit_predict(data)
    return pred_labels    


def perform_hierarchical_clustering(n, data):
    cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
    pred_labels = cluster.fit_predict(data)
    return pred_labels


def perform_em_clustering(n, data):
    model = GaussianMixture(n_components=n, covariance_type='full')
    pred_labels = model.fit_predict(data)
    return pred_labels


def main():

    plot = True
    #for i in range(1, 5):
    data = Spiral.get_data()
    labels = Spiral.get_labels()


    n = 4

    k_pred = perform_k_means(n, data)
    hc_pred = perform_hierarchical_clustering(n, data)
    em_pred = perform_em_clustering(n, data)

    if labels:
        k_acc = calculate_accuracy(labels, k_pred)
        hc_acc = calculate_accuracy(labels, hc_pred)
        em_acc = calculate_accuracy(labels, em_pred)
        print("k-means: {}\nHC: {}\nEM: {}".format(k_acc, hc_acc, em_acc))
        
    if plot:
            fig, ax1 = plt.subplots(1, 1)
            ax1.tick_params(
                axis='both',
                which='both',
                labelbottom=False,
                labelleft=False)
            #ax2.tick_params(
            #   axis='both',
            #   which='both',
            #   labelbottom=False,
            #   labelleft=False)

            #ax1.set_title('k-means++')
            ax1.set_title('Hierarchical (complete linkage)')
            #ax3.set_title('Hierarchical (single linkage)')
            #ax1.scatter(data['x'], data['y'], c=k_pred, cmap='viridis', s=10)
            ax1.scatter(data['x'], data['y'], c=hc_pred, cmap='viridis', s=10)
            #ax3.scatter(data['x'], data['y'], c=em_pred, cmap='viridis', s=10)

            plt.savefig('images/spiral.png')


if __name__ == "__main__":
    main()