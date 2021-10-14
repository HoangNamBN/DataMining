import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# thư viện dùng để tính khoảng cách
from scipy.spatial.distance import cdist

datas = pd.read_csv("Mall_Customers.csv").drop('CustomerID', axis=1).values
train = datas[:, 1:4]
n_cluster = 2


def kmeans_init_centers(train, n_cluster):
    return train[np.random.choice(train.shape[0], n_cluster, replace=False)]

def kmeans_predict_labels(train, centers):
    D = cdist(train,centers)
    return np.argmin(D, axis=1)

def kmeans_updates_centers(train, labels, n_cluster):
    centers = np.zeros((n_cluster, train.shape[1]))
    for k in range(n_cluster):
        Xk = train[labels == k, :]
        centers[k:] = np.mean(Xk, axis=0)
    return centers

def kmeans_has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

def kmeans_visualize(train, centers, labels, n_cluster, title):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt_colors = ['b','g','r','c','m','y','k','w']
    for i in range(n_cluster):
        data = train[labels == i]
        plt.plot(data[:, 0], data[:, 1], plt_colors[i] + '^', markersize=4,
                 label='cluster_' + str(i))  # Vẽ cụm i lên đồ thị
        plt.plot(centers[i][0], centers[i][1], plt_colors[i + 4] + 'o', markersize=10,
                 label='center_' + str(i))  # Vẽ tâm cụm i lên đồ thị
    plt.legend()  # Hiện bảng chú thích
    plt.show()

def kmeans(init_centes, init_labels, train, n_cluster):
    centers = init_centes
    labels = init_labels
    times = 0
    while True:
        labels = kmeans_predict_labels(train, centers)
        kmeans_visualize(train, centers, labels, n_cluster, 'Assigned label for data at time = ' + str(times + 1))
        new_centers = kmeans_updates_centers(train, labels, n_cluster)
        if kmeans_has_converged(centers, new_centers):
            break
        centers = new_centers
        kmeans_visualize(train, centers, labels, n_cluster, 'Update center possition at time = ' + str(times + 1))
        times += 1
    return (centers, labels, times)

if __name__ == "__main__":
    init_centes = kmeans_init_centers(train, n_cluster)
    init_labels = np.zeros(train.shape[0])
    kmeans_visualize(train, init_centes, init_labels, n_cluster,
                     'Init centers in the first run. Assigned all data as cluster 0')
    centers, labels, times = kmeans(init_centes, init_labels, train, n_cluster)

    print('Done! Kmeans has converged after', times, 'times')