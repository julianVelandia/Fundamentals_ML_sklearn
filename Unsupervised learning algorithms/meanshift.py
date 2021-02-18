import pandas as pd
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift

if __name__ == "__main__":

    dataset = pd.read_csv("../data/candy.csv")#Load DataSet

    X = dataset.drop('competitorname', axis=1)#Drop irrelevant axis

    meanshift = MeanShift().fit(X)#Model creation 
    
    print(max(meanshift.labels_))
    print("="*64)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    print("="*64)
    print(dataset)

    pca = PCA(n_components=2)
    pca.fit(X)
    pca_data = pca.transform(X)
        
    meanshift = MeanShift().fit(pca_data)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=meanshift.predict(pca_data))
    plt.scatter(meanshift.cluster_centers_[:, 0], meanshift.cluster_centers_[:, 1], c='black', s=200)
    plt.show()
