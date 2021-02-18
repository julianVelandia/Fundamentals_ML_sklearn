import pandas as pd 
import sklearn
import matplotlib.pyplot as plt


from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA


from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']
    dt_features = StandardScaler().fit_transform(dt_features)
    x_train, x_test, y_train, y_test=train_test_split(dt_features, dt_target, test_size=.3, random_state=True)


    kpca = KernelPCA(n_components=4, kernel = 'poly')
    kpca.fit(x_train)

    dt_train = kpca.transform(x_train)
    dt_test = kpca.transform(x_test)

    logistic= LogisticRegression(solver='lbfgs')
    logistic.fit(dt_train, y_train)
    print('score kpca: ', logistic.score(dt_test,y_test))


    pca = PCA(n_components=3)
    pca.fit(x_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(x_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    logistic = LogisticRegression(solver="lbfgs")

    dt_train = pca.transform(x_train)
    dt_test = pca.transform(x_test)

    logistic.fit(dt_train,y_train)

    print("score pca: ", logistic.score(dt_test,y_test))

    dt_train = ipca.transform(x_train)
    dt_test = ipca.transform(x_test)
    logistic.fit(dt_train,y_train)
    
    print("score ipca: ", logistic.score(dt_test,y_test))

