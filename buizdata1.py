import matplotlib.pyplot  as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as metric
import sklearn.metrics as metrics
from sklearn import cluster,datasets
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets.samples_generator import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets.samples_generator import make_circles
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mpl_toolkits import mplot3d


def gau(X,y):
    model =GaussianNB()
    model.fit(X,y)
    y =model.predict(X)
    lim =plt.axis()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='RdBu', alpha=0.1)
    plt.plot(y)
    plt.axis(lim)
    print(model.score(X,y))
    plt.show()
def kmeans(X,y):
    kmean = KMeans(n_clusters=4)
    kmean.fit(X)
    y_kmeans = kmean.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)
    centers =kmean.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
def hiclus(X,y):
    ward2 = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X)
    ward3 = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
    ward4 = cluster.AgglomerativeClustering(n_clusters=4, linkage='ward').fit(X)
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=ward2.labels_)
    plt.title("K=2")
    plt.subplot(132)
    plt.scatter(X[:, 0], X[:, 1], c=ward3.labels_)
    plt.title("K=3") 
    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1], c=ward4.labels_)
    plt.title("K=4")
    plt.show()
def svm(X,y):
    X, y = make_blobs(n_samples=50, centers=2,
                      random_state=0, cluster_std=0.60)
    plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap ='autumn')
    plt.show()
def randforest(X,y):
    forest = RandomForestClassifier(n_estimators = 100)
    forest.fit(X, y)
    forest.score(X,y)
    print("#Errors: %i" % np.sum(y != forest.predict(X)))
    plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap ='autumn')
    plt.show()
def multcla(X,y):
    RANDOM_STATE = 42
    FIG_SIZE = (10, 7) 
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=RANDOM_STATE)
    # Fit to data and predict using pipelined GNB and PCA.
    unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
    unscaled_clf.fit(X_train, y_train)
    pred_test = unscaled_clf.predict(X_test)
    # Fit to data and predict using pipelined scaling, GNB and PCA.
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
    std_clf.fit(X_train, y_train)
    pred_test_std = std_clf.predict(X_test)
    print('\nPrediction accuracy for the normal test dataset with PCA')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))
    print('\nPrediction accuracy for the standardized test dataset with PCA')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))
    print('{:.2%}\n'.format(metrics.classification_report(y_test, pred_test_std)))
    pca = unscaled_clf.named_steps['pca']
    pca_std = std_clf.named_steps['pca']
    # Show first principal components
    print('\nPC 1 without scaling:\n', pca.components_[0])
    print('\nPC 1 with scaling:\n', pca_std.components_[0])
    
    # Show first principal components
    print('\nPC 1 without scaling:\n', pca.components_[0])
    print('\nPC 1 with scaling:\n', pca_std.components_[0])
    
    # Use PCA without and with scale on X_train data for visualization.
    X_train_transformed = pca.transform(X_train)
    scaler = std_clf.named_steps['standardscaler']
    X_train_std_transformed = pca_std.transform(scaler.transform(X_train))
    
    # visualize standardized vs. untouched dataset with PCA performed
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)
    for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
     ax1.scatter(X_train_transformed[y_train == l, 0],
     X_train_transformed[y_train == l, 1],color=c,label='class %s' % l,alpha=0.5,marker=m)
     ax1.set_title('Training dataset after PCA')
     ax2.set_title('Standardized training dataset after PCA')
     for ax in (ax1, ax2):
         ax.set_xlabel('1rst principal component')
         ax.set_ylabel('2nd principal component')
         ax.legend(loc='upper right')
         ax.grid()
     plt.tight_layout()
     plt.show()
def trea(X,y):
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    decision_tree = decision_tree.fit(X, y)
    r = export_text(decision_tree)
    print(r)

    
def scr2(X,y):
    print("**************************************")
    print("*     Classification Model           *")
    print("*    1-GaussianNB                    *")
    print("*    2- KMeans                       *")
    print("*    3- hierachie Clustering         *")
    print("*    4- Support Vector Machine       *")
    print("*    5- Multiclassification          *")
    print("*    6- Tree Classication            *")
    print("**************************************")
    var2 =int(input('Enter the model No\t'))
    if var2==1:
        gau(X,y)
    else:
        if var2 ==2:
           kmeans(X,y)
        else:
            if var2 ==3:
               hiclus(X, y)
            else:
                if var2 ==4:
                   svm(X, y)
                else:
                    if var2 ==5:
                        multcla(X, y)
                    else:
                        trea(X,y)

    scr2(X,y)


        
    
    
    
