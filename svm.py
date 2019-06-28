import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

'''
SVM: an supervised algorithm that could accomplish clean nonlinear classification

'''

file = pd.read_csv('wine.data')
data = file.values
y_data = data[:, 0]
x_data = data[:, 1:]

shuffle = np.random.permutation(x_data.shape[0])
x_data = x_data[shuffle]
y_data = y_data[shuffle]

total = x_data.shape[0]
train_end = int(total * 0.8)
x_train = x_data[0:train_end, :]
y_train = y_data[0:train_end]
x_test = x_data[train_end:, :]
y_test = y_data[train_end:]  

print(x_train.shape)

scaler = StandardScaler() # use standardize to scale the data
scaler.fit(x_train) # compute variance & mean for the scaler object attribute
# Scaling
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

#test_prob = svm.predict_proba(x_test) # predict probability of each class

'''def PCA(x):
    
    cov_x = np.cov(x.T)
    u, s, v = np.linalg.svd(cov_x)
    k = 2
    proj = u[:, 0:k]
    pca_x = np.matmul(x, proj)
    
    return pca_x

x_train_std = PCA(x_train_std)
x_test_std = PCA(x_test_std)'''

svm = svm.SVC(kernel = 'rbf', probability = True)
svm.fit(x_train_std, y_train)

predict_y = svm.predict(x_test_std)
label_y = y_test
print(predict_y)
print(label_y)

correct = (label_y == predict_y).astype(int)
correct_rate = np.mean(correct)

print('Correct test rate', correct_rate)


def plot_decision_boundary(X, y, clf, test_ind = None, resolution = 0.02):
    '''
    x: 2D array, size [batch, features] , features = 2
    '''
    
    markers = ('s', 'o', 'v') # markers for plot
    colors = ('red', 'green', 'blue', 'gray')
    n_class = len(np.unique(y))
    cmap = ListedColormap(colors[:n_class])
    
    x1min, x1max = X[:, 0].min(), X[:, 0].max()
    x2min, x2max = X[:, 1].min(), X[:, 1].max()
    
    xx, yy = np.meshgrid(np.arange(x1min, x1max, resolution), np.arange(x2min, x2max, resolution))
    grid_point = np.c_[xx.ravel(), yy.ravel()] # [feature, sampples]

    z = svm.predict(grid_point).reshape(xx.shape)
    plt.contour(xx, yy, z, alpha = 0.4, cmap = cmap)
    plt.xlim(x1min, x1max)
    plt.ylim(x2min, x2max)
    
    # plot data points
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(
                x = X[y == c1, 0], # data points of each class separately
                y = X[y == c1, 1],
                cmap = cmap[idx], # use index of class to get from cmap
                alpha = 0.4,
                edgecolor = 'black',
                markers = markers[idx],
                label = c1
                )
    # highlight test samples
    if test_ind:
       plt.scatter(
                x = x_test[:, 0],
                y = x_test[:, 1],
                c = '',
                alpha = 1.0,
                markers = 'o',
                edgecolor = 'black',
                label = 'test set'
                )
       
plot_decision_boundary(x_train_std, y_train, True)
plt.xlabel('component 1')
plt.ylabel('component 2')

plt.show()





