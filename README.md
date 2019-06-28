# SVM
SVM classification work on dataset

kernel = linear:

accuracy = 0.91 - 0.97

kernel = rbf:

accuracy = 1.0

Above performance is without PCA, means all 13 features is used. 


Implmentation of result:

applying PCA on dataset, using SVM to get optimization

SVM with linear kernel:

![img](https://github.com/laurence-lin/SVM/blob/master/svm_result.png)

SVM with rbf kernel:

Get performance that is not that good. We can see that PCA influences the performance very much. Even the best performance with rbf kernel couldn't reach good performance by the principal component.

![img](https://github.com/laurence-lin/SVM/blob/master/svm_result2.png)



