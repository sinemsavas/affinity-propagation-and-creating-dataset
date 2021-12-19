# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 11:59:33 2021

@author: Sinem
"""

#Using affinity propagation after creating synthetic classification dataset
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot


#I created a dataset with 1000 samples (random state tells how the distribution should be)
X,y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

#adding affinity propagation to the model
model = AffinityPropagation()

#run the machine learning model
model.fit(X)

#prediction
yhat = model.predict(X)

#part of taking prediction results and sorting them into clusters
#retrieve unique clusters
clusters = unique(yhat)

for cluster in clusters:
	# get row indexes 
	row_ix = where(yhat == cluster)
	# create scatter 
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
#pyplot
pyplot.show()


