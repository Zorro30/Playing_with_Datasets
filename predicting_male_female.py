from sklearn import tree, svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score


# [ height, weight, shoe_size]
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47], [175,64,39], [177,70,40], [171,75,42], [181,85,43]]

Y = ['female','female','female','male','female','male','female','male','female','male']

clf_tree = tree.DecisionTreeClassifier()
clf_svm = svm.SVC()
clf_PR = Perceptron()
clf_gnb = GaussianNB()
clf_cntrd = NearestCentroid()

# Desision Tree classifier 
clf_tree = clf_tree.fit(X,Y)
# SVM Classifier
clf_svm = clf_svm.fit(X,Y)
# Perceptron
clf_PR = clf_PR.fit(X,Y)
# Naive Bayes Classifier
clf_gnb = clf_gnb.fit(X,Y)
# Nearest Centroid Classifier
clf_cntrd = clf_cntrd.fit(X,Y)


#using accuracy_score
prediction_tree = clf_tree.predict(X)
accuracy = accuracy_score(Y,prediction_tree) *100

prediction_svm = clf_svm.predict([[181,80,44]])
prediction_PR = clf_PR.predict([[181,80,44]])
prediction_gnb = clf_gnb.predict([[181,80,44]])
prediction_cntrd = clf_cntrd.predict([[181,80,44]])
print(accuracy, prediction_svm, prediction_PR, prediction_gnb,prediction_cntrd)