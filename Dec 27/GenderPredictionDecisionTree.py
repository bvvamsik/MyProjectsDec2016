###############################################################################
# Dependencies:                                                               #
###############################################################################
# -> Need numpy to be installed before executing the code
# -> Preferred using WINPYTHON 3.5.2.3 for all libraries included before execution

# Importing SCIPY libraries
from scipy import sparse

# Importing SKLEARN library for performing machine learning task lke creating a decision tree using multiple parameters
from sklearn import tree

#[Height, Weight, Shoe size]
X=[[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,37],[171,75,42],[181,85,43]]

Y=['male','female','female','female','male','male','male','female','male','female','male']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

# 'prediction' variable now returns the prediction with maximum probability as a result from decision tree
prediction = clf.predict([[180,79,43]])

print(prediction)