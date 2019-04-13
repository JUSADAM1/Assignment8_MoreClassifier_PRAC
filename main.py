#importing data sets
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

#SPLITTING THE TEST DATA  INTO X AND Y VALUES.
X = iris.data
y = iris.target

#what I am doing now splitting the data into two partitions
#which will be my training  and my test data.

#1/2 for test and half for training
#here I am using X_train and y_train for my training data
# and the x AND Y_TEST for my test data
from sklearn.model_selection import train_test_split
#splitting the data in half
#the way the data is being split is by calling the X's and Y's slpitting with(test_size=.5)
#meaning we are splitting the iris data in half for both data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

#now I am going to our go to classififer
#decision tree!! whoo hoo
from sklearn import tree

decision_tree_classifier = tree.DecisionTreeClassifier()

#now I want to train the classifier on my trainning data
decision_tree_classifier.fit(X_train, y_train)

#Call predict method to classify our testing data
predictions_from_decision_tree_classifier = decision_tree_classifier.predict(X_test)


print("AWESOME NOW!!! Lets print our iris predictions out.")
print("The data you see before you correponds with the iris type predicted on each row of the test data")
print("Here we will also see how accurate the testing data, by calculating our accuracy.")
print("The way we are going to do this is by comparing our predictions, the labels, also by labels that are only true")
print("SHOW ME THE SCORE/RESULTS!!!!!!!!")
from sklearn.metrics import accuracy_score
print("We are using these to type of classifiers to compare data of the iris data")
print(predictions_from_decision_tree_classifier)
print("What you will see below is the our test data being classified")
print("print predictions from decision tree classifier is: ")

print(accuracy_score(y_test, predictions_from_decision_tree_classifier))

#switching types of classifier
################SWITCH############################################
from sklearn.neighbors import KNeighborsClassifier
#using our training data for KNeighborsClassifier
my_K_nearest_neighbors_classifier = KNeighborsClassifier()
#putting data to work
my_K_nearest_neighbors_classifier.fit(X_train, y_train)

prediction_from_KNeighborsClassifier = my_K_nearest_neighbors_classifier.predict(X_test)
print("We have another classifier that computes a majority vote of each piece of data")
print("The predictions from KNN classifier is: ")

print(accuracy_score(y_test, prediction_from_KNeighborsClassifier))



