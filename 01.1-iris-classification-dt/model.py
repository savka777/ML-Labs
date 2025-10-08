# plot the tree
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


# Load data
iris = load_iris()

# select attributes and features
x = iris.data
y = iris.target

# split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.2, random_state = 22)

# create a decision tree model
dt_model = DecisionTreeClassifier()

# fit the data
dt_model.fit(X_train, Y_train)

# predict
y_pred = dt_model.predict(X_test)

# measure accuracy
y_predict = accuracy_score(Y_test, y_pred)

# make first predication
print(f"Model Accuracy: {y_predict*100:.2f}%")

# Cross validate across different scenerios
cross_validate_score = cross_val_score(dt_model, X_train, Y_train, cv=5)

# print cross validation score (average)
print(f"Cross Validation Score: {cross_validate_score.mean()*100:.2f}%")

# plot the tree
plt.figure(figsize=(20, 10))

tree.plot_tree(
    dt_model,
    feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
    class_names=['Setosa', 'Versicolor', 'Virginica'],
    filled=True,    
    rounded=True,    
    fontsize=12     
)

plt.title("Decision Tree for Iris Classification", fontsize=16)
# plt.show()


'''
Summerary:
# 93.33% - SVM
# 95.56% - Decision Tree
'''