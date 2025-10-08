from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data # features in the flowers
y = iris.target # target variable (flower species in this case 0 = setosa , 1 = veriscolor , 2 = virgincia)

'''
split dataset, why ? to prevent overfitting. Image you are studying for an exam, your teach gives you 100 questions to practice.
Those same 100 questions appear on the exam. Did you actaully learn the material or just memorize the answers?
This is called overfitting. To prevent this we split the dataset into training and testing sets
'''
from sklearn.model_selection import train_test_split

'''
X_train = attribute features
X_test = attribute features for testing
Y_train = target variables (lables of flowers, rose, tulip etc)
Y_test = target variables for testing
test_size = 30% testing, 70% training
random_state = 42 (a random number, can be any number, it is used to ensure that the split is reproducible and also random)
'''
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.3 , random_state = 99) # 42 is lucky


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

'''
Scaling Data: Many ML algorithms needs to figure out the distance between two items to see how similiar they are. 
Some attributes have high ranges and differences, while others have low ranges and differences so when we caluclate the distance, 
some values that are bigger would dominiate, So we need to scale them accordingly so that everyone has a share vote.

X_train_scaled use fit transform to learn the scaling parameters (mean and standard deviation) from the training data and then apply the scaling to the training data itself.
X_test_scaled use this new transform to apply the same scaling parameters (mean and standard deviation) learned from the training data to the test data.
'''
# Laymens Terms: Where scaling the data to give every attribute an equal vote on determining the output
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVC

'''
SVM tried to draw boundries in xD space, in this case we have 4 targets to 4D space. 
'''
svm_model = SVC() # import a model for training, in this case its SVM

'''
This is where the learning happens:
"Hmm, when petal width is small (< 0.5), it's usually species 0 (Setosa)"
"When petal length is large (> 5) AND sepal length is large (> 6), it's usually species 2 (Virginica)"

How does it learn? We teach it based on the 70% of training data we have.
We take 70% of the attributes (X) and 70% of the features (Y) and we say, 

"Hey model, this flower has measurements [5.1, 3.5, 1.4, 0.2] → it's a Setosa (0)"
"This flower has measurements [7.0, 3.2, 4.7, 1.4] → it's a Versicolor (1)"
"This flower has measurements [6.3, 3.3, 6.0, 2.5] → it's a Virginica (2)"
'''
svm_model.fit(X_train_scaled, Y_train) # fit the model to the training data

from sklearn.metrics import accuracy_score

y_pred = svm_model.predict(X_test_scaled) # predict the output for the test data
accuracy = accuracy_score(Y_test, y_pred) # compare the predicted output with the actual output

print(f"Model Accuracy for SVM: {accuracy * 100:.2f}%")

'''
Cross Validation is a way to test how good your model is. 
Remeber when we did this : X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)
We split the data into training and testing sets. But what if we did this multiple times with different random splits?
In other words, we test on multiple and different types of splits, not just one.
"My model scored 100 - but maybe I just got lucky with a good split"
"My model got 100 , 96, 100, 98, 95 - I can be more confident that my model is good"
'''
from sklearn.model_selection import cross_val_score
cross_validation_scores = cross_val_score(SVC(), X_train_scaled, Y_train, cv = 5)
print(f"Cross Validation Scores for SVM: {cross_validation_scores.mean() * 100:.2f}%")

'''
GridSearchCV is a way to find the best parameters for your model.
Allows you to adjust the knobs on your model (parametric learning) to find the best combination of parameters that gives you the highest accuracy.

Kernal : How we draw boundries, i.e linear or curved
C: Controls the trade off between smooth decision boundary and classifying training points correctly. A high C may cause overfitting for stricter boundries. 
But sometimes we can allow points to be wrongly classified to get a smoother boundry. Because some points might outliers and to always try to fit them will result in 
overfitting and other points to be wrongly classified because the boundry is so tight.

What is a good balance than of C ? The short answer is let GridSearchCV figure it out for you.
There's no universal "perfect" C value. It depends on:

How noisy your data is
How much data you have
The complexity of the problem
What you care about more (precision vs. generalization)

Approach tried a range of values [0.1, 1, 10, 100] and see which one works best for your specific dataset and problem.
Than let GridSearchCV figure it out for you.
'''
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']} # rbf = radial basis function, poly = polynomial
grid_search_cv = GridSearchCV(SVC(), param_grid, cv = 5)
grid_search_cv.fit(X_train_scaled, Y_train)
best_model = grid_search_cv.best_estimator_
best_params = grid_search_cv.best_params_

print(f"Best Parameters from GridSearchCV: {best_params}")

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()), # step 1: scale the data
    # ('svm', SVC(C=1, kernel='rbf')) 
    ('svm', SVC(C = best_params['C'], kernel = best_params['kernel'])) # step 2: train the model with the best parameters found by GridSearchCV

])

pipeline.fit(X_train, Y_train) # fit the pipeline to the training data
y_pred_pipeline = pipeline.predict(X_test) # predict the output for the test data
accuracy_pipeline = accuracy_score(Y_test, y_pred_pipeline) # compare the predicted output with the actual output
print(f"Model Accuracy with Pipeline: {accuracy_pipeline * 100:.2f}%")



'''
CONLUSION: 

Loading and understanding data - You know what features and targets are
Train/test split - You understand why we split data to prevent overfitting
Scaling - You grasp why features need equal "votes"
Training models - You understand .fit() is where learning happens
Making predictions - Using the trained model on new data
Cross-validation - Testing on multiple splits for reliability
Hyperparameter tuning - Using GridSearchCV to find optimal C and kernel
Pipelines - Streamlining your workflow AND using the best parameters dynamically


Superivise Learning Workflow: 

1. Raw Data (Load Data)
2. Split Data (Train/Test Split)
3. Scale Features (StandardScaler)
4. Train Model
5. Predict on New Data (Test Set)
6. Evaluate Performance (Accuracy, Cross-Validation)
'''
