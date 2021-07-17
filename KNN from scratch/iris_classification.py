import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets # builtin sklearn datasets

#load the csv dataset, a clean dataset was chosen
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

#separating to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

#time to train the model
from k_nearest_neighbour.KNN import KNN
model = KNN(neighbours=10)
model.fit(X_train, y_train)

#evaluate the results of the predictions
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error as mae
new_model_accuracy = 1 - mae(y_test, y_pred)

#train with sklearn KNN
from sklearn.neighbors import KNeighborsClassifier as KNN_old
old_model = KNN_old(n_neighbors=10)
old_model.fit(X_train, y_train)

#evaluate the results of sklearn predictions
old_y_pred = old_model.predict(X_test)
old_model_accuracy = 1 - mae(y_test, old_y_pred)
print(f"the accuracy of my KNN algorithm is %s %%" %(new_model_accuracy*100))
print(f"the accuracy of sklearn's KNN algorithm is %s %%" %(old_model_accuracy*100))

