# %% [code] {"execution":{"iopub.status.busy":"2021-07-16T23:29:11.484171Z","iopub.execute_input":"2021-07-16T23:29:11.484577Z","iopub.status.idle":"2021-07-16T23:29:11.499504Z","shell.execute_reply.started":"2021-07-16T23:29:11.484541Z","shell.execute_reply":"2021-07-16T23:29:11.498464Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T00:13:01.522501Z","iopub.execute_input":"2021-07-17T00:13:01.523042Z","iopub.status.idle":"2021-07-17T00:13:01.540574Z","shell.execute_reply.started":"2021-07-17T00:13:01.523007Z","shell.execute_reply":"2021-07-17T00:13:01.539596Z"}}
#load the csv dataset, a clean dataset was chosen
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target


#separating to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T00:23:19.150497Z","iopub.execute_input":"2021-07-17T00:23:19.150909Z","iopub.status.idle":"2021-07-17T00:23:19.170079Z","shell.execute_reply.started":"2021-07-17T00:23:19.150875Z","shell.execute_reply":"2021-07-17T00:23:19.168951Z"}}
#time to train the model
from k_nearest_neighbour.KNN import KNN
#from sklearn.linear_model import LogisticRegression
model = KNN(neighbours=6)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T00:23:25.121943Z","iopub.execute_input":"2021-07-17T00:23:25.122275Z","iopub.status.idle":"2021-07-17T00:23:25.132507Z","shell.execute_reply.started":"2021-07-17T00:23:25.122247Z","shell.execute_reply":"2021-07-17T00:23:25.131403Z"}}
#evaluate the results of the predictions
from sklearn.metrics import mean_absolute_error as mae
accuracy = 1 - mae(y_test, y_pred)
print(f"the accuracy is %s %%" %(accuracy*100))