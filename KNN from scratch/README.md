# KNN

KNN is a custom made Python library for dealing with K nearest neighbours algorithm for calculation.
KNN says show me your neighbour and I''ll tell you who you are, the algortihm was built based on the equation below

![KNN algorithm equations](https://www.saedsayad.com/images/KNN_similarity.png)

The Pseudocode for Predicting the class value for new data:
- Calculate distance(X1, X2)
where X1= new data point, X2= training data, distance as per your chosen distance metric.
- Sort these distances in increasing order with corresponding train data.
- From this sorted list, select the top ‘K’ rows.
- Find the most common class from these chosen ‘K’ rows. This will be your predicted class.


## Installation

clone the repo from https://github.com/tobaadesugba/Machine_Learning/tree/projects-on-my-dell/KNN%20from%20scratch/k_nearest_neighbour and save it with your python libraries.

A sample code (iris classification) is given in https://github.com/tobaadesugba/Machine_Learning/tree/projects-on-my-dell/KNN%20from%20scratch to test the performance of the algorithm in relation to sklearn's


## Usage

```python
from k_nearest_neighbour import KNN

# initializes the model
model_name = KNN(neighbours=3)

# trains the model
model_name.fit(X_train, y_train)

# makes inference, returns an array of predictions
model_name.predict(X_test)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

##Author
[Adesugba Oluwatoba](https://linkedin.com/in/tobaadesugba/)

## License
[MIT](https://choosealicense.com/licenses/mit/)