import numpy as np
from collections import Counter

def get_eucl_dist(x1, x2):
    #calculate the eucleadian distance between neighbours
    displacement = np.sqrt(np.sum((x1-x2)**2))
    return displacement

class KNN:
    def __init__(self, neighbours=3):
        #copy the local neighbours to the global max_distance
        self.max_distance = neighbours

    def fit(self, X, y):
        #copy local x & y to global x & y
        self.X = X
        self.y = y

    def predict(self, X):
        #get the predicted labels from the get_inference function
        predicted_labels = [self.get_inference(feature) for feature in X]
        return predicted_labels

    def get_inference(self, feature):
        #compute distances
        distances = [get_eucl_dist(feature, feature_x) for feature_x in self.X]

        #sort the distances and
        #store the indices, limited to max_distance, in k_sorted
        k_sorted = np.argsort(distances)[:self.max_distance]

        #store the nearest y labels in k_nearest_labels
        k_nearest_labels = [self.y[index] for index in k_sorted]

        #store the most common label in major_label
        nearest_label = Counter(k_nearest_labels)
        major_label = nearest_label.most_common(1)

        return major_label[0][0]