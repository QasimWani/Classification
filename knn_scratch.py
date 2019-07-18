import numpy as np
import pandas as pd
from collections import Counter

def custom_KNN(point, X_train, y_train, n):
    """
Point refers to the trip (list) with X_0 as time start and Y_0 as time end.
X_train : ALL filtered out trips (based on location and dates)
Y_train : Labels : {0, +1} i.e. going / not-going < this refers to the riders previous trips, given the data from Trips.JS >
n : refers to the nearest neigbors to filter for. By default, n = 3.

The purpose of this function is to find K-n-n to the rider's requested alerts. 
Note : All times are expressed in timestamps (i.e. seconds since Jan. 1970)
Example : 

point = [1563430388, 1563470388]
X_train = [[1553670388, 156321388], [1558928302,15573892], ... , [156347103,156347314]]
y_train = [0, 1, ... , 0] 
n = 3

print(custom_KNN(point, x_train, y_train, n)[:3])
^^ The above result tells us that the values (list) is what the driver will most likely opt for.
    """
    diff = point - X_train
    dists = np.apply_along_axis(np.linalg.norm, 1, diff)
    temp = pd.DataFrame(dists)
    temp.index = y_train

    temp = temp.sort_values(by=[0])
    sorted_labels = temp.index[:n]
    mc = Counter(sorted_labels).most_common(1)[0][0]
    return mc
