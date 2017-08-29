from __future__ import division
import numpy as np
from math import sqrt
import random
import sys
import matplotlib.pyplot as plt

# defining the attributes and classes of the data
attributes = {'buying': ['vhigh', 'high', 'med', 'low'],
              'maint': ['vhigh', 'high', 'med', 'low'],
              'doors': ['2', '3', '4', '5more'],
              'persons': ['2', '4', 'more'],
              'lug_boot': ['small', 'med', 'big'],
              'safety': ['low', 'med', 'high']}
attributes_list = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
classes = ['unacc', 'acc', 'good', 'vgood']

# loading the data
in_file = raw_input("Enter the input data File Name: ")
K = raw_input("Enter the number of clusters you want: ")
# testing if the file is not available.
try:
    cardata = np.genfromtxt(in_file, dtype=None, delimiter=',')
except IOError:
    print "File", in_file ,"not found"
    sys.exit(0)
# testing if user input is integer and assigning the value to the variable K.
try:
   K = int(K)
except ValueError:
   print("That's not an integer!")
   sys.exit(0)

# this is the raw categorical data used for error calculation at the end.
formatted_data = list([(i[0:6], i[6]) for i in cardata])
# ignoring the class for the data that will be converted to numerical values.
cardata = [i[0:6] for i in cardata]
# transform the categorical data into numerical data
for i in cardata:
    for j in range(len(attributes_list)):
        i[j] = attributes[attributes_list[j]].index(i[j]) + 1
cardata = [i.astype(np.integer) for i in cardata]

def distance(m, p):
    ''' distance is a function that return the euclidean distance between two points'''
    return sqrt(sum([(m[i]-p[i])**2 for i in range(len(p))]))

def avg(L):
    ''' avg is a function that takes as an input a cluster of points and returns the average of these points
    (the new centroid)'''
    sums = [sum(i) for i in zip(*L)]
    count = len(L)
    return [j/count for j in sums]

def random_init(data, K):
    '''random_init is a function that takes data and number of clusters (K) and return K random points of
    the data as the centroids of the clusters '''
    indexes = []
    while len(indexes) < K:
        r = random.randint(1, len(data)-1)
        if r not in indexes:
            indexes.append(r)
    return [list(data[i]) for i in indexes]

def kmeans(data, K):
    ''' the main function that implements the k means algorithm. it takes the data and number of clusters
    K and returns the centroids and the clusters assignment list for the points'''
    # initiate K centrooids.
    centroids = random_init(data, K)
    # infinite loop that only breaks when the algorithm converges and the function returns the final centroids and clusters.
    while True:
        clusters = []
        # cluster assignment for each point in the data based on the distance to centroids.
        for i in data:
            clusters.append(min(range(len(centroids)), key=lambda x:distance(centroids[x], i)))
        # movement of the centroids based on the average of the points assigned to it.
        new_centroids = []
        for i in range(len(centroids)):
            centroid = avg([data[j] for j in range(len(clusters)) if clusters[j] == i])
            new_centroids.append(centroid)
        # if there is no movement of the centroids then the algorithm has converged and the function returns.
        # if not we repeat.
        if new_centroids == centroids:
            return new_centroids, clusters
        else:
            centroids = new_centroids

def error(data, clusters, K, classified_data, classes):
    ''' error is a function that takes as an input the numerical data, the clustes assignment list, the number
    of clusters K, the raw categorical data and the list of classes and return the count of errors in classification, the clusters and
    the ratio of error'''
    real_clusters = []
    for i in range(K):
        cluster =  [classified_data[j] for j in range(len(clusters)) if clusters[j] == i]
        counts = []
        for c in classes:
            count = len([i for i in cluster if i[1] == c])
            counts.append(count)
            cluster_class = classes[counts.index(max(counts))]
        real_clusters.append((cluster_class, cluster))

    error = 0
    for i in real_clusters:
        for j in i[1]:
            if j[1] != i[0]:
                error+=1
    return error, real_clusters, error/len(data)*100

# run the kmeans function to get centroids and cluster assignment list.
centroids, clusters = kmeans(cardata, K)
# use the previous output to calculate the error.
e, real_clustes, ratio = error(cardata, clusters, K, formatted_data, classes)
# print the results.
print 'With clustering the data into', str(K), 'clusters, we got', str(e), 'errors with classification accuracy of', str(round(100-ratio, 2)), '%'

# this is an optional part of the code. you can choose the lower and upper limit of number of clusters to use for
# clustering and get a plot of K agianst error.

# UNCOMMENT THE FOLLOWING LINES FOR IT TO WORK
# YOU NEED TO HAVE 'matplotlib' FOR IT TO WORK!
#
errors = []
lower_limit = 1
upper_limit = 12
clusters_range = range(lower_limit, upper_limit)
for i in clusters_range:
    centroids, clusters = kmeans(cardata, i)
    e, real_clustes, ratio = error(cardata, clusters, i, formatted_data, classes)
    errors.append(e)


plt.scatter(clusters_range, errors)
plt.xlabel('Number of clusters')
plt.ylabel('Error in calssification')
plt.title('Errors in calssification')
plt.show()
