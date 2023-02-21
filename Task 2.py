#Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans as km

#Loading Data, removing useless column
data = pd.read_csv('iris.csv')
print(data)
data.drop('Id', axis=1, inplace=True)
print(data)
print('\n')

#Confirming there are no null values
print(data.isnull().sum())
print('\n')
#Listing all unique species classifications
print(data['Species'].unique())

#Visualization reveals that choosing K is not straightforward
plot.scatter(data['SepalLengthCm'], data['SepalWidthCm'], data['PetalLengthCm'], data['PetalWidthCm'])
plot.show()

#Trying different values of k (1-10) for clustering
x = data.iloc[:,0:4].values
wcss = []
for k in range(1,11):
    kmeans = km(n_clusters = k)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
#Elbow plot reveals 3 is the optimal cluster count
plot.plot(range(1,11), wcss)
plot.title('Elbow Plot')
plot.xlabel('Number of clusters')
plot.ylabel('Within cluster sum of squares')
plot.show()

#Clustering with k = 3
kmeans = km(n_clusters = 3)
kmeans_y = kmeans.fit_predict(x)

# Visualising the clusters based on the last 2 columns
plot.scatter(x[kmeans_y == 0, 2], x[kmeans_y == 0, 3], s = 35, c = 'r', label = 'Iris setosa')
plot.scatter(x[kmeans_y == 1, 2], x[kmeans_y == 1, 3], s = 35, c = 'b', label = 'Iris versicolour')
plot.scatter(x[kmeans_y == 2, 2], x[kmeans_y == 2, 3], s = 35, c = 'g', label = 'Iris virginica')

# Plotting the centroids of the clusters
plot.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], s = 35, c = 'black', label = 'Centroid')
plot.legend()