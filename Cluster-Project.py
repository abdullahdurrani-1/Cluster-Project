import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler


# we have data of wine we will perform unsuperviesed learning on it
# now checkthe basics info realted to the datset first we willimport the dataset
df = pd.read_csv("wine-clustering.csv")
df.head()
df.isnull().sum()
df.info()
df.describe()
df.shape
df.columns
# plot the distribution of the data
plt.scatter(df["Alcohol"], df["Color_Intensity"])
plt.xlabel("Alcohol")
plt.ylabel(" Color_Intensity")
plt.title("Alcohol vs Color INtensity Distribution")
plt.legend()
plt.show()


# now we will perform the clustering on the data via unsupervised learning 
# we will use Kmeans clustring algoritham firstly, lets go 
# we will use the elbow method to find the optimal number of clusters
# and also we will do teh standrd scaling of the data 
scalar = StandardScaler()
df_scaled = scalar.fit_transform(df)
print(df_scaled.shape)
# Kmeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clsuser = KMeans.fit_predict(kmeans,df_scaled)
# now we will plot the clusters 
plt.scatter(df_scaled[:, 0], df_scaled[:,1] , cmap="viridis")
plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.title("KMeans Clustering of Wine Data")
plt.colorbar(label="Color Intensity")
plt.legend()
plt.show()
# now use elbow method to find the best number of clustrs 
wscs= []
for i in range (1,11) :
    kmeans = KMeans(n_clusters = i, random_state=42)
    kmeans.fit(df_scaled)
    wscs.append(kmeans.inertia_)
#plot the elbow method   
plt.plot(range(1, 11), wscs,marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

# now we will use the cluster algoritham to predict the hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage 
#create a linkage matrix
linkage_matrix = linkage(df_scaled, method="single")
# plot the dendrogram
dendrogram(linkage_matrix, labels=df.index, leaf_rotation=90)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
# now apply tsne model for best vizulas
from sklearn.manifold import TSNE 
m = TSNE(n_components=2, random_state=42)
tsne_data = m.fit_transform(df_scaled)
# plot the tsne data
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=clsuser, palette="viridis")
plt.title("t-SNE Visualization of Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster")
plt.show()
# now we will save the model for future use
import joblib
# save the kmeans model
joblib.dump(kmeans, "kmeans_model.pkl")
