import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#resources used:
#dataset: https://www.kaggle.com/datasets/cutedango/league-of-legends-champions
#standardizing preprocessing techniques: https://www.geeksforgeeks.org/python/how-to-standardize-data-in-a-pandas-dataframe/ 
#PCA preprocessing techniques: https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/
#k-means clustering technique: https://www.w3schools.com/python/python_ml_k-means.asp

#The goal of this model is to cluster League of Legends champions based on their in-game attributes/stats 
#in order to classify them under their respective role according to the LoL role classification system.
# (e.g., Controller, Fighter, Mage, Marksman, Slayer, Tank, Specialist).

#read data from csv file
df = pd.read_csv('LoL_champions.csv')

#drop rows with missing values
X = df.dropna()

#drop non-numeric columns, Names is not needed for clustering, Tags and Role are categorical variables,
#Resource type is dropped for simplicity, 

X = X[X['Mana per lvl'] > 0] #champions that don't use mana are dropped for simplicity

names = X['Name'] #store champion names for plotting later

X = X.drop(['Name', 'Tags', 'Role', 'Resourse type'], axis=1)

X = X.replace({'Melee': 0, 'Ranged': 1}) #convert range values to numeric values (0 = Melee, 1 = Ranged)

#standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#use PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#k-means clustering
#here i opt for 6 clusters, as there are 6 main roles in LoL, omitting the Specialist role,
#as it is a more niche role that doesn't fit well into the main 6 categories
kmeans = KMeans(n_clusters=6)
kmeans.fit(X_pca)

pred = kmeans.labels_

#plot the points and their respective clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pred)

#add champion names
for i, name in enumerate(names):
    plt.text(X_pca[i, 0] + 0.02,
             X_pca[i, 1] + 0.02,
             name,
             fontsize=6)
    
plt.show()