#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 23:02:24 2019

@author: ruiqianyang
"""

import math

#def Euclidean_distance(feat_one, feat_two):#Assuming correct input where the lengths of 2 features are the same
#    squared_distance = 0
#    for i in range(len(feat_one)):
#            squared_distance += (feat_one[i] – feat_two[i])**2
#    return sqrt(squared_distances)
#my:

def fit():
    centriod={}
    for i in range(k):
        centriod[i]=data[i]
        
    for i in range(max_iter):
        classes={}
        for i in range(k):
            classes[i]=[]
        for x in data:
            distance=[np.linalg.norm(i-center) for center in centroid]
            classify=distance.index(min(distance))
            classes[classify].append(x)
        previous=dict(centriod)
        
        for c in classes:
            centroid[c]=np.average(class[c],axis=0)
        
        optimal=True
        for i in range(k):
            origial=previous[i]
            cur=centriods[i]
            if np.sum((cur-original).original*100.0)>self.tolerance:
                optimal=False
               
        if optimal:
            break

#%%  template          
class K_Means:
    def __init__(self, k =3, tolerance = 0.0001, max_iterations = 500):
    self.k = k
    self.tolerance = tolerance
    self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = {}
    #initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        for i in range(self.k):
            self.centroids[i] = data[i]
             
        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []
    
    		#find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
    			  classification = distances.index(min(distances))
    			  self.classes[classification].append(features)
            
            previous = dict(self.centroids)
        
        #average the cluster datapoints to re-calculate the centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis = 0)
        
            isOptimal = True
        
            for centroid in self.centroids:
        
        	       original_centroid = previous[centroid]
        	       curr = self.centroids[centroid]
        
        	       if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
        		       isOptimal = False
        
        	#break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
        	
            if isOptimal:
        		  break
        
    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
        

def main():

#df = pd.read_csv(r"ipl.csv")
df = pd.read_csv(r"CustomerData4.csv",nrows=200)
#df = df[['one', 'two']]
df=df[['MRank','FRank','RRank']]
dataset = df.astype(float).values.tolist()
X = df.values 

#df 
dataset = df.astype(float).values.tolist()
X = df.values #returns a numpy array

km = K_Means(5)

km.fit(X)
#y_kmeansP=km.fit(X)

# Plotting starts here
colors = 10*["r", "g", "c", "b", "k"]
#prediction = pd.DataFrame(km.fit(X), columns=['predictions']).to_csv('prediction.csv')

for centroid in km.centroids:
    plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s = 130, marker = "x")




for classification in km.classes:
    color = colors[classification]
    for features in km.classes[classification]:
        print(classification)
        df['Cluster'] = classification
        plt.scatter(features[0], features[1], color = color,s = 30)


df.to_csv("clusteringfromscrtach.csv")
#plt.show()
print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()          
#%%version 2
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification





            
