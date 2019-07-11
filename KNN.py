#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 22:32:05 2019

@author: ruiqianyang
"""

import csv
import random
import math
import operator
 
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):    #4 features
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
#Version2
def loadData(filename,train=[],test=[],split=0.6,n_features):
    with open(filename,'rb') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for i in range(len(dataset)):
            for j in range(n_features): 
                dataset[i][j]=float(dataset[i][j])
            if random.random<split:
                train.append(dataset[i][j])           
            else:
                test.append(dataset[i][j])
 
    
 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
#Version2
def euclidean(x1,x2,length):
    distance=0
    for i in range(length):
        distance+=pow((x1[i]-x2[i]),2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
#Version2
def getNeighbors(train,testInstance,k):
    distance=[]
    n_features=len(testInstance)-1
    for i in range(len(train)):
        dist=eulicdean(train[i],testInstance,n_features)
        distance.append((train[i],dist))
    distance.sort(key=lambda x:x[1])
    neighbors=[]
    for i in range(k):
        neighbors.append(distance[i][0])
    return neighbors        



def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

#Version2
def getResponse(neighbors):
    votes={}
    for x in range(len(neighbors)):
        vote=neighbors[x][-1]
        if vote in votes:
            votes[vote]+=1
        else:
            votes[vote]=1
        sortVotes=sorted(votes.iteritems(),key=operator.itemgetter(1),reverse=True)
        return sortVotes[0][0]
    

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
#Version2
def getAccuracy(test,preds):
    correct=0
    for x in range(len(test)):
        if test[x][-1]==pred[x]:
            correct+=1
    return (correct/float(len(test)))*100
    

	
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('iris.data', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()    

#Version2
def main():
    train,test=[],[]
    split=0.6
    loadData('iris.data',train,test,split)
    preds=[]
    k=3
    for i in range(len(test)):
        neighbors=getNeighbors(train,test[i],k)
        res=getResponse(neighbors)
        preds.append(res)
        print('for' +i+'testInstance: Prediction ='+repr(res)+', actual='+train[i][-1])
        
    accuracy=getAccuracy(test,preds)
    print(accuracy)












