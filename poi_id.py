#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
#Original features list with all features
#features_list = ['poi', 'total_payments', 'salary', 'bonus', 'to_messages',
#                 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
#                 'deferral_payments', 'loan_advances', 'restricted_stock_deferred',
#                 'deferred_income', 'total_stock_value',
#                 'exercised_stock_options', 'expenses', 'other', 'long_term_incentive',
#                 'restricted_stock', 'director_fees', 'shared_receipt_with_poi']

#Features list after SelectKBest - using top 10 and poi
features_list = ['poi', 'total_payments', 'salary', 'bonus', 'exercised_stock_options', 
                 'total_stock_value', 'deferred_income', 'long_term_incentive', 
                 'restricted_stock', 'total_payments',] 

#for Task 4 - removed salary and bonus for tuning, but added back due to no improvment in scores

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#Total had very large numbers, so it was throwing off the calcs, also Total is not a person
del data_dict['TOTAL']
#The Travel Agency in the Park has no useful information and is not a person
del data_dict['THE TRAVEL AGENCY IN THE PARK']
#Eugene Lockart has no information attached in the data set
del data_dict['LOCKHART EUGENE E']

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict 
my_feature_list = features_list

#Remove NaNs
my_dataset = {k: {k2: 0 if v2 == 'NaN' else v2 for k2, v2 in v.items()} \
                    for k, v in my_dataset.items()}

#Based on mini project from Udacity course - ratio of messages sent 
#from a person to a poi/in all from messages
#and messages received from a poi/in all to messages
for k in my_dataset.values():
    k['to_poi_message_fraction'] = 0
    k['from_poi_message_fraction'] = 0
    if float(k['from_messages']) > 0:
        k['to_poi_message_fraction'] = float(k['from_this_person_to_poi'])/float(k['from_messages'])
    if float(k['to_messages']) > 0:
        k['from_poi_message_fraction'] = float(k['from_poi_to_this_person'])/float(k['to_messages'])

#Trying out another feature - the total of salary + bonus
for k in my_dataset.values():
    k['sum_salary_bonus']=0
    if float(k['salary']) or float(k['bonus']) >= 0:
        k['sum_salary_bonus'] = float(k['salary']) + float(k['bonus'])
        
#adding the new features to the features list
my_feature_list.extend(['to_poi_message_fraction', 'sum_salary_bonus'])
#Removed 'from_poi_message_fraction' because it's not in top 10 when running SelectKBest 

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from time import time

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print 'Predictions', pred
print 'Scores', clf.score(features_test, labels_test)
accuracy = clf.score(features_test, labels_test)
print 'Accuracy', accuracy
print "Precision Score", precision_score(labels_test, pred)
print "Recall Score", recall_score(labels_test, pred)

########
#Did not use
#Tryng decision tree from Udacity mini project
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from time import time

#clf2 = tree.DecisionTreeClassifier(min_samples_split=40)
#t0 = time()
#clf2.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#print "Score", clf2.score(features_test, labels_test)

#pred2 = clf2.predict(features_test)
#acc2 = accuracy_score(pred2, labels_test)

#print "Predictions", pred2
#print "Accuracy: ", acc2
#print "Precision score", precision_score(labels_test, pred2)
#print "Recall Score", recall_score(labels_test, pred2)
########

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!

##Code used with GridSearchCV for tuning
#clf_parameters = {}
#clf_gnb = GridSearchCV(clf, clf_parameters)
#clf_gnb.fit(features_train, labels_train)
##accessing the parameter values
#clf_gnb.best_params_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)