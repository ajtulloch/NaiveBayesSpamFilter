#!/usr/bin/env python
# encoding: utf-8
"""
NaiveBayesClassifier.py

Created by Andrew John Tulloch on 2010-05-15.
Copyright (c) 2010 Andrew Tulloch. All rights reserved.
"""

import sys
import os
import csv
from utils import *
import math
import random


#------------------------------------------------------------------------------

class NaiveBayesClassifier():
    """Naive Bayes Classifier class
    Implements the methods:
        CSV Read    - reads a data file
        Train       - Trains on a set of messages
        Feature_class_mean_sd - Calculates mean and sd
                    for FEATURE when CLASS = SPAM CLASS
        Classify    - Classifies a message
        P_spam_not_spam - Calculates probabilities a message
                        is spam or not spam
        Classification_test - tests if a message is correctly
                        classified
        Stratification_test - Performs 10-fold cross validation"""
        
    def __init__(self, corpus):
        self.type = corpus # Type of corpus - body or subject
        self.corpus_header, self.corpus_data = self.csv_read(corpus)
        self.corpus_data = self.cosine_normalisation()
        # Reads the corpus data
    
    def csv_read(self, corpus):
        """Reads a CSV file.  Outputs two lists: 
            corpus_float_data   - a list of messages
            corpus_header       - a list of headers"""
        corpus_data = []
        corpus_file = self.type + ".csv" # e.g. subject.csv
        reader = csv.reader(open(corpus_file)) 
        for row in reader:
            corpus_data.append(row) 
            # Scans through the rows, appending to the file
        corpus_header = corpus_data[:1] # Header data "f1, f2..."
        corpus_data = corpus_data[1:]   # Message data with TF-IDF scores
        corpus_float_data = [] 
        for row in corpus_data:
            # Converts strings to floats
            float_row = [float(i) for i in row[:-1]]
            float_row.append(row[-1])
            corpus_float_data.append(float_row)
        return corpus_header, corpus_float_data



    def cosine_normalisation(self):
        """Performs the cosine normalisation of data"""
        self.normalised_data = []
        for message in self.corpus_data:
            normalised_scores = []
            tf_idf_scores = message[:-1]
            normalisation_factor = math.sqrt(sum([i**2 for i in tf_idf_scores])) 
            # Calculate \sum_{k} tf-idf(t_k, d_j)^2
            if normalisation_factor == 0:
                # Prevents dividing by zero
                self.normalised_data.append(message)
            else:       
                for score in tf_idf_scores:
                    normalised_scores.append(score/float(normalisation_factor))
                normalised_scores.append(message[-1])
                self.normalised_data.append(normalised_scores)
        return self.normalised_data
        
        
    def train(self, training_set):
        """Trains the classifier by calculating the prior normal distribution
        parameters for the feature sets and TRUE/FALSE"""
        training_messages = [self.corpus_data[i] for i in training_set] 
        # The set of training messages
        
        self.mean_sd_data = {}
        # Empty dictionary to hold mean and sd data
        
        for feature in range(200):
            self.mean_sd_data[feature] = {"Not Spam":[0, 0], "Spam":[0, 0]}
            for spam_class in ["Not Spam", "Spam"]:
                self.mean_sd_data[feature][spam_class] = []
            # Initialise the dictionary
            
        for feature in range(200):
            for spam_class in ["Not Spam", "Spam"]:
                # Fill the dictionary with values calculated from the feature_class_mean_sd method
                self.mean_sd_data[feature][spam_class] = self.feature_class_mean_sd(spam_class, feature, training_messages)
                
                
        # Calculate the a-priori spam and not-spam probabilities
        spam_count = 0
        for message in training_messages:
            if message[-1] == "Spam":
                spam_count += 1
        
        self.mean_sd_data["Spam"] = spam_count / float(len(training_set))
        self.mean_sd_data["Not Spam"] = 1 - (spam_count / float(len(training_set)))

        
        
    def feature_class_mean_sd(self, spam_class, feature, training_messages):
        """Calculates the mean and standard deviations for:
            FEATURE when CLASS = SPAM CLASS"""
        feature_list = []
        for message in training_messages:
            # Loop through all messages
            if spam_class == message[-1]:
                # If our message is in the right class
                feature_list.append(message[feature])
                # Take of the corresponding feature TF-IDF score
        # Return the summary statistics of the relevant feature / class
        return [mean(feature_list), sd(feature_list)]
        
        
    def classify(self, message):
        """Classify a message as spam or not spam"""
        p_spam = self.bayes_probability(message, "Spam") 
        # Probability that message is spam
        p_not_spam = self.bayes_probability(message, "Not Spam")
        # Probability that message is not spam
        # print p_spam, p_not_spam
        
        if p_spam > p_not_spam:
            return "Spam"
            # Message is not spam
        else:
            return "Not Spam"
            
            # Message is spam
            
            
    def bayes_probability(self, message, spam_class):
        """Probability that a message is or is not spam"""
            
        a_priori_class_probability = self.mean_sd_data[spam_class]
        # Probability that a single message is spam or not spam i.e. P(spam_id)
        # print "Commencing Bayes Probability on Message 0"
        # print "A priori Class Probability of {0} class is {1}".format(spam_class, a_priori_class_probability)
        class_bayes_probability = a_priori_class_probability


        body_best_features = [ 6,8,11,34,35,45,48,117,124,134,141,174] 
        # Feature selection from WEKA

        subject_best_features = range(1,200)
        
        
        if self.type == "body":
            """Converts the features f1, f2, ...fn into Python list indices"""
            best_features = map(lambda x :x -1, body_best_features)
        else:
            best_features = map(lambda x :x - 1, subject_best_features)
        for feature in best_features:
            # For all features
        
            message_tf_idf_score = message[feature]
            # Message tf_idf value
        
            tf_idf_mean = self.mean_sd_data[feature][spam_class][0]
            tf_idf_sd = self.mean_sd_data[feature][spam_class][1]
            # Get the parameters of the probability distribution governing this class
            
            prob_feature_given_class = norm_dist(message_tf_idf_score, tf_idf_mean, tf_idf_sd)
            # Find the probabilty P(tf-idf_feature = score | msg_class = class)
            class_bayes_probability = class_bayes_probability * prob_feature_given_class
            # Multiply together to obtain total probabilitiy
            # as per the Naive Bayes independence assumption

        return class_bayes_probability # Our probability that a message is spam or not spam
        
    def classification_test(self, message):
        """Tests if a message is correctly classified"""
        if self.classify(message) == message[-1]:
            return True
        else:
            return False
    
    def stratification_test(self):
        """Performs 10-fold stratified cross validation"""
        already_tested = []
        test_set  = []
        for i in range(10):
            """Create the set of 10 sixty element random bins"""
            sample = random.sample([i for i in range(600) if i not in already_tested], 60)
            already_tested.extend(sample)
            test_set.append(sample)
            
            
        results = []
        for validation_data in test_set:
            """Create the training set (540 elements) and the validation data (60 elements)"""
            training_sets = [training_set for training_set in test_set if training_set is not validation_data]
            training_data = []
            for training_set in training_sets:
                training_data.extend(training_set)
                
            self.train(training_data)
            # Train the probabilities of the Bayes Filter
            
            count = 0
            for index in validation_data:
                """Calculate the percentage of successful classifications"""
                if self.classification_test(self.corpus_data[index]):
                    count += 1
            results.append(float(count)/len(validation_data))
        return results  
            
#------------------------------------------------------------------------------


def print_results(results):
    """Formats results and prints them, along with summary statistic"""
    for result, index in zip(results, range(len(results))):
        print "Stratification Set {0} \t {1:.1f}% classified correctly.".format(index+1, result*100)
    print "##"*30
    print "\n\tOverall Accuracy is {0:.1f}%".format(mean(results) * 100)

if __name__ == '__main__':
    import random
    random.seed(18)
    #  Sets the seed, for result reproducibility
    test = NaiveBayesClassifier("subject")
    print "\tTesting Subject Corpus"
    print "##"*30
    results = test.stratification_test()
    print_results(results)
    print 
    print "\tTesting Body Corpus"
    print "##"*30
    test = NaiveBayesClassifier("body")
    results = test.stratification_test()
    print_results(results)
