import csv
import os
import math
import random

#------------------------------------------------------------------------------

stop_words = []

# Punctuation marks
forbidden_words = [",", "+", "&", "-", "_", ".", ")", "(", ":", "=", "/", "'", "\"", "*"]

# Import our stop words from our file, and place them in a list
stop_file = csv.reader((open('stop_words.txt')))
stop_words.extend(stop_file)
stop_words = stop_words[0]
    

def counter(word_list):
    """Given a list of words, return a dictionary
    associating each word with the number of times it occurs"""
    counts = {}
    for word in word_list:
        # Intialise dictionary
        counts[word] = 0
    for word in word_list:
        # Calculate word counts
        counts[word] += 1
    
    return counts

    
def norm_dist(value, mean, sd):
    """Calculates the probability density of a normal distribution
    given the mean and standard deviation of the distribution."""
    if sd == 0.0:
        # If SD = 0, return a small non-zero number
        # as discussed in the report.
        return 0.05
    else:
        # PDF for normal distribution
        result = math.exp(-float((value - mean)**2) / (2.0*(sd**2))) * 1.0/(sd * math.sqrt(2 * math.pi))
    return result

def mean(data):
    if data == []:
        return 0.0
    else:
        """Calculates the mean of a list"""
        sum = 0.0
        for item in data:
            sum += item
        return float(sum)/len(data)

def sd(data):
    if data == []:
        return 0.0
    else:
        "Calculates the standard deviation of a list"
        data_mean = mean(data)
        sums = 0.0
        for item in data:
            sums += (item-data_mean)**2
        return math.sqrt(float(sums)/(len(data)-1))

#------------------------------------------------------------------------------

