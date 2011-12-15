#!/usr/bin/env python
# encoding: utf-8
"""
Message.py

Created by Andrew Tulloch on 2010-04-27.
Copyright (c) 2010 Andrew Tulloch. All rights reserved.
"""
from utils import *
import math
import Stemmer
#------------------------------------------------------------------------------
        
class Message():
    """Implements the message class.
    Attributes
        subject - subject data
        body - body data
        subject word count - dictionary containing word --> count for subject
        body word count - dictionary containing word --> count for body
        spam - identifier if message is spam or not spam"""
    def __init__(self, filename):
        file = open("./Data/" + filename, 'r')
        data = file.readlines()
        file.close()
        self.subject = data[0][9:].strip()
        self.body = [line.strip() for line in data[2:]][0]
        # Initialise data
        
        self.stem_data()
        self.numeric_filter()
        # Perform the stemmer and numeric methods to further process the data
        
        
        self.subject_word_count = counter(self.subject.split())
        self.body_word_count = counter(self.body.split())
        # Calculate word counts for the data
        
        self.filename = filename
        self.spam = self.spam_class()
        # Message attributes
        
        
    def spam_class(self):   
        """From the filename, classes the message as spam or not spam"""
        if self.filename[:5] == 'spmsg':
            return "Spam"
        else:
            return "Not Spam"

    def stem_data(self):
        """Stems the data, using Porters algorithm"""

        stemmer = Stemmer.Stemmer('english')
        # The stemming object
        
        def stem_string(string):
            """Input a string, returns a string with the 
            words replaced by their stemmed equivalents"""
            stemmed_list = []
            for word in string.split():
                stemmed_word = stemmer.stemWord(word)
                stemmed_list.append(stemmed_word)
            
            stemmed_string = " ".join(stemmed_list)
            return stemmed_string
            
        self.body = stem_string(self.body)
        self.subject = stem_string(self.subject)
            
    def numeric_filter(self):
        """Replaces instances of numbers in a string with 
        a "NUMERIC" placeholder
                    e.g.("112", "22" ---> "NUMERIC")"""
        def num_filter_string(string):
            """Input a string, returns a string with
            strings of digits replaced with "NUMERIC"
            """
            
            filtered_list = []
            for word in string.split():
                if word.isdigit():
                    filtered_list.append("NUMERIC")
                else:
                    filtered_list.append(word)
                    
            filtered_string = " ".join(filtered_list)
            return filtered_string
            
            
        self.body = num_filter_string(self.body)
        self.subject = num_filter_string(self.subject)
        
    
    def tf_idf(self, corpus):
        """Input a corpus (with its list of document frequencies)
        calculates the tf-idf score for the message for every feature"""
        top200list = [(word, count) for count, word in corpus.top200]

        
        if corpus.type == "subject":
            word_count = self.subject_word_count
        else: 
            word_count = self.body_word_count
        
        self.tf_idf_scorelist = []
        # print word_count
        for word, document_frequency in top200list:
            if word not in word_count:
                # If word does not appear in the message, tf-idf == 0
                self.tf_idf_scorelist.append([word, 0])
            else:
                # calculate the tf-idf score for the word, appending the pair (word, score) to the list
                tf_idf_score = word_count[word] * math.log10(corpus.length / float(document_frequency)) + 1.0/200
                self.tf_idf_scorelist.append([word, tf_idf_score])
    
        
        return self.tf_idf_scorelist
        

#------------------------------------------------------------------------------


def testing():
    pass

#------------------------------------------------------------------------------
    
    
if __name__ == '__main__':
    testing()