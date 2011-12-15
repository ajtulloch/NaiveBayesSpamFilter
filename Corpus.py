# /usr/bin/env python
# encoding: utf-8
"""
Corpus.py

Created by Andrew Tulloch on 2010-04-27.
Copyright (c) 2010 Andrew Tulloch. All rights reserved.
"""
from utils import *
from Message import *
import Stemmer




#------------------------------------------------------------------------------

class Corpus():
    """Corpus class.  A superclass for the classes SubjectCorpus and BodyCorpus"""
    
    def csv_write(self):
        """Writes a csv file
        201 columns - 200 features and class identifier
        601 rows - header (f1, f2...f200, class) and 600 examples"""
        
        headers = []
        for index in xrange(len(self.top200)):
            # Create the list [f1, f2, ..., f200]
            headers.append("f" + str(index + 1) )
        
        headers.append("Spam Class")
        # Create the list [f1, f2..., f200, Spam Class]
        
        csv_file = []
        csv_file.append(headers)
        
        for message in self.messages:
            # Append the row of tf-idf scores for each feature
            msg_scores = [scores[1] for scores in message.tf_idf_scorelist]
            
            # Append the spam class in the last column
            msg_scores.append(message.spam)
            
            # Append the row to the file
            csv_file.append(msg_scores)
        
        csv_filename = self.type + ".csv"
        writer = csv.writer(open(csv_filename, "wb"))
        for row in csv_file:
            writer.writerow(row)
            # Write the CSV file
        
        
        
    def get_length(self):
        """Find the number of examples in the corpus"""
        self.length = len(self.data)
    
    def tf_idf_scores(self):
        """Calculate tf-idf scores for all messages in the corpus"""
        for message in self.messages:
            message.tf_idf(self)
    
    
    def DF_score(self):
        """Calculate the document frequency score for all words in the corpus"""
        self.DF_counts = {}
        
        for message in self.cleaned_data:
            for word in message.split():
                # Initialise the dictionary
                self.DF_counts[word] = 0
        
        for message in self.cleaned_data:
            word_added_already = []
            for word in message.split():
                if word not in word_added_already:
                    # Avoids double counting a word if it appears twice in a message
                    self.DF_counts[word] += 1
                    word_added_already.append(word)
        
        
        word_list = sorted((value,key) for (key,value) in self.DF_counts.items())
        # Sort our list, in order of least prevalent to most prevalent
        
        word_list.reverse()
        # Reverse this list
        
        self.top200 = word_list[:200]
        # Return the top 200 words
        
        return self.top200
    
    def word_count(self):
        """Counts the number of unique words in the corpus"""
        word_string = []
        for message in self.data:
            for words in message.split():
                word_string.append(words)
        
        word_counts = counter(word_string)
        return word_counts
    
    
    def remove_stop_words(self):
        """Performs the filtering described in the data preprocessing
        section of the report.
            
            Removes stop words
            Removes punctuation
            Stems words
            Filters numeric data
        """
        
        self.cleaned_data = []
        stemmer = Stemmer.Stemmer('english')
        for data in self.data:
            words = data.split()
            stemmed_words = [stemmer.stemWord(word) for word in words \
                            if word not in stop_words \
                             and word not in forbidden_words]
            # Perhaps an overly complex line - returns a list of stemmed words,
            # if the word is not a stop word or forbidden
            
            words = []
            for word in stemmed_words:
                # Filters out our numeric features
                #   e.g "112" --> "NUMERIC"
                if word.isdigit():
                    words.append("NUMERIC")
                else:
                    words.append(word)
            
            clean_data = " ".join(words)
            # Converts list to string
            
            self.cleaned_data.append(clean_data)
        
        return self.cleaned_data
    
    
    def creation(self):
        """A container method, performing the following operations:
            
            filtering out stop words and punctuation
            performing the stemming algorithm
            calculates tf-idf scores
            writes the CSV file
        """
        self.remove_stop_words()
        self.DF_score()
        self.tf_idf_scores()
        self.csv_write()
        print "DONE! - {0} CSV file created".format(self.type)
#------------------------------------------------------------------------------

class SubjectCorpus(Corpus):
    """Subject Corpus
    Message data is the subjects of the individual messages
    """
    def __init__(self, message_list):
        self.messages = message_list
        self.data = [message.subject for message in message_list]
        self.get_length()
        self.type = "subject"



class BodyCorpus(Corpus):
    """Body Corpus
    Message data is the body of the individual messages
    """
    def __init__(self, message_list):
        self.messages = message_list
        self.data = [message.body for message in message_list]
        self.get_length()
        self.type = "body"




#------------------------------------------------------------------------------

def Create_BC_SC_CSV():
    
    file_list = [(file, file[-3:]) for file in os.listdir("./Data")]
    proper_files = [file for file, extension in file_list if extension == "txt"]
    # Filters out files that are not text files
    
    message_list = [Message(file) for file in proper_files]
    # Our list of message objects
    
    SC = SubjectCorpus(message_list)
    SC.creation()
    
    BC = BodyCorpus(message_list)
    BC.creation()




if __name__ == '__main__':
    Create_BC_SC_CSV()
    