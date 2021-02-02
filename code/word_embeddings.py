import os
import csv
import string
from gensim.models.word2vec import PathLineSentences
from gensim.models import Word2Vec
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec


ROOT = '/mnt/data0/lucy/gpt3_bias/'
LOGS = ROOT + 'logs/'

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

    def on_train_begin(self, model): 
        print("Training start")

    def on_train_end(self, model): 
        print("Training end")

def preprocess_text(): 
    '''
    cleans text
    lowercases
    removes numbers and punctuation 
    returns a list of sentences, tokens seperated by white space
    '''
    outpath = LOGS + 'word2vec_train_data/' 

    # read in generated stories
    genpath = LOGS + 'plaintext_stories_0.9_tokens/'
    for f in os.listdir(genpath): 
       new_f = 'GEN_' + f.replace('.tokens', '')
       print(new_f)
       outfile = open(outpath + new_f, 'w') 
       with open(genpath + f, 'r') as infile: 
           reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
           currSentID = -1
           currSent = []
           for row in reader: 
               if row['deprel'] != 'punct' and row['originalWord'] != '@': 
                   word = row['originalWord'].lower()
                   if word in string.punctuation: continue 
                   if row['sentenceID'] != currSentID: 
                       outfile.write(' '.join(currSent) + '\n') 
                       currSent = []
                   currSentID = row['sentenceID']
                   currSent.append(word)
           outfile.write(' '.join(currSent) + '\n') 
       outfile.close()

    # read in books
    bookpath = LOGS + 'tokens/'
    for f in os.listdir(bookpath): 
       new_f = 'ORIG_' + f
       print(new_f)
       outfile = open(outpath + new_f, 'w') 
       with open(bookpath + f, 'r') as infile:
           reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
           currSentID = -1
           currSent = []
           for row in reader: 
               if row['deprel'] != 'punct': 
                   word = row['originalWord'].lower()
                   if word in string.punctuation: continue
                   if row['sentenceID'] != currSentID: 
                       outfile.write(' '.join(currSent) + '\n') 
                       currSent = []
                   currSentID = row['sentenceID']
                   currSent.append(word)
           outfile.write(' '.join(currSent) + '\n')
       outfile.close() 


def train_embeddings(): 
    inpath = LOGS + 'word2vec_train_data/'
    sentences = PathLineSentences(inpath)
    epoch_logger = EpochLogger()
    print("Starting word2vec....")
    model = Word2Vec(sentences, size=100, window=5, min_count=5, 
          workers=multiprocessing.cpu_count(), seed=0, callbacks=[epoch_logger])
    model.save(LOGS + 'fiction_word2vec_model')

def main(): 
    #preprocess_text()
    #train_embeddings()

if __name__ == "__main__":
    main()
