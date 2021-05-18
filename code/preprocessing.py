import csv 
import os
from nltk.tokenize.treebank import TreebankWordDetokenizer
import string
import re
from fuzzywuzzy import fuzz

BOOKLIST = '/mnt/data0/lucy/gpt3_bias/data/contemporary_litbank_booklist.csv'
TOKENS = '/mnt/data0/lucy/gpt3_bias/logs/tokens/'
ORIGINAL = '/mnt/data1/corpora/contemporary_litbank/english/originals/'
LOGS = '/mnt/data0/lucy/gpt3_bias/logs/'

def which_books(): 
    '''
    Compares the books I have with the books in the spreadsheet
    '''
    booklist = set()
    with open(BOOKLIST, 'r') as csvfile: 
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['ID if scanned'].strip() != '': 
                booklist.add(row['ID if scanned'].strip() + '.txt')
    processed_books = set()
    processed_books.update(os.listdir(ORIGINAL))
    print("Missing from processed:", len(booklist - processed_books), print(booklist - processed_books))
    print("Missing from spreadsheet:", len(processed_books - booklist), print(processed_books - booklist))

def main(): 
    which_books()

if __name__ == "__main__":
    main()
