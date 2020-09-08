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

def remove_punct(s): 
    #regex = re.compile('[%s]' % re.escape(string.punctuation))
    regex = re.compile('[^a-zA-Z0-9]')
    return regex.sub('', s) 

def fuzzy_match(s1, s2):
    return fuzz.ratio(s1, s2)

def get_start_end_tokenids(): 
    """
    This function DOES NOT WORK and is DEPRECATED
    """
    book_start = {}
    book_end = {}
    detokenizer = TreebankWordDetokenizer()
    with open(BOOKLIST, 'r') as csvfile: 
       reader = csv.DictReader(csvfile)
       for row in reader: 
           book = row['ID if scanned'].strip()
           if book == '': continue
           start = row['Start'].strip().lower()
           end = row['End'].strip().lower()
           book_start[book] = remove_punct(start)
           book_end[book] = remove_punct(end)

    outfile = open(LOGS + 'start_end_token_ids', 'w')

    for f in sorted(os.listdir(TOKENS)):
       toks = []
       with open(TOKENS + f, 'r') as infile: 
           reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
           for row in reader: 
              toks.append(row['normalizedWord'])
       best_start_score = 0
       best_end_score = 0
       start_id = 0
       end_id = 0
       for i, w in enumerate(toks): 
           if i < len(toks)/2: 
              text = remove_punct(detokenizer.detokenize(toks[i:i+10]).lower())
              min_len = min(len(text), len(book_start[f]))
              score = fuzzy_match(book_start[f][:min_len], text[:min_len])  
              if score > best_start_score: 
                  start_id = i
                  best_start_score = score
           if i > len(toks)/2: 
              text = remove_punct(detokenizer.detokenize(toks[i-10:i+1]).lower())
              min_len = min(len(text), len(book_end[f]))
              score = fuzzy_match(book_end[f][-min_len:], text[-min_len:])  
              if score > best_end_score: 
                  end_id = i
                  best_end_score = score
       outfile.write(f + '\t' + str(start_id) + '\t' + str(end_id) + '\n')
       outfile.write('*****' + '\t' + ''.join(toks[start_id:start_id+10]).lower() + '\t' + book_start[f] + '\n')
       outfile.write('*****' + '\t' + ''.join(toks[end_id-10:end_id+1]).lower() + '\t' +  book_end[f] + '\n') 

    outfile.close()

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
    #which_books()
    get_start_end_tokenids()

if __name__ == "__main__":
    main()
