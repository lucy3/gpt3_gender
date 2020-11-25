"""
This script should 
1) find prompt in original book
2) get average length of generations for prompt
3) grab up to that number of tokens in original book
4) write that part of book to file 
"""
import os
from collections import defaultdict
import stanza
import numpy as np
import json
import string
import csv
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.metrics.distance import edit_distance
import edlib

ROOT = '/mnt/data0/lucy/gpt3_bias/'
PROMPTS = ROOT + 'logs/original_prompts/'
TOKENS = ROOT + '/logs/tokens/'
STORIES = ROOT + 'logs/generated_0.9/' 
LOGS = ROOT + 'logs/'

def get_generation_len(): 
    '''
    Get generated story lengths 
    '''
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    input2lengths = defaultdict(list)
    for f in os.listdir(STORIES): 
        with open(STORIES + f, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                for j in range(len(d['choices'])): 
                    text = d['choices'][j]['text']
                    doc = nlp(text)
                    input2lengths[d['input']].append(doc.num_tokens)
    input2len = defaultdict(int)
    for inp in input2lengths: 
        input2len[inp] = np.mean(input2lengths[inp])
    with open(LOGS + 'generated_story_len_0.9.json', 'w') as outfile: 
        json.dump(input2len, outfile)

def clean_words(w): 
    w = w.replace('--', ' ').replace('—', ' ')
    w = w.translate(str.maketrans('', '', string.punctuation))
    w = w.replace(' ', '')
    return w
 
def standardize_prompts(): 
    '''
    The input file of prompts doesn't necessarily
    match the strings in the output by gpt-3 due to punctuation
    differences, especially quotation marks, in 42 prompts. 
    '''
    with open(LOGS + 'generated_story_len_0.9.json', 'r') as infile: 
        input2len = json.load(infile)
    input2len_nopunct = {}
    for k in input2len: 
        nopunct = clean_words(k)
        input2len_nopunct[nopunct] = input2len[k]
    prompts = set()
    prompts_nopunct = set()
    book2prompt = defaultdict(list)
    for f in os.listdir(PROMPTS): 
        with open(PROMPTS + f, 'r') as infile:
            for line in infile: 
                p = line.split('\t')[2].strip()
                prompts.add(p)
                pnp = clean_words(p)
                prompts_nopunct.add(pnp)
                book2prompt[f].append(pnp)
    assert len(prompts_nopunct - set(input2len_nopunct.keys())) == 0
    assert len(set(input2len_nopunct.keys()) - prompts_nopunct) == 0
    return input2len_nopunct, book2prompt

def get_book_excerpts(): 
    input2len, book2prompt = standardize_prompts() # punctuationless input to length
    detokenizer = TreebankWordDetokenizer()
    for f in book2prompt: 
        span = defaultdict(tuple) # prompt : (start token ID, end token ID)
        with open(TOKENS + f, 'r') as infile:
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            curr_sent = []
            start_tokenID = None
            curr_sentID = None
            for row in reader: 
                if row['sentenceID'] != curr_sentID and curr_sentID is not None: 
                    sent = ''.join(curr_sent) 
                    sent = clean_words(sent)
                    if sent in book2prompt[f]: 
                        span[sent] = (start_tokenID, start_tokenID + int(input2len[sent]))
                    '''
                    for prompt in book2prompt[f]: 
                        ed = edlib.align(sent, prompt)['editDistance'] 
                        if ed < best_matches[prompt][1]: 
                            best_matches[prompt] = (sent, ed)
                    '''
                    curr_sent = []
                    curr_sentID = row['sentenceID']
                    start_tokenID = int(row['tokenId'])
                if curr_sentID is None: 
                    curr_sentID = row['sentenceID']
                    start_tokenID = int(row['tokenId'])
                curr_sent.append(row['normalizedWord'].replace('’', '\''))
        missing = set(book2prompt[f]) - set(span.keys())
        print(missing)
        maybe_found = set()
        with open(TOKENS + f, 'r') as infile:
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            curr_sent = []
            start_tokenID = None
            curr_sentID = None
            for row in reader: 
                if row['sentenceID'] != curr_sentID and curr_sentID is not None: 
                    sent = ''.join(curr_sent) 
                    sent = clean_words(sent)
                    for prompt in missing: 
                        if prompt in sent: 
                            maybe_found.add(prompt)
                    curr_sent = []
                    curr_sentID = row['sentenceID']
                    start_tokenID = int(row['tokenId'])
                if curr_sentID is None: 
                    curr_sentID = row['sentenceID']
                    start_tokenID = int(row['tokenId'])
                curr_sent.append(row['normalizedWord'].replace('’', '\''))
        print(missing-maybe_found)
        '''
        for prompt in best_matches: 
            if best_matches[prompt][1] > 0: 
                print(prompt)
                print(best_matches[prompt])
        '''

def main(): 
    #get_generation_len()
    get_book_excerpts()
    #standardize_prompts()

if __name__ == '__main__': 
    main()
