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
        nopunct = k.translate(str.maketrans('', '', string.punctuation))
        input2len_nopunct[nopunct] = input2len[k]
    prompts = set()
    prompts_nopunct = set()
    book2prompt = defaultdict(list)
    for f in os.listdir(PROMPTS): 
        with open(PROMPTS + f, 'r') as infile:
            for line in infile: 
                p = line.split('\t')[2].strip()
                prompts.add(p)
                pnp = p.translate(str.maketrans('', '', string.punctuation))
                prompts_nopunct.add(pnp)
                book2prompt[f].append(pnp)
    assert len(prompts_nopunct - set(input2len_nopunct.keys())) == 0
    assert len(set(input2len_nopunct.keys()) - prompts_nopunct) == 0
    return input2len_nopunct, book2prompt

def get_book_excerpts(): 
    input2len, book2prompt = standardize_prompts() # punctuationless input to length
    detokenizer = TreebankWordDetokenizer()
    for f in book2prompt: 
        found = set()
        with open(TOKENS + f, 'r') as infile:
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            curr_sent = []
            curr_sentID = None
            for row in reader: 
                if row['sentenceID'] != curr_sentID and curr_sentID is not None: 
                    sent = detokenizer.detokenize(curr_sent) 
                    sent = sent.translate(str.maketrans('', '', string.punctuation))
                    if 'Frenshams mouth hardened' in sent: print(sent)
                    if sent in book2prompt[f]: 
                        found.add(sent)
                    # detokenize curr_sent 
                    # depunctuate 
                    # check for sentence in book2prompt[f]
                    curr_sent = []
                    curr_sentID = row['sentenceID']
                if curr_sentID is None: 
                    curr_sentID = row['sentenceID']
                curr_sent.append(row['originalWord'].replace('’', '\''))
        print(len(found), len(book2prompt[f]))
        print(set(book2prompt[f])-found)
        break

def main(): 
    #get_generation_len()
    get_book_excerpts()
    #standardize_prompts()

if __name__ == '__main__': 
    main()
