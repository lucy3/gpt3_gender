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
from nltk.tokenize import word_tokenize 
from nltk.metrics.distance import edit_distance
import edlib

ROOT = '/mnt/data0/lucy/gpt3_bias/'
PROMPTS = ROOT + 'logs/original_prompts/'
TOKENS = ROOT + '/logs/tokens/'
STORIES = ROOT + 'logs/generated_0.9/' 
LOGS = ROOT + 'logs/'
OUTPUT = LOGS + 'book_excerpts/'

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
    w = w.replace('--', ' ').replace('—', ' ').replace('’', '\'')
    w = w.translate(str.maketrans('', '', string.punctuation))
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
    prompts_nopunct = defaultdict(str)
    book2prompt = defaultdict(list)
    for f in os.listdir(PROMPTS): 
        with open(PROMPTS + f, 'r') as infile:
            for line in infile: 
                p = line.split('\t')[2].strip()
                prompts.add(p)
                pnp = clean_words(p)
                prompts_nopunct[p] = pnp
                book2prompt[f].append(p)
    assert len(set(prompts_nopunct.values()) - set(input2len_nopunct.keys())) == 0
    assert len(set(input2len_nopunct.keys()) - set(prompts_nopunct.values())) == 0
    return input2len_nopunct, book2prompt, prompts_nopunct

def get_book_excerpts():
    detokenizer = TreebankWordDetokenizer()
    input2len, book2prompt, prompts_nopunct = standardize_prompts()
    prompt2tokens = {}
    for prompt in prompts_nopunct:
        toks = word_tokenize(prompt)
        clean_toks = []
        for tok in toks: 
            clean_tok = clean_words(tok)
            if clean_tok.strip() != '': 
                clean_toks.append(clean_tok) 
        if clean_toks[0] == 'Cap' and clean_toks[1] == 'n': 
            # single edge case of tokenizer difference
            clean_toks = ['Capn'] + clean_toks[2:]
        prompt2tokens[prompt] = clean_toks
    for f in book2prompt:
        bookTokens = []
        originalbookTokens = []
        bookTokenIDs = []
        with open(TOKENS + f, 'r') as infile: 
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader: 
                bookTokens.append(row['normalizedWord'])
                originalbookTokens.append(row['originalWord'].replace('’', "'"))
                bookTokenIDs.append(int(row['tokenId']))
        for i, idx in enumerate(bookTokenIDs): 
            assert i == idx
        cleanIDs = []
        cleanbookTokens = []
        outfile = open(OUTPUT + f, 'w') 
        for i, tok in enumerate(bookTokens): 
            clean_tok = clean_words(tok)
            if clean_tok.strip() != '': 
                cleanIDs.append(bookTokenIDs[i])
                cleanbookTokens.append(clean_tok)
        for prompt in book2prompt[f]: 
            found = False
            prompt_toks = prompt2tokens[prompt]
            prompt_len = len(prompt_toks)
            for i in range(len(cleanbookTokens)): 
                if cleanbookTokens[i:i + prompt_len] == prompt_toks:
                    story_start = cleanIDs[i]
                    story_end = story_start + int(input2len[prompts_nopunct[prompt]])
                    story_tokens = originalbookTokens[story_start:story_end]
                    s = 0
                    story = ''
                    for j, tok in enumerate(story_tokens): 
                        if tok == '.' or tok == '?' or tok == '!': 
                            story += detokenizer.detokenize(story_tokens[s:j+1]) + ' '
                            s = j + 1
                    found = True
                    outfile.write(story + '\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n')
                    break 
            if not found: 
                print(f, prompt, prompt2tokens[prompt])
        outfile.close()

def main(): 
    #get_generation_len()
    get_book_excerpts()
    #standardize_prompts()

if __name__ == '__main__': 
    main()
