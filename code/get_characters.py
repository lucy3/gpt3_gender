"""
Gets characters 
"""
import csv
from collections import Counter, defaultdict
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random

TOKENS = '/mnt/data0/lucy/gpt3_bias/logs/tokens/'

def extract_people(filepath):
    '''
    Groups characters by character IDs in narrative non-quote text, finds the characters
    that are overall mentioned the most (top %2). Then, from that pool 
    of characters, finds characters mentioned by first name at least 50 times. 
    We want only sentences where only 1 character is mentioned
    '''
    pronouns = set(['she', 'She', 'her', 'Her', 'he', 'He', 'his', 
          'His', 'him', 'Him', 'herself', 'Herself', 'himself', 'Himself'])
    curr_character = ''
    curr_character_ID = None 
    sentence_ID = None
    character_counts = defaultdict(Counter)
    sentences = defaultdict(list)
    with open(filepath, 'r') as csvfile: 
        reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['inQuotation'] == 'I-QUOTE': continue # exclude quotations
            if row['characterId'] != '-1': 
                if row['characterId'] != curr_character_ID and curr_character_ID is not None: 
                   # new character 
                   character_counts[curr_character_ID][curr_character.strip()] += 1
                   sentences[curr_character.strip()].append(sentence_ID)
                   curr_character_ID = row['characterId'] 
                   curr_character = row['normalizedWord'] + ' '
                   sentence_ID = row['sentenceID']
                elif row['characterId'] == curr_character_ID: 
                   # continue a previous token for same character
                   curr_character += row['normalizedWord'] + ' ' 
                else: 
                   # new character 
                   curr_character_ID = row['characterId']
                   curr_character = row['normalizedWord'] + ' '
                   sentence_ID = row['sentenceID']
            elif row['characterId'] == '-1' and curr_character_ID is not None: 
                # finish off previous character
                character_counts[curr_character_ID][curr_character.strip()] += 1
                sentences[curr_character.strip()].append(sentence_ID)
                curr_character_ID = None
                sentence_ID = None
                curr_character = ''
    if curr_character_ID is not None: 
       character_counts[curr_character_ID][curr_character.strip()] += 1

    # get sentence ID to characters in sentence
    sentences_rev = defaultdict(list)
    for c in sentences: 
       for sentence_ID in sentences[c]: 
           sentences_rev[sentence_ID].append(c)
    # get sentence IDs to sentence tokens
    IDs_sents = {}
    with open(filepath, 'r') as csvfile: 
        reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        curr_sentence_ID = None
        curr_sentence = []
        num_people = 0
        for row in reader:
            if row['inQuotation'] == 'I-QUOTE': continue # exclude quotations
            if row['sentenceID'] == curr_sentence_ID:
                curr_sentence.append(row['normalizedWord'])
                if row['characterId'] != '-1' or row['supersense'] == 'B-noun.person' or row['supersense'] == 'I-noun.person': 
                    # some characters do not have a supersense, and some people do not have character IDs
                    num_people += 1
            else:
                if num_people == 1 and not set(curr_sentence) & pronouns and \
                        '\\\"' not in curr_sentence and '``' not in curr_sentence: 
                    # we want a single token person in the sentence
                    IDs_sents[curr_sentence_ID] = curr_sentence
                curr_sentence_ID = row['sentenceID']
                curr_sentence = [row['normalizedWord']]
                if row['characterId'] != '-1' or row['supersense'] == 'B-noun.person' or row['supersense'] == 'I-noun.person': 
                    num_people = 1
                else: 
                    num_people = 0
        if not set(curr_sentence) & pronouns: 
            IDs_sents[curr_sentence_ID] = curr_sentence
 
    # get sentence IDs that only contain one character
    # no character speaker tags
    # sentence does not contain gendered pronouns 
    single_char_sentences = set(IDs_sents.keys())
    # get total number of characters mentioned
    total_mentions = 0 
    for c in character_counts:
       num_mentions = sum(character_counts[c].values())
       total_mentions += num_mentions
    # get main characters
    main_characters = []
    for c in character_counts: 
       num_mentions = sum(character_counts[c].values())
       if num_mentions/float(total_mentions) >= 0.02: 
           for alias in character_counts[c]: 
               if alias.lower() not in pronouns and len(alias.split()) == 1:  
                   sIDs = sentences[alias]
                   num_sents = 0 
                   for s_ID in sIDs: 
                       if s_ID in single_char_sentences: 
                           num_sents += 1
                   if num_sents >= 20: 
                       main_characters.append((c, alias))                 
    # write to file 
    for mc in main_characters: 
       sents = set(sentences[mc[1]]) & single_char_sentences
       sents = random.sample(sents, 20)
       print(mc)
       for s_ID in sents: 
          s = IDs_sents[s_ID]
          s = TreebankWordDetokenizer().detokenize(s)
          print(s)

def main(): 
    f = 'lee_to_kill_a_mockingbird' 
    extract_people(TOKENS + f)

if __name__ == '__main__': 
    main()
