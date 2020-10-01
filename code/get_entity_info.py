"""
"""
import os
import csv
from collections import defaultdict, Counter
import json
import re

LOGS = '/mnt/data0/lucy/gpt3_bias/logs/'

def remove_punct(s): 
    #regex = re.compile('[%s]' % re.escape(string.punctuation))
    regex = re.compile('[^a-zA-Z0-9]')
    return regex.sub('', s) 

def get_characters_to_prompts(prompts_path, tokens_path, txt_path):
    '''
    Input: path to original prompts, path to tokens 
    
    Assumes that the generated stories are in the same order as the prompts,
    with 5 prompts per story. 
    '''
    num_gens = 5
    for title in os.listdir(txt_path): 
        #print(title)
        char_order = [] # character, where index is generated story index
        with open(prompts_path + title, 'r') as infile: 
            reader = csv.reader(infile, delimiter='\t')
            for row in reader: 
                char_ID = row[0]
                char_name = row[1]
                prompt = row[2]
                char_order.extend([(char_ID, char_name)]*num_gens)
        if len(char_order) == 0: continue
        #print(len(char_order))
        char_idx = defaultdict(list) # {character : [idx in tokens_path]}

        # sanity check that story has character in it
        with open(txt_path + title, 'r') as infile: 
            story = ''
            story_idx = 0
            dot_count = 0
            for line in infile: 
                if line.strip() == '.': 
                    dot_count += 1
                else: 
                    dot_count = 0 
                if dot_count == 20: 
                    assert char_order[story_idx][1] in story 
                    story = ''
                    story_idx += 1
                    dot_count = 0
                else: 
                    story += line

        idx_tokenIDs = defaultdict(list) # { story idx : (start token ID, end token ID) }
        with open(tokens_path + title + '.tokens', 'r') as infile: 
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            start_tokenID = 0
            end_tokenID = 0
            story_idx = 0
            dot_count = 0
            for row in reader:
                if row['normalizedWord'] == '.': 
                    dot_count += 1
                else: 
                    dot_count = 0
                if dot_count == 20: 
                    end_tokenID = row['tokenId']
                    idx_tokenIDs[story_idx] = (start_tokenID, int(end_tokenID))
                    story_idx += 1
                    start_tokenID = int(end_tokenID) + 1
                    dot_count = 0
        if len(idx_tokenIDs) != len(char_order): 
            print(title, len(idx_tokenIDs), len(char_order))
            continue
        char_story = defaultdict(list) # {character name: [(start token idx, end token idx)] }
        for story_idx in idx_tokenIDs: 
            char_story[char_order[story_idx][1]].append(idx_tokenIDs[story_idx])
            
        with open(LOGS + 'char_indices_0.9/' + title + '.json', 'w') as outfile: 
            json.dump(char_story, outfile)

def get_entities_gender(ents_path, prompts_path): 
    pronouns = {'he' : 'masc', 'his' : 'masc', 'him' : 'masc', 
              'himself' : 'masc', 'she' : 'fem', 'her' : 'fem', 
              'hers' : 'fem', 'herself' : 'fem', 'they' : 'neut', 'their' : 'neut', 
              'them' : 'neut', 'theirs' : 'neut', 'theirself' : 'neut'}
    for title in os.listdir(ents_path): 
        entities = {} # { (start, end) : entity name } 
        with open(ents_path + title + '/' + title + '.ents', 'r') as infile: 
            for line in infile: 
                contents = line.strip().split('\t')
                start = contents[0]
                end = contents[1]
                ner = contents[2]
                entity = contents[3]
                if ner == 'PROP_PER': 
                    entities[(start, end)] = entity
        coref_label = {} # { (start, end, entity name) : coref_group_id } 
        with open(ents_path + title + '/' + title + '.predicted.conll.ents', 'r') as infile: 
            for line in infile: 
                contents = line.strip().split('\t')
                group = contents[0] 
                entity = contents[1]
                start = contents[2]
                end = contents[3]
                if (start, end) in entities: 
                    coref_label[(start, end, entities[start, end])] = group
        coref_chain = defaultdict(list) # { coref_group_ID : [pronouns] } 
        with open(ents_path + title + '/' + title + '.predicted.conll.ents', 'r') as infile: 
            for line in infile: 
                contents = line.strip().split('\t')
                group = contents[0] 
                entity = contents[1]
                start = contents[2]
                end = contents[3]
                if group in coref_label: 
                    if entity.lower() in pronouns: 
                        coref_chain[group].append(pronouns[entity.lower()])            
        all_pns = Counter()
        char_pronouns = defaultdict(Counter)
        # This is a list because one name might have multiple coref chains
        char_group_ids = defaultdict(list)
        for tup in coref_label:
            char = tup[2]
            char_group_ids[char].append(coref_label[tup])
        for char in char_group_ids: 
            pronouns = []
            for group in char_group_ids[char]: 
                pronouns.extend(coref_chain[group])
            pns = Counter(pronouns)
            all_pns.update(pns)
            char_pronouns[char] = pns
            print(char, pns)
        print("OVERALL:", all_pns)
        break
        
def main(): 
    ents_path = LOGS + 'generated_0.9_ents/'
    tokens_path = LOGS + 'plaintext_stories_0.9_tokens/'
    txt_path = LOGS + 'plaintext_stories_0.9/'
    prompts_path = LOGS + 'original_prompts/'
    get_characters_to_prompts(prompts_path, tokens_path, txt_path)
    #get_entities_gender(ents_path, prompts_path)

if __name__ == '__main__': 
    main()