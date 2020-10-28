"""
Getting entities in stories
and the pronouns associated with them. 
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
        print(title)
        char_order = [] # character, where index is generated story index
        with open(prompts_path + title, 'r') as infile: 
            reader = csv.reader(infile, delimiter='\t')
            for row in reader: 
                char_ID = row[0]
                char_name = row[1]
                prompt = row[2]
                char_order.extend([(char_ID, char_name)]*num_gens)
        if len(char_order) == 0: continue 

        char_idx = defaultdict(list) # {character : [idx in tokens_path]}

        # sanity check that story has character in it
        with open(txt_path + title, 'r') as infile: 
            story = ''
            story_idx = 0
            dot_count = 0
            for line in infile: 
                if line.strip() == '@': 
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
                if row['normalizedWord'] == '@':
                    dot_count += 1
                else: 
                    dot_count = 0
                if dot_count == 20: 
                    end_tokenID = row['tokenId']
                    idx_tokenIDs[story_idx] = (start_tokenID, int(end_tokenID))
                    story_idx += 1
                    start_tokenID = int(end_tokenID) + 1
                    dot_count = 0
        '''
        if len(idx_tokenIDs) != len(char_order): 
            print("PROBLEM!!!!!", len(idx_tokenIDs), len(char_order))
            continue
        '''
        assert len(idx_tokenIDs) == len(char_order)
        
        char_story = defaultdict(list) # {character name: [(start token idx, end token idx)] }
        for story_idx in idx_tokenIDs: 
            char_story[char_order[story_idx][1]].append(idx_tokenIDs[story_idx])
            
        with open(LOGS + 'char_indices_0.9/' + title + '.json', 'w') as outfile: 
            json.dump(char_story, outfile)
            
def get_entities_dict(ents_path, title): 
    entities = {} # { (start, end) : entity name } 
    with open(ents_path + title + '/' + title + '.ents', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            start = int(contents[0])
            end = int(contents[1])
            ner = contents[2]
            entity = contents[3]
            if ner == 'PROP_PER': 
                entities[(start, end)] = entity
    return entities

def get_coref_label_dict(ents_path, title, entities): 
    coref_label = {} # { (start, end, entity name) : coref_group_id } 
    with open(ents_path + title + '/' + title + '.predicted.conll.ents', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            group = contents[0] 
            entity = contents[1]
            start = int(contents[2])
            end = int(contents[3])
            if (start, end) in entities: 
                coref_label[(start, end, entities[start, end])] = group
    return coref_label

def get_coref_chain_dict(ents_path, title, pronouns, coref_label): 
    coref_chain = defaultdict(list) # { coref_group_ID : [pronouns] } 
    with open(ents_path + title + '/' + title + '.predicted.conll.ents', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            group = contents[0] 
            entity = contents[1]
            start = int(contents[2])
            end = int(contents[3])
            # only groups containing proper names matter
            if group in list(coref_label.values()): 
                if entity.lower() in pronouns: 
                    coref_chain[group].append(pronouns[entity.lower()])
    return coref_chain

def print_character_network(char_neighbors, char_pronouns): 
    for char in char_neighbors: 
        neighbor_dict = char_neighbors[char]
        print(char_pronouns[char])
        s = char + ' --- '
        neighbor_names = set()
        for neighbor in neighbor_dict: 
            neighbor_names.add(neighbor['character_name'])
        for name in neighbor_names: 
            s += name + ', '
        print(s)

def get_entities_gender(ents_path, prompts_path): 
    pronouns = {'he' : 'masc', 'his' : 'masc', 'him' : 'masc', 
              'himself' : 'masc', 'she' : 'fem', 'her' : 'fem', 
              'hers' : 'fem', 'herself' : 'fem', 'they' : 'neut', 'their' : 'neut', 
              'them' : 'neut', 'theirs' : 'neut', 'theirself' : 'neut'}
    for title in os.listdir(ents_path): 
        print(title)
        entities = get_entities_dict(ents_path, title)
        coref_label = get_coref_label_dict(ents_path, title, entities)
        coref_chain = get_coref_chain_dict(ents_path, title, pronouns, coref_label) 

        char_pronouns = defaultdict(Counter) # {character name : [pronouns in all coref chains]}
        # This is a list because one name might have multiple coref chains
        char_group_ids = defaultdict(list)
        for ent in entities:
            char = entities[ent]
            if (ent[0], ent[1], char) in coref_label: 
                char_group_ids[char].append(coref_label[(ent[0], ent[1], char)])
        for char in char_group_ids: 
            pns = []
            for group in char_group_ids[char]: 
                pns.extend(coref_chain[group])
            pns = Counter(pns)
            char_pronouns[char] = pns
        
        # now, get characters associated with a character 
        if not os.path.exists(LOGS + 'char_indices_0.9/' + title + '.json'): continue
        with open(LOGS + 'char_indices_0.9/' + title + '.json', 'r') as infile: 
            char_story = json.load(infile) # {character name: [(start token idx, end token idx)] }
            
        char_story_rev = {}
        for char in char_story: 
            story_indices = char_story[char] # list of story starts and ends
            for story_idx in story_indices: 
                char_story_rev[tuple(story_idx)] = char
        
        # {character name : [{"character name": "", "gender": {masc: #, fem: #, neut: #}}] }
        char_neighbors = defaultdict(list) 
        for ent in entities: 
            # find story containing this entity
            ent_start = ent[0]
            ent_end = ent[1]
            for story_idx in char_story_rev:
                # get main character associated with story
                char = char_story_rev[story_idx]
                story_start = story_idx[0]
                story_end = story_idx[1]
                if ent_start >= story_start and ent_end < story_end: 
                    # story found
                    neighbor = entities[ent]
                    neighbor_dict = {}
                    neighbor_dict['character_name'] = neighbor
                    if neighbor in char_pronouns: 
                        neighbor_dict['gender'] = char_pronouns[neighbor]
                    else: 
                        neighbor_dict['gender'] = {}
                    char_neighbors[char].append(neighbor_dict)
                    break
        
        with open(LOGS + 'char_neighbors_0.9/' + title + '.json', 'w') as outfile: 
            json.dump(char_neighbors, outfile)
        
def main(): 
    ents_path = LOGS + 'generated_0.9_ents/'
    tokens_path = LOGS + 'plaintext_stories_0.9_tokens/'
    txt_path = LOGS + 'plaintext_stories_0.9/'
    prompts_path = LOGS + 'original_prompts/'
    get_characters_to_prompts(prompts_path, tokens_path, txt_path)
    #get_entities_gender(ents_path, prompts_path)

if __name__ == '__main__': 
    main()
