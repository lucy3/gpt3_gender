"""
Getting entities in stories
and the pronouns associated with them. 
"""
import os
import csv
from collections import defaultdict, Counter
import json
import re
import numpy as np

LOGS = '/mnt/data0/lucy/gpt3_bias/logs/'

def remove_punct(s): 
    #regex = re.compile('[%s]' % re.escape(string.punctuation))
    regex = re.compile('[^a-zA-Z0-9]')
    return regex.sub('', s) 

def get_characters_to_prompts(prompts_path, tokens_path, txt_path, char_idx_path, num_gens=5):
    '''
    Input: path to original prompts, path to tokens 
    
    Assumes that the generated stories are in the same order as the prompts,
    with num_gens stories per prompt. 
    '''
    count = 0
    char_story_count = 0
    for filename in os.listdir(tokens_path): 
        title = filename.replace('.tokens', '')
        print(title)
        char_order = [] # character, where index is generated story index
        with open(prompts_path + title, 'r') as infile: 
            reader = csv.reader(infile, delimiter='\t')
            for row in reader: 
                char_ID = row[0]
                char_name = row[1]
                prompt = row[2]
                char_order.extend([(char_ID, char_name)]*num_gens)
        if len(char_order) == 0: 
            print("----- No prompts -----")
            continue 
        count += 1

        # sanity check that story has the character in it
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

        # get mapping from story idx to its token span 
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
        
        # the number of stories should be the number of character prompts * num_gens
        if len(idx_tokenIDs) != len(char_order): 
            print("PROBLEM!!!!!", len(idx_tokenIDs), len(char_order))
            continue
        
        assert len(idx_tokenIDs) == len(char_order)
        # mapping from character to story spans 
        char_story = defaultdict(list) # {character name: [(story_idx, start token idx, end token idx)] }
        for story_idx in idx_tokenIDs: 
            tup = (story_idx, idx_tokenIDs[story_idx][0], idx_tokenIDs[story_idx][1])
            char_story[char_order[story_idx][1]].append(tup)
            
        with open(char_idx_path + title + '.json', 'w') as outfile: 
            char_story_count += 1
            json.dump(char_story, outfile)
    print(count)
    print(char_story_count)
            
def get_entities_dict(ents_path, title, main_characters): 
    '''
    Gets the start and end tokens for every entity 
    '''
    entities = {} # { (start, end) : entity name } 
    with open(ents_path + title + '/' + title + '.ents', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            start = int(contents[0])
            end = int(contents[1])
            ner = contents[2]
            entity = contents[3]
            if ner == 'PROP_PER' or entity in main_characters: 
                entities[(start, end)] = entity
    return entities

def get_coref_label_dict(ents_path, title, entities, idx2story): 
    '''
    Get the coref group for every proper name person entity
    Coref group is groupnumber_storyID, where storyIDs are unique, 
    and group number is from the coref results. 
    '''
    coref_label = {} # { (start, end, entity name) : coref_group_id } 
    max_group = 0
    with open(ents_path + title + '/' + title + '.predicted.conll.ents', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            group = contents[0] 
            entity = contents[1]
            start = int(contents[2])
            end = int(contents[3]) 
            chain_id = group + '_' + str(idx2story[start])
            max_group = max(max_group, int(group))
            if (start, end) in entities: 
                coref_label[(start, end, entities[start, end])] = chain_id
    # some entities don't have coref chains
    max_group += 1
    for tup in entities: 
        start = tup[0]
        end = tup[1]
        if (start, end, entities[tup]) not in coref_label: 
           chain_id = str(max_group) + '_' + str(idx2story[start])
           coref_label[(start, end, entities[start, end])] = chain_id
           max_group += 1
    return coref_label

def get_coref_chain_dict(ents_path, title, pronouns, coref_label, idx2story): 
    '''
    Get all of the pronouns associated with a 
    coref group associated with entities 
    Each coref group is split by story, so that one story's pronouns are not
    connected to another. 
    '''
    coref_chain = defaultdict(list) # { coref_group_ID : [pronouns] } 
    with open(ents_path + title + '/' + title + '.predicted.conll.ents', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            group = contents[0] 
            entity = contents[1]
            start = int(contents[2])
            end = int(contents[3])
            chain_id = group + '_' + str(idx2story[start])
            # only groups containing proper names matter
            if chain_id in list(coref_label.values()): 
                if entity.lower() in pronouns: 
                    coref_chain[chain_id].append(pronouns[entity.lower()])
    return coref_chain

def print_character_network(char_neighbors, char_pronouns): 
    for main_char in char_neighbors: 
        neighbor_dict = char_neighbors[main_char]
        print(main_char) 
        print(' ------------------------- ')
        for neighbor in neighbor_dict: 
            print(neighbor['character_name'], neighbor['aliases'], neighbor['gender'])
            print()

def get_entities_pronouns(ents_path, prompts_path, char_idx_path, char_nb_path): 
    '''
    For each named person, find how GPT-3 tends to gender that name
    based on coref chains in the text
    
    For each main character, what are the genders of other named people in their
    stories? 

    inputs: 
    - path to entities
    - path to prompts 
    '''
    pronouns = {'he' : 'masc', 'his' : 'masc', 'him' : 'masc', 
              'himself' : 'masc', 'she' : 'fem', 'her' : 'fem', 
              'hers' : 'fem', 'herself' : 'fem', 'they' : 'neut', 'their' : 'neut', 
              'them' : 'neut', 'theirs' : 'neut', 'theirself' : 'neut'}
    for title in os.listdir(ents_path): 
        print(title)
        # now, get characters associated with a character 
        if not os.path.exists(char_idx_path + title + '.json'): continue
        with open(char_idx_path + title + '.json', 'r') as infile: 
            char_story = json.load(infile) # {character name: [(start token idx, end token idx)] }
        
        idx2story = {} # token id to story idx
        for char in char_story: 
            for tup in char_story[char]: 
                story_idx, start, end = tup
                for i in range(start, end + 1): 
                    idx2story[i] = story_idx
        
        main_characters = set(char_story.keys())
        entities = get_entities_dict(ents_path, title, main_characters) # (start, end) : entity name
        coref_label = get_coref_label_dict(ents_path, title, entities, idx2story) # entities to coref group
        coref_chain = get_coref_chain_dict(ents_path, title, pronouns, coref_label, idx2story) # coref group to pronouns
        
        char_pronouns = defaultdict(Counter) # {character name : [pronouns in all coref chains]}
        # This is a list because one name, e.g. Michelle, might have multiple coref chains to create a cluster
        char_group_ids = defaultdict(list)
        for ent in entities:
            char = entities[ent]
            story_idx = idx2story[ent[0]]
            if (ent[0], ent[1], char) in coref_label:
                char_group_ids[char + '_' + str(story_idx)].append(coref_label[(ent[0], ent[1], char)])
        # if "Michelle" and "Michelle Obama" are in the same coref chain together, we group their clusters together
        # We can have one name be the "base char" that other renamings of that character are then grouped with
        chainID2name = {} # one to one mapping of already-seen chain_id to base_char
        aliases = defaultdict(set) # base_char : [other chars that share coref chains with it]
        for char in char_group_ids: 
            pns = []
            base_char = char
            for group in char_group_ids[char]: 
                if group in chainID2name: 
                    base_char = chainID2name[group]
                    char_name = '_'.join(char.split('_')[:-1])
                    aliases[base_char].add(char_name)
                pns.extend(coref_chain[group])
            for group in char_group_ids[char]: 
                # assign this group to base 
                chainID2name[group] = base_char
            pns = Counter(pns)
            char_pronouns[base_char] += pns

        char_story_rev = {} # story idx to character
        for char in char_story: 
            story_indices = char_story[char] # list of story starts and ends
            for story_span in story_indices: 
                story_idx = story_span[0]
                char_story_rev[story_idx] = char

        # {character name : [{"character name": "", "gender": {masc: #, fem: #, neut: #}, "aliases": [name]}] }
        char_neighbors = defaultdict(list) 
 
        seen_mains = set() # set of main character _ story idx
        for base_char in char_pronouns: 
            story_idx = int(base_char.split('_')[-1])
            main_char = char_story_rev[story_idx]
            if base_char == main_char + '_' + str(story_idx): 
                seen_mains.add(base_char)
            neighbor_dict = {}
            neighbor_dict['character_name'] = base_char
            neighbor_dict['aliases'] = list(aliases[base_char])
            neighbor_dict['gender'] = char_pronouns[base_char]
            char_neighbors[main_char].append(neighbor_dict)
           
        # due to NER error, some main characters weren't recognized and thus have no pronouns
        for story_idx in char_story_rev: 
            main_char = char_story_rev[story_idx]
            if main_char + '_' + str(story_idx) not in seen_mains:
                neighbor_dict = {}
                base_char = main_char + '_' + str(story_idx)
                neighbor_dict['character_name'] = base_char
                neighbor_dict['aliases'] = []
                neighbor_dict['gender'] = {}
                char_neighbors[main_char].append(neighbor_dict)
        with open(char_nb_path + title + '.json', 'w') as outfile: 
            json.dump(char_neighbors, outfile)

def calculate_recurrence(tokens_path, char_idx_path):
    num_times = [] # number of times main character occurs in story
    ranges = [] # range of tokens the main character spans
 
    for f in os.listdir(tokens_path):
        print(f) 
        title = f.replace('.tokens', '')
        if not os.path.exists(char_idx_path + title + '.json'): continue # TODO: fix this!
        with open(char_idx_path + title + '.json', 'r') as infile: 
            char_story = json.load(infile)
        start2char = {}
        for charname in char_story: 
            for tup in char_story[charname]: 
                start2char[tup[0]] = charname
        with open(tokens_path + f, 'r') as infile: 
            curr_start = None
            charId = None
            main_char_idx = [] # all of the indices in which the main character occurs
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if int(row['tokenId']) in start2char: 
                    if curr_start is not None: 
                        num_times.append(len(main_char_idx))
                        ranges.append(main_char_idx[-1] - main_char_idx[0])
                    curr_start = int(row['tokenId'])
                    charId = None
                    main_char_idx = []
                if row['originalWord'] == start2char[curr_start] and charId is None: 
                    charId = row['characterId']
                if row['characterId'] == charId: 
                    main_char_idx.append(int(row['tokenId']))
            num_times.append(len(main_char_idx))
            ranges.append(main_char_idx[-1] - main_char_idx[0])
    print(np.mean(num_times), np.mean(ranges))
    
def get_topics_for_txt(txt_path, prompts_path, topic_out_path, \
                       gender_path, generated, story_topics, num_gens=5):
    gender_topics = {'gender':[], 'topic':[], 'value':[]}
    for title in sorted(os.listdir(txt_path)):
        char_order = [] # character, where index is generated story index
        with open(prompts_path + title, 'r') as infile: 
            reader = csv.reader(infile, delimiter='\t')
            for row in reader: 
                char_ID = row[0]
                char_name = row[1]
                prompt = row[2]
                char_order.extend([char_name]*num_gens)
        if len(char_order) == 0: continue
        
        with open(gender_path + title + '.json', 'r') as infile: 
            gender_dict = json.load(infile)

        for i, char in enumerate(char_order): 
            story_title_id = title + str(i+1)
            if not generated: 
                story_title_id = 'ORIG_' + story_title_id
            topic_dict = story_topics[story_title_id]
            assert char in gender_dict 
            neighbors = gender_dict[char]
            gender = None
            for neighbor in neighbors: 
                if neighbor['character_name'] == char + '_' + str(i): 
                    gender = neighbor['gender_label']
            if gender is None:
                # failed to detect main character entity
                print("PROBLEM!!!!", title, char, i)
                gender = 'other'
            for topic_id in topic_dict: 
                gender_topics['gender'].append(gender)
                gender_topics['topic'].append(topic_id)
                gender_topics['value'].append(topic_dict[topic_id])
    with open(topic_out_path, 'w') as outfile: 
        json.dump(gender_topics, outfile)

def get_gendered_topics(txt_path, prompts_path, topic_out_path, \
                        gender_path, generated): 
    topic_dir = LOGS + 'topics_0.9' 
    doc_topic_file = '%s/doc-topics.gz' % topic_dir
    doc_topics = open(doc_topic_file).read().splitlines() # list of topics
    story_ids = open(topic_dir + '/story_id_order').read().splitlines() # story IDs 
    story_topics = defaultdict(dict) # story ID : {topic id : value, topic id: value}
    for i, doc in enumerate(doc_topics): 
        contents = doc.split('\t')
        topics = [float(i) for i in contents[2:]]
        story_title_id = story_ids[i]
        if (generated & (not story_title_id.startswith("ORIG_"))) or \
            (not generated & story_title_id.startswith("ORIG_")):             
            assert len(topics) == 50
            for topic_id, value in enumerate(topics): 
                story_topics[story_title_id][topic_id] = value
    if generated: 
        get_topics_for_txt(txt_path, prompts_path, \
                           topic_out_path, gender_path, generated, story_topics)
    else: 
        get_topics_for_txt(txt_path, prompts_path, \
                           topic_out_path, gender_path, generated, story_topics, num_gens=1)

def main(): 
    generated = False
    if generated: 
        ents_path = LOGS + 'generated_0.9_ents/'
        tokens_path = LOGS + 'plaintext_stories_0.9_tokens/'
        txt_path = LOGS + 'plaintext_stories_0.9/'
        char_idx_path = LOGS + 'char_indices_0.9/'
        char_nb_path = LOGS + 'char_neighbors_0.9/'
        topic_out_path = LOGS + 'gender_topics_0.9.json'
        gender_path = LOGS + 'char_gender_0.9/'
        num_gens = 5
    else: 
        ents_path = LOGS + 'book_excerpts_ents/'
        tokens_path = LOGS + 'book_excerpts_tokens/'
        txt_path = LOGS + 'book_excerpts/'
        char_idx_path = LOGS + 'orig_char_indices/' 
        char_nb_path = LOGS + 'orig_char_neighbors/'
        topic_out_path = LOGS + 'orig_gender_topics.json'
        gender_path = LOGS + 'orig_char_gender/'
        num_gens = 1
    prompts_path = LOGS + 'original_prompts/' 
    get_characters_to_prompts(prompts_path, tokens_path, txt_path, char_idx_path, num_gens=num_gens)
    get_entities_pronouns(ents_path, prompts_path, char_idx_path, char_nb_path)
    #calculate_recurrence(tokens_path, char_idx_path)
    #get_gendered_topics(txt_path, prompts_path, topic_out_path, gender_path, generated)

if __name__ == '__main__': 
    main()
