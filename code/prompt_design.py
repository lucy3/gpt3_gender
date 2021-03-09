import json
import os
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv 
from collections import defaultdict, Counter
import random
from nltk.stem import WordNetLemmatizer

LOGS = '/mnt/data0/lucy/gpt3_bias/logs/'
DATA = '/mnt/data0/lucy/gpt3_bias/data/'

def get_gendered_prompts(): 
    '''
    Gather prompts that have consistentally fem or masc characters
    '''
    num_gens = 5
    fem_prompts = set() # tuple of char, prompt
    masc_prompts = set() # tuple of char, prompt
    for f in os.listdir(LOGS + 'original_prompts/'):
        if not os.path.exists(LOGS + 'char_gender_0.9/' + f + '.json'): continue
        with open(LOGS + 'char_gender_0.9/' + f + '.json', 'r') as infile: 
            gender_dict = json.load(infile)
        char_set = {} # main character name _ story ID : gender
        for char in gender_dict: 
            neighbor_dict = gender_dict[char]
            for neighbor in neighbor_dict: 
                is_main = False
                gender = neighbor['gender_label']
                neighbor_n = neighbor['character_name']
                if neighbor_n.startswith(char + '_'): 
                    # main character
                    is_main = True
                else: 
                    for al in neighbor['aliases']: 
                        if al == char: 
                            is_main = True
                if is_main: 
                    char_set[neighbor_n] = gender 
        
        story_idx = 0 
        with open(LOGS + 'original_prompts/' + f, 'r') as infile: 
            for line in infile: 
                contents = line.strip().split('\t')
                characterID = contents[0]
                char = contents[1]
                prompt = contents[2]
                genders = set()
                for idx in range(story_idx, story_idx + num_gens): 
                    char_ID = char + '_' + str(idx)
                    genders.add(char_set[char_ID])
                if len(genders) == 1: 
                    if '-RRB-' in prompt or '-LRB-' in prompt: 
                        prompt = prompt.replace(' -RRB-', ')').replace('-LRB- ', '(')
                        prompt = prompt.replace('-RRB-', ')').replace('-LRB-', '(')
                    if 'masc' in genders: 
                        masc_prompts.add((f, story_idx, char, prompt))
                    elif 'fem' in genders: 
                        fem_prompts.add((f, story_idx, char, prompt))
                story_idx += num_gens
    return fem_prompts, masc_prompts

def get_embed_sim(fem_prompts, masc_prompts, replaced_name): 
    model = SentenceTransformer('stsb-roberta-large')
    start = time.time()
    fem_embed = model.encode(fem_prompts, show_progress_bar=False)
    print('TIME:', time.time() - start)
    fem_embed = np.array(fem_embed)
    start = time.time()
    masc_embed = model.encode(masc_prompts, show_progress_bar=False)
    print('TIME:', time.time() - start)
    masc_embed = np.array(masc_embed)
    sims = cosine_similarity(fem_embed, masc_embed)
    
    outpath = LOGS + 'prompt_matching/'
    np.save(outpath + replaced_name + '_prompt_sim.npy', sims)
    
def get_paired_prompts(): 
    '''
    Input: list of runs with different gender neutral names
    
    The output here is a file containing
    two book titles + main character_storyIDx per line,
    separated by a tab character, with their prompts
    '''
    inpath = LOGS + 'prompt_matching/'
    fem_prompts = []
    with open(inpath + 'fem_prompt_order.txt', 'r') as infile: 
        reader = csv.reader(infile, delimiter='\t')
        for row in reader: 
            fem_prompts.append(row)
    masc_prompts = []
    with open(inpath + 'masc_prompt_order.txt', 'r') as infile: 
        reader = csv.reader(infile, delimiter='\t')
        for row in reader: 
            masc_prompts.append(row)
    sims = np.load(inpath + 'the_person_prompt_sim.npy')
        
    rank = np.argsort(sims, axis=1)
    count = 0
    already_seen = set() # masc prompt idx that have already been written
    with open(inpath + 'prompt_pairs.txt', 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for i in range(len(fem_prompts)): 
            # find closest neighbor that isn't already found
            start = -1
            while sims[i][rank[i][start]] >= 0.85: 
                if rank[i][start] not in already_seen: 
                    writer.writerow([sims[i][rank[i][-1]]] + fem_prompts[i] + masc_prompts[rank[i][start]])
                    already_seen.add(rank[i][start])
                    count += 1
                    break
                start -= 1
    print("Total pairs:", count)
    
def get_similarities(): 
    fem_prompts, masc_prompts = get_gendered_prompts()
    fem_prompts = sorted(fem_prompts)
    masc_prompts = sorted(masc_prompts)
    outpath = LOGS + 'prompt_matching/'
    with open(outpath + 'fem_prompt_order.txt', 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for tup in fem_prompts:
            title, story_idx, char, prompt = tup
            writer.writerow([title, story_idx, char, prompt])
    with open(outpath + 'masc_prompt_order.txt', 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for tup in masc_prompts:
            title, story_idx, char, prompt = tup
            writer.writerow([title, story_idx, char, prompt])
            
    new_f_prompts = []
    for tup in fem_prompts: 
        title, story_idx, char, prompt = tup
        prompt = prompt.replace(char, 'the person')
        if prompt.startswith('the person'): 
            prompt = 'T' + prompt[1:]
        new_f_prompts.append(prompt)

    new_m_prompts = []
    for tup in masc_prompts: 
        title, story_idx, char, prompt = tup
        prompt = prompt.replace(char, 'the person')
        if prompt.startswith('the person'): 
            prompt = 'T' + prompt[1:]
        new_m_prompts.append(prompt)
    print(len(new_f_prompts), len(new_m_prompts))
    get_embed_sim(new_f_prompts, new_m_prompts, 'the_person')

def get_same_prompt_diff_gender(): 
    num_gens = 5
    num_pairs = 0
    matched_pairs = {}
    for f in os.listdir(LOGS + 'original_prompts/'):
        if not os.path.exists(LOGS + 'char_gender_0.9/' + f + '.json'): continue
        with open(LOGS + 'char_gender_0.9/' + f + '.json', 'r') as infile: 
            gender_dict = json.load(infile)
        char_set = {} # main character name _ story ID : gender
        for char in gender_dict: 
            neighbor_dict = gender_dict[char]
            for neighbor in neighbor_dict: 
                is_main = False
                gender = neighbor['gender_label']
                neighbor_n = neighbor['character_name']
                if neighbor_n.startswith(char + '_'): 
                    # main character
                    is_main = True
                else: 
                    for al in neighbor['aliases']: 
                        if al == char: 
                            is_main = True
                if is_main: 
                    if len(neighbor['gender']) == 0: 
                        gender += ' (name)'
                    char_set[neighbor_n] = gender 
           
        story_idx = 0 
        
        with open(LOGS + 'original_prompts/' + f, 'r') as infile:
            matched_pairs[f] = [] 
            for line in infile: 
                contents = line.strip().split('\t')
                characterID = contents[0]
                char = contents[1]
                prompt = contents[2]
                if '-RRB-' in prompt or '-LRB-' in prompt: 
                    prompt = prompt.replace(' -RRB-', ')').replace('-LRB- ', '(')
                    prompt = prompt.replace('-RRB-', ')').replace('-LRB-', '(')
                genders = defaultdict(list)
                for idx in range(story_idx, story_idx + num_gens): 
                    char_ID = char + '_' + str(idx)
                    if char_set[char_ID] == 'masc' or char_set[char_ID] == 'fem': 
                        genders[char_set[char_ID]].append(char_ID)
                if len(genders) == 2: 
                    n_pairs = min(len(genders['masc']), len(genders['fem']))
                    matched_pairs[f].extend(random.sample(genders['masc'], n_pairs))
                    matched_pairs[f].extend(random.sample(genders['fem'], n_pairs))
                    num_pairs += n_pairs
                story_idx += num_gens
    with open(LOGS + 'prompt_matching/same_prompt_pairs.json', 'w') as outfile: 
        json.dump(matched_pairs, outfile)
    print(num_pairs)
    
def get_prompts_with_verbs(): 
    num_gens = 5
    lem = WordNetLemmatizer()
    # load verb lexicons
    power_verbs = set()
    with open(DATA + 'verbs/agency_power.csv', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split(',')
            if contents[2] == 'power_agent': 
                verb = lem.lemmatize(contents[0], 'v')
                power_verbs.add(verb)
    cognitive_verbs = set()
    with open(DATA + 'verbs/blooms_taxonomy', 'r') as infile: 
        for line in infile: 
            cognitive_verbs.add(lem.lemmatize(line.strip()))
    
    res = defaultdict(list)
    verb_freq = defaultdict(Counter)
    
    prompts_path = LOGS + 'original_prompts/'
    for title in os.listdir(prompts_path): 
        print(title)
        char_order = [] # character, where index is generated story index
        prompts = []
        with open(prompts_path + title, 'r') as infile: 
            reader = csv.reader(infile, delimiter='\t')
            for row in reader: 
                char_ID = row[0]
                char_name = row[1]
                prompt = row[2]
                prompts.extend([prompt]*num_gens)
                char_order.extend([char_name]*num_gens)
        if len(char_order) == 0: 
            # no prompts
            continue 
            
        # get first token in story that is char name 
        tokens2words = {} # token id: lemma, part of speech, story idx
        ret = []
        with open(LOGS + 'plaintext_stories_0.9_tokens/' + title + '.tokens', 'r') as infile: 
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            story_idx = 0
            dot_count = 0
            main_char_seen = False
            for row in reader:
                tid = row['tokenId']
                w = row['originalWord']
                pos = row['pos']
                sentenceID = row['sentenceID'] 
                tokens2words[tid] = (row['lemma'].lower(), pos, story_idx) 

                if row['normalizedWord'] == '@':
                    dot_count += 1
                else: 
                    dot_count = 0
                    # only get first instance of main character in story
                    if w == char_order[story_idx] and not main_char_seen: 
                        main_char_seen = True
                        if row['deprel'] == 'nsubj':
                            htid = row['headTokenId'] 
                            ret.append((w, tid, pos, row['deprel'], htid, story_idx)) 
                if dot_count == 20: 
                    story_idx += 1
                    main_char_seen = False
                    dot_count = 0
                    
        for tup in ret: 
            w, tid, pos, deprel, htid, story_idx = tup
            hw, hpos, hstory_idx = tokens2words[htid] 
            if story_idx != hstory_idx: continue
            if not hpos.startswith('VB'): continue
            if hw in power_verbs: 
                res['power'].append(title + '#' + w + '_' + str(story_idx))
                verb_freq['power'][hw] += 1
            if hw in cognitive_verbs: 
                res['cognitive'].append(title + '#' + w + '_' + str(story_idx))
                verb_freq['cognitive'][hw] += 1
    for label in verb_freq: 
        print("Total", label, sum(verb_freq[label].values()))
        print(verb_freq[label].most_common(10))
        
    with open(LOGS + 'prompt_design_verbs.json', 'w') as outfile: 
        json.dump(res, outfile)

def main(): 
    #get_similarities()
    #get_paired_prompts()
    #get_same_prompt_diff_gender()
    get_prompts_with_verbs()

if __name__ == "__main__":
    main()
