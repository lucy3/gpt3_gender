import json
import os
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv 

LOGS = '/mnt/data0/lucy/gpt3_bias/logs/'

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
            genders = set()
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
                    if 'masc' in genders: 
                        if '-RRB-' in prompt or '-LRB-' in prompt: 
                            s = prompt.replace(' -RRB-', ')').replace('-LRB- ', '(')
                            s = s.replace('-RRB-', ')').replace('-LRB-', '(')
                        masc_prompts.add((f, story_idx, char, prompt))
                    elif 'fem' in genders: 
                        if '-RRB-' in prompt or '-LRB-' in prompt: 
                            s = prompt.replace(' -RRB-', ')').replace('-LRB- ', '(')
                            s = s.replace('-RRB-', ')').replace('-LRB-', '(')
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
    
def get_paired_prompts(replaced_names): 
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
    avg_sims = np.zeros((4748, 8044))
    for r_name in replaced_names: 
        sims = np.load(inpath + r_name + '_prompt_sim.npy')
        avg_sims += sims
    avg_sims = avg_sims / len(replaced_names)
        
    rank = np.argsort(avg_sims, axis=1)
    count = 0
    already_seen = set() # masc prompt idx that have already been written
    with open(inpath + 'prompt_pairs.txt', 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for i in range(len(fem_prompts)): 
            # find closest neighbor that isn't already found
            start = -1
            while avg_sims[i][rank[i][start]] >= 0.85: 
                if rank[i][start] not in already_seen: 
                    writer.writerow([avg_sims[i][rank[i][-1]]] + fem_prompts[i] + masc_prompts[rank[i][start]])
                    already_seen.add(rank[i][start])
                    count += 1
                    break
                start -= 1
    print("Total pairs:", count)
    
def get_similarities(replaced_names): 
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
    for name in replaced_names: 
        print(name)
        new_f_prompts = []
        for tup in fem_prompts: 
            title, story_idx, char, prompt = tup
            prompt = prompt.replace(char, name)
            new_f_prompts.append(prompt)
        
        new_m_prompts = []
        for tup in masc_prompts: 
            title, story_idx, char, prompt = tup
            prompt = prompt.replace(char, name)
            new_m_prompts.append(prompt)
        get_embed_sim(new_f_prompts, new_m_prompts, name)

def main(): 
    replaced_names = ['Quinn', 'Riley', 'Kris', 'Lennon', 'Robbie']
    #get_similarities(replaced_names)
    get_paired_prompts(replaced_names)

if __name__ == "__main__":
    main()
