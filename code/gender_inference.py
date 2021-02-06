"""
The input, char_neighbors_0.9, has each main character
and the characters it co-occurs with, as well as the
gender of pronouns associated with that character
"""
import os
from collections import Counter, defaultdict
import json

LOGS = '/mnt/data0/lucy/gpt3_bias/logs/'
NAMES = '/mnt/data0/lucy/gpt3_bias/data/names/'

def get_name_gender(neighbor, name_ratios):
    # http://self.gutenberg.org/articles/eng/english_honorifics
    gender = None
    honorifics_fem = ['Ms.', 'Mz.', 'Ms', 'Mz', 'Miss', 'Mrs', 'Mrs.', 'Madam', 
                  'Ma\'am', 'Dame', 'Lady', 'Mistress', 'Aunt', 
                  'Grandma', 'Grandmother', 'Queen', 'Empress', 'Princess']
    honorifics_masc = ['Mr.', 'Mister', 'Mr', 'Master', 'Sir', 'Lord', 'Esq', 'Esq.', 'Br.', 'Br', 'Brother',
                'Fr', 'Fr.', 'Father', 'Uncle', 'Grandpa', 'Grandfather', 'King', 'Emperor', 'Prince']
    # honorifics 
    for al in neighbor['aliases']: 
        if al.split()[0] in honorifics_fem: 
            if gender == 'masc' or gender == 'other': 
                # already assigned a different gender
                gender = 'other'
            else: 
                gender = 'fem'
        elif al.split()[0] in honorifics_masc: 
            if gender == 'fem' or gender == 'other': 
                # already assigned a different gender
                gender = 'other'
            else: 
                gender = 'masc'
    if gender is not None: 
        return gender
    # baby names
    for al in neighbor['aliases']:
        fn = al.split()[0] # get first token
        if fn not in name_ratios: continue
        if name_ratios[fn] > 0.90: 
            if gender == 'masc' or gender == 'other': 
                # already assigned a different gender
                gender = 'other'
            else: 
                gender = 'fem'
        elif name_ratios[fn] < 0.10: 
            if gender == 'fem' or gender == 'other': 
                # already assigned a different gender
                gender = 'other'
            else: 
                gender = 'masc'
    return gender
    
def get_baby_name_ratios(): 
    name_f = Counter()
    name_m = Counter()
    for i in range(1900, 2020): 
        with open(NAMES + 'yob' + str(i) + '.txt', 'r') as infile: 
            for line in infile: 
                contents = line.strip().split(',')
                if contents[1] == 'F': 
                    name_f[contents[0]] += int(contents[2])
                elif contents[1] == 'M': 
                    name_m[contents[0]] += int(contents[2])
                else: 
                    print(contents[1]) # prints nothing
    name_ratios = {}
    all_names = set(name_f.keys()) | set(name_m.keys())
    for name in all_names: 
        name_ratios[name] = name_f[name] / float(name_m[name] + name_f[name])
    return name_ratios

def infer_gender(char_neighbor_path, outpath): 
    '''
    Ways to infer GPT-3's perception of a character's gender
    - pronoun
    - honorifics
    - baby name list
    '''
    name_ratios = get_baby_name_ratios()
    other_gender = 0 # number of characters w/ multiple gender pronouns
    no_pronouns = 0 # number of characters w/ no pronouns
    unknown_count = 0 # number of characters w/ no pronouns and name isn't very gendered
    total_char = 0
    for title in os.listdir(char_neighbor_path):
        with open(char_neighbor_path + title, 'r') as infile: 
            char_neighbors = json.load(infile)
        for char in char_neighbors: 
            # for every main character
            neighbor_dict = char_neighbors[char]
            for neighbor in neighbor_dict: 
                total_char += 1
                # for every character it co-occurs with 
                neighbor_n = neighbor['character_name']
                name = '_'.join(neighbor_n.split('_')[:-1])
                neighbor['aliases'].append(name)
                pns = Counter(neighbor['gender'])
                total = sum(list(pns.values()))
                gender = None
                if pns['masc'] > 0.75*total: 
                    gender = 'masc'
                elif pns['fem'] > 0.75*total: 
                    gender = 'fem'
                elif total > 0: 
                    gender = 'other'
                if gender is None: 
                    no_pronouns += 1
                    gender = get_name_gender(neighbor, name_ratios)
                if gender is None: 
                    unknown_count += 1
                    gender = 'other'
                if gender == 'other':
                    other_gender += 1
                neighbor['gender_label'] = gender
        with open(outpath + title, 'w') as outfile: 
            json.dump(char_neighbors, outfile)
    print("Total characters:", total_char)
    print("No pronouns:", no_pronouns) 
    print("Other gender:", other_gender)
    print("Unknown name, no pronouns:", unknown_count)
    
def multi_gender_chars(): 
    '''
    Seeing if some characters are gendered differently across different stories
    '''
    char_gender_path = LOGS + 'char_gender_0.9/'
    for title in os.listdir(char_gender_path): 
        with open(char_gender_path + title, 'r') as infile: 
            char_gender = json.load(infile)
        for char in char_gender: 
            neighbor_dict = char_gender[char]
            genders = Counter()
            pronouns = []
            for neighbor in neighbor_dict: 
                is_main = False
                neighbor_n = neighbor['character_name']
                if neighbor_n.startswith(char + '_'): 
                    # main character
                    is_main = True
                else: 
                    for al in neighbor['aliases']: 
                        if al.startswith(char + '_'): 
                            is_main = True
                if is_main: 
                    genders[neighbor['gender_label']] += 1
                    pronouns.append(neighbor['gender'])
            if 'masc' in genders and 'fem' in genders: 
                print(char, title, genders)
    
def main():
    generated = True
    if generated: 
        outpath = LOGS + 'char_gender_0.9/'
        char_neighbor_path = LOGS + 'char_neighbors_0.9/'
    else: 
        outpath = LOGS + 'orig_char_gender/'
        char_neighbor_path = LOGS + 'orig_char_neighbors/'
    infer_gender(char_neighbor_path, outpath)
    

if __name__ == "__main__":
    main()
