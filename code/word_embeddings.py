import os
import csv
import string
from gensim.models.word2vec import PathLineSentences
from gensim.models import Word2Vec
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec
from collections import defaultdict, Counter
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

ROOT = '/mnt/data0/lucy/gpt3_bias/'
LOGS = ROOT + 'logs/'

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

    def on_train_begin(self, model): 
        print("Training start")

    def on_train_end(self, model): 
        print("Training end")

def read_stereotypes(): 
    inpath = '/mnt/data0/corpora/lexicons/fast_icwsm_2016_gender_stereotypes.csv'
    men = set(['active', 'angry', 'arrogant', 'dominant', 'sexual', 'strong', 'violent'])
    women = set(['afraid', 'beautiful', 'childish', 'dependent', 
          'domestic', 'emotional', 'hysterical', 'submissive', 'weak'])
    men_d = defaultdict(set)
    women_d = defaultdict(set)
    with open(inpath, 'r') as infile: 
        reader = csv.DictReader(infile, delimiter=',')
        for row in reader: 
            for key in row:
                if row[key] == '': continue 
                if key in men: 
                    men_d[key].add(row[key])
                elif key in women: 
                    women_d[key].add(row[key])
    inpath = '/mnt/data0/corpora/lexicons/empath_categories.tsv'
    with open(inpath, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            if 'intellectual' in contents: 
                for w in contents: 
                    men_d['intellectual'].add(w)
    for key in men_d: 
        men_d[key].add(key)
    for key in women_d: 
        women_d[key].add(key) 
    return men_d, women_d

def preprocess_text(): 
    '''
    cleans text
    lowercases
    removes numbers and punctuation 
    returns a list of sentences, tokens seperated by white space
    '''
    outpath = LOGS + 'word2vec_train_data/' 

    # read in generated stories
    genpath = LOGS + 'plaintext_stories_0.9_tokens/'
    for f in os.listdir(genpath): 
        new_f = 'GEN_' + f.replace('.tokens', '')
        print(new_f)
        outfile = open(outpath + new_f, 'w') 
        with open(genpath + f, 'r') as infile: 
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            currSentID = -1
            currSent = []
            for row in reader: 
                if row['deprel'] != 'punct' and row['originalWord'] != '@': 
                    word = row['originalWord'].lower()
                    if word in string.punctuation: continue 
                    if row['sentenceID'] != currSentID: 
                        outfile.write(' '.join(currSent) + '\n') 
                        currSent = []
                    currSentID = row['sentenceID']
                    currSent.append(word)
            outfile.write(' '.join(currSent) + '\n') 
        outfile.close()

    # read in books
    bookpath = LOGS + 'tokens/'
    for f in os.listdir(bookpath): 
        new_f = 'ORIG_' + f
        print(new_f)
        outfile = open(outpath + new_f, 'w') 
        with open(bookpath + f, 'r') as infile:
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            currSentID = -1
            currSent = []
            for row in reader: 
                if row['deprel'] != 'punct': 
                    word = row['originalWord'].lower()
                    if word in string.punctuation: continue
                    if row['sentenceID'] != currSentID: 
                        outfile.write(' '.join(currSent) + '\n') 
                        currSent = []
                    currSentID = row['sentenceID']
                    currSent.append(word)
            outfile.write(' '.join(currSent) + '\n')
        outfile.close() 


def train_embeddings(): 
    inpath = LOGS + 'word2vec_train_data/'
    sentences = PathLineSentences(inpath)
    epoch_logger = EpochLogger()
    print("Starting word2vec....")
    model = Word2Vec(sentences, size=100, window=5, min_count=5, 
          workers=multiprocessing.cpu_count(), seed=0, callbacks=[epoch_logger])
    model.save(LOGS + 'fiction_word2vec_model')

def play_with_lexicon_words(): 
    men_d, women_d = read_stereotypes()
    model = Word2Vec.load(LOGS + 'fiction_word2vec_model')
    c = 0
    nc = 0
    for k in men_d: 
        for w in men_d[k]: 
            if w in model.wv.vocab: 
                c += 1
            else: 
                nc += 1
    for k in women_d: 
        for w in women_d[k]: 
            if w in model.wv.vocab: 
                c += 1
            else: 
                nc += 1
    print("Found:", c, "..... Missing:", nc)
    # get categories with biggest overlap
    lexicons = {**men_d, **women_d}
    sim = Counter()
    for cat1 in lexicons: 
        for cat2 in lexicons: 
            if cat1 != cat2 and (cat2, cat1) not in sim:  
                sim[(cat1, cat2)] = len(lexicons[cat1] & lexicons[cat2])
    print(sim.most_common(20)) 
    # check overlap of proposed axes
    axes_A1 = ['strong', 'dominant']
    axes_A2 = ['weak', 'dependent', 'submissive', 'afraid']
    a = lexicons['strong'] | lexicons['dominant']
    b = set()
    for cat in axes_A2: 
        b |= lexicons[cat]
    print(a & b)
    print("Similarity between words weak and strong:", model.similarity('weak', 'strong'))
    axes_B1 = ['beautiful', 'sexual']
    axes_B2 = ['intellectual']
    print((lexicons['beautiful'] | lexicons['sexual']) & lexicons['intellectual'])

def get_axes(): 
    men_d, women_d = read_stereotypes()
    lexicons = {**men_d, **women_d}
    axes_A1 = ['strong', 'dominant']
    axes_A2 = ['weak', 'dependent', 'submissive', 'afraid']
    axes_B1 = ['beautiful', 'sexual']
    axes_B2 = ['intellectual']
    lexicon_d  = defaultdict(set)
    lexicon_d['strong'] = lexicons['strong'] | lexicons['dominant']
    for cat in axes_A2: 
        lexicon_d['weak'] |= lexicons[cat]
    lexicon_d['physical'] = lexicons['beautiful'] | lexicons['sexual']
    lexicon_d['intellectual'] = lexicons['intellectual']
    for k in lexicon_d: 
        print(k, len(lexicon_d[k])) 
    return lexicon_d

def get_matrices(train, test, pole1, pole2, model): 
    train_A = []
    for w in train[pole1]: 
        if w in model.wv.vocab: 
            train_A.append(model.wv[w])
    train_A = np.array(train_A)
    train_B = []
    for w in train[pole2]: 
        if w in model.wv.vocab: 
            train_B.append(model.wv[w])
    train_B = np.array(train_B)
    test_A = []
    for w in test[pole1]: 
        if w in model.wv.vocab: 
            test_A.append(model.wv[w])
    test_A = np.array(test_A)
    test_B = []
    for w in test[pole2]: 
        if w in model.wv.vocab: 
            test_B.append(model.wv[w])
    test_B = np.array(test_B)
    return train_A, train_B, test_A, test_B

def turney_littman(train, test, pole1, pole2, model): 
    '''
    pole1 is the minority class, pole2 is the majority class
    '''
    train_A, train_B, test_A, test_B = get_matrices(train, test, pole1, pole2, model)
    diff = np.sum(cosine_similarity(test_A, train_A), axis=1) - \
           np.sum(cosine_similarity(test_A, train_B), axis=1)
    # count how many diffs are positive
    tp = np.sum(np.array(diff) >= 0) 
    fn = np.sum(np.array(diff) < 0) 
    diff = np.sum(cosine_similarity(test_B, train_A), axis=1) - \
           np.sum(cosine_similarity(test_B, train_B), axis=1)
    tn = np.sum(np.array(diff) <= 0)
    fp = np.sum(np.array(diff) > 0) 
    # count how many diffs are negative
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2*(p*r)/(p+r)
    return f1

def semaxis(train, test, pole1, pole2, model):
    train_A, train_B, test_A, test_B = get_matrices(train, test, pole1, pole2, model)
    v = np.mean(train_A, axis=0) - np.mean(train_B, axis=0).reshape(1, -1)
    score = cosine_similarity(test_A, v)
    tp = np.sum(score >= 0)
    fn = np.sum(score < 0)
    score = cosine_similarity(test_B, v)
    tn = np.sum(score <= 0)
    fp = np.sum(score > 0)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2*(p*r)/(p+r)
    return f1
     
def evaluate_lexicon_induction(): 
    '''
    With the two stereotype axes: strong/weak and beauty/intellect

    turney & littman: sum_A cos(x, a) - sum_B cos(x, b)
    semaxis: cos(x, avg(A) - avg(B))
    '''
    random.seed(0)
    model = Word2Vec.load(LOGS + 'fiction_word2vec_model')
    lexicons = get_axes()
    # random splits
    num_splits = 10
    lexicon_shuffled = defaultdict(list)
    for k in lexicons: 
        l = list(lexicons[k])
        random.shuffle(l)
        if len(l) % num_splits == 0: 
            chunk_size = int(len(l) / num_splits)
        else: 
            chunk_size = int(len(l) / num_splits) + 1
        chunks = [l[x:x+chunk_size] for x in range(0, len(l), chunk_size)]
        lexicon_shuffled[k] = chunks
    tl_fa = []
    sa_fa = [] 
    for i in range(num_splits): 
        train = defaultdict(set)
        test = defaultdict(set) 
        for k in lexicon_shuffled: 
            for j in range(num_splits): 
                if j != i: 
                    train[k].update(lexicon_shuffled[k][j])
            test[k].update(lexicon_shuffled[k][i])
        # turney & littman
        tl_fa.append(turney_littman(train, test, 'strong', 'weak', model))
        # semaxis
        sa_fa.append(semaxis(train, test, 'strong', 'weak', model))
    print(tl_fa, np.mean(tl_fa))
    print(sa_fa, np.mean(sa_fa))

def get_nouns_and_adj(inpath, outpath, ents_path, gender_path): 
    '''
    inpath is tokens
    ents_path is entities 
    '''
    # get all amod and nsubj
    gendered_pronouns = set(['he', 'He', 'she', 'She'])
    for f in os.listdir(inpath):
        print(f)
        bookname = f.replace('.tokens', '')
        if not os.path.exists(gender_path + bookname + '.json'): continue
        with open(gender_path + bookname + '.json', 'r') as infile: 
            gender_dict = json.load(infile)
        # mapping from named entities w/ story IDs to gender 
        name2gender = {}
        for char in gender_dict: 
            neighbors = gender_dict[char]
            for neighbor in neighbors:
                gender = neighbor['gender_label']
                base_char = neighbor['character_name']
                story_idx = base_char.split('_')[-1]
                name2gender[base_char] = gender
                for alias in neighbor['aliases']: 
                    name2gender[alias + '_' + story_idx] = gender
        
        # set of all token IDs that are named entities 
        ne_tokens = {}
        with open(ents_path + bookname + '/' + bookname + '.ents', 'r') as infile: 
            for line in infile: 
                contents = line.strip().split('\t')
                start = int(contents[0])
                end = int(contents[1])
                ner = contents[2]
                entity = contents[3]
                if ner == 'PROP_PER' or entity in gender_dict: 
                    for i in range(start, end+1): 
                        ne_tokens[i] = entity

        tokens2words = {}
        ret = []
        with open(inpath + f, 'r') as infile: 
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            story_idx = 0
            dot_count = 0
            for row in reader:
                tid = row['tokenId']
                w = row['originalWord']
                pos = row['pos']
                tokens2words[tid] = (w, pos, story_idx) 

                if row['normalizedWord'] == '@':
                    dot_count += 1
                else: 
                    dot_count = 0
                    if row['deprel'] == 'amod' or row['deprel'] == 'nsubj':
                        htid = row['headTokenId'] 
                        ret.append((w, tid, pos, row['deprel'], htid, story_idx)) 
                if dot_count == 20: 
                    story_idx += 1
                    dot_count = 0
        with open(outpath + f.replace('.tokens', ''), 'w') as outfile: 
            for tup in ret: 
                w, tid, pos, deprel, htid, story_idx = tup
                hw, hpos, hstory_idx = tokens2words[htid]
                # dep parse shouldn't cross stories 
                if story_idx != hstory_idx: continue
                gender = None
                if deprel == 'nsubj': 
                    # the head needs to be an adj or verb
                    if hpos != 'JJ' and not hpos.startswith('VB'): continue
                    # the word needs to be a pronoun or named entity
                    if w.lower() == 'he':
                        gender = 'masc'
                    if w.lower() == 'she': 
                        gender = 'fem'
                    if int(tid) in ne_tokens: 
                        gender = name2gender[ne_tokens[int(tid)] + '_' + str(story_idx)]
                if deprel == 'amod': 
                    # the head must be a named entity
                    if int(htid) in ne_tokens:
                        gender = name2gender[ne_tokens[int(htid)] + '_' + str(story_idx)]
                if gender == 'masc' or gender == 'fem': 
                    outfile.write(w + '\t' + tid + '\t' + pos + '\t' + deprel + '\t' + htid + '\t' + \
                        hw + '\t' + hpos + '\t' + str(story_idx) + '\t' + gender + '\n')

def update_gen_word(gen_word, line): 
    contents = line.strip().split('\t')
    w = contents[0]
    deprel = contents[3]
    hw = contents[5]
    gender = contents[8]
    if deprel == 'amod': 
        gen_word[gender].append(w)
    elif deprel == 'nsubj': 
        gen_word[gender].append(hw)
    return gen_word

def get_sim_score(lexicon, all_words_m, model): 
    m = []
    for lw in lexicon: 
        if lw in model.wv.vocab: 
            m.append(model.wv[lw])
    m = np.array(m)
    sims = cosine_similarity(all_words_m, m)
    return np.mean(sims, axis=1)

def get_semaxis_score(pole1, pole2, all_words_m, model):
    pole_A = []
    pole_B = []
    for lw in pole1: 
        if lw in model.wv.vocab: 
            pole_A.append(model.wv[lw])
    for lw in pole2: 
        if lw in model.wv.vocab: 
            pole_B.append(model.wv[lw])
    pole_A = np.array(pole_A)
    pole_B = np.array(pole_B)
    v = np.mean(pole_A, axis=0) - np.mean(pole_B, axis=0).reshape(1, -1)
    score = cosine_similarity(all_words_m, v)
    return score

def get_lexicon_scores(): 
    '''
    How many words are in each lexicon? 
    '''
    # get generated words
    genpath = LOGS + 'generated_adj_noun/'
    gen_word = {'fem':[], 'masc':[]}
    for f in os.listdir(genpath):
        with open(genpath + f, 'r') as infile: 
            for line in infile: 
                gen_word = update_gen_word(gen_word, line)
    # get excerpt words
    book_word = {'fem':[], 'masc':[]}
    origpath = LOGS + 'orig_adj_noun/'
    for f in os.listdir(origpath):  
        with open(origpath + f, 'r') as infile: 
            for line in infile: 
                gen_word = update_gen_word(gen_word, line)
    # get lexicons
    lexicons = get_axes()
    lexicon_words = lexicons['strong'] | lexicons['intellectual'] | lexicons['weak'] | lexicons['physical']
    # calculate overlap
    all_words = set()
    all_words.update(gen_word['fem'])
    all_words.update(book_word['fem'])
    all_words.update(gen_word['masc'])
    all_words.update(book_word['masc'])
    overlap = lexicon_words & all_words
    print("All words in dataset:", len(all_words))
    print("Size of overlap with lexicon:", len(overlap))
    
    # calculate average of cosine similarities of a word to all words in lexicon
    model = Word2Vec.load(LOGS + 'fiction_word2vec_model')
    all_words_m = []
    sorted_words = []
    for w in all_words: 
        w = w.lower()
        if w in model.wv.vocab: 
            all_words_m.append(model.wv[w])
            sorted_words.append(w)
    all_words_m = np.array(all_words_m)
    # these are in the order of sorted(all_words)
    s = get_sim_score(lexicons['intellectual'], all_words_m, model)
    assert s.shape[0] == len(sorted_words)
    
    intellect_scores = {}
    physical_scores = {}
    strength_scores = {}
    for i, w in enumerate(sorted_words): 
        intellect_scores[w] = float(s[i])
    s = get_sim_score(lexicons['physical'], all_words_m, model)
    for i, w in enumerate(sorted_words): 
        physical_scores[w] = float(s[i])
    s = get_semaxis_score(lexicons['strong'], lexicons['weak'], all_words_m, model)
    for i, w in enumerate(sorted_words): 
        strength_scores[w] = float(s[i])
        
    with open(LOGS + 'intellect_scores.json', 'w') as outfile: 
        json.dump(intellect_scores, outfile)
    with open(LOGS + 'physical_scores.json', 'w') as outfile: 
        json.dump(physical_scores, outfile)
    with open(LOGS + 'strength_scores.json', 'w') as outfile: 
        json.dump(strength_scores, outfile)

def main(): 
    #preprocess_text()
    #train_embeddings()
    play_with_lexicon_words()
    #evaluate_lexicon_induction()
    generated = True
    if generated: 
        ents_path = LOGS + 'generated_0.9_ents/'
        tokens_path = LOGS + 'plaintext_stories_0.9_tokens/'
        gender_path = LOGS + 'char_gender_0.9/'
        outpath = LOGS + 'generated_adj_noun/'
    else: 
        ents_path = LOGS + 'book_excerpts_ents/' 
        tokens_path = LOGS + 'book_excerpts_tokens/'
        gender_path = LOGS + 'orig_char_gender/'
        outpath = LOGS + 'orig_adj_noun/' 
    #get_nouns_and_adj(tokens_path, outpath, ents_path, gender_path)
    #get_lexicon_scores()

if __name__ == "__main__":
    main()
