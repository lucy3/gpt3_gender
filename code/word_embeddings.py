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
    tl_fb = []
    sa_fa = [] 
    sa_fb = []
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
        tl_fb.append(turney_littman(train, test, 'intellectual', 'physical', model))
        # semaxis
        sa_fa.append(semaxis(train, test, 'strong', 'weak', model))
        sa_fb.append(semaxis(train, test, 'intellectual', 'physical', model))
    print(tl_fa, np.mean(tl_fa))
    print(tl_fb, np.mean(tl_fb))
    print(sa_fa, np.mean(sa_fa))
    print(sa_fb, np.mean(sa_fb))

def get_nouns_and_adj(inpath, outpath): 
    '''
    TODO: for original books, need to get book excerpt segment from logs/tokens
    '''
    # get all amod and nsubj
    gendered_pronouns = set(['he', 'He', 'she', 'She'])
    for f in os.listdir(inpath):
        # TODO: get mapping from named entities with story IDs to gender (char_gender jsons)
        # TODO: get mapping from token IDs to named entities (tokenID2ne)
        # TODO: the token ID mapping needs to be inclusive of all token IDs for that named entity
        tokens2words = {}
        ret = []
        print(f)
        with open(inpath + f, 'r') as infile: 
            reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_NONE)
            # TODO: calculate story id
            for row in reader:
                tid = row['tokenId']
                w = row['originalWord']
                pos = row['pos']
                tokens2words[tid] = (w, pos) 
                if row['deprel'] == 'amod':
                    htid = row['headTokenId'] 
                    ret.append((w, tid, pos, row['deprel'], htid)) 
                elif row['deprel'] == 'nsubj':
                    w_storyID = w + '_' + str(story_ID)
                    htid = row['headTokenId'] 
                    # first column needs to be pronoun or named entity
                    if w in gendered_pronouns or tid in tokenID2ne: 
                        ret.append((w, tid, pos, row['deprel'], htid))
        with open(outpath + f.replace('.tokens', ''), 'w') as outfile: 
            for tup in ret: 
                # TODO: last column should be inferred gender 
                w, tid, pos, deprel, htid = tup
                hw = tokens2words[htid][0]
                hpos = tokens2words[htid][1]
                if deprel == 'nsubj': 
                    # the head needs to be an adj or verb
                    if hpos != 'JJ' and not hpos.startswith('VB'): continue
                if deprel == 'amod': 
                    # the head must be a named entity
                    if htid not in tokenID2ne:
                        continue  
                outfile.write(w + '\t' + tid + '\t' + pos + '\t' + deprel + '\t' + htid + '\t' + \
                    hw + '\t' + hpos + '\n')


def main(): 
    #preprocess_text()
    #train_embeddings()
    #play_with_lexicon_words()
    #evaluate_lexicon_induction()
    get_nouns_and_adj(LOGS + 'plaintext_stories_0.9_tokens/', LOGS + 'generated_adj_noun/')

if __name__ == "__main__":
    main()
