import os
import json
import csv
import string
from collections import Counter
from nltk.tokenize import sent_tokenize

ROOT = "/mnt/data0/lucy/gpt3_bias/"
LOGS = ROOT + 'logs/'

INPUTS = LOGS + 'original_prompts/' 

def sanity_check_outputs(gen_path, input_path): 
    '''
    For each file in gen_path, count the number of prompts and number of generations for each
    Count the number of prompts in input_path
    '''
    files = os.listdir(gen_path)
    total_prompts_run = 0
    for i, filename in enumerate(files): 
        num_lines = 0
        with open(gen_path + filename, 'r') as infile: 
            for line in infile: 
                num_lines += 1
                d = json.loads(line)
                assert len(d['choices']) == 5
        bookname = filename.replace('.json', '')
        num_prompts = 0
        with open(input_path + bookname, 'r') as infile: 
            num_prompts += len(infile.readlines())
        assert num_lines == num_prompts
        total_prompts_run += num_prompts
    print(total_prompts_run)


def sanity_check_redo_outputs(gen_path, input_path): 
    '''
    In this new format, each dictionary now represents a single
    generation instead of five. So, we check that there are in fact
    five generations per prompt, and that all prompts are accounted
    for. 
    '''
    files = os.listdir(gen_path)
    total_prompts_run = 0
    for i, filename in enumerate(files): 
        gen_count = Counter() # input : num of generations
        with open(gen_path + filename, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                gen_count[d['input']] += 1
        for input_text in gen_count:
            # there is one input with 10 generations due to bookNLP error
            assert gen_count[input_text] % 5 == 0 
        num_lines = sum(list(gen_count.values()))
        bookname = filename.replace('.json', '')
        num_prompts = 0
        with open(input_path + bookname, 'r') as infile: 
            num_prompts += len(infile.readlines())
        if len(gen_count) != num_prompts: print(filename, len(gen_count), num_prompts)
        total_prompts_run += num_prompts
    print(total_prompts_run)

def replace_bad_outputs(redo_gen_path, old_gen_path, outpath): 
    '''
    old_generated_0.9
    '''
    for i, filename in enumerate(os.listdir(redo_gen_path)): 
        new_gens = [] # list of dictionaries 
        with open(redo_gen_path + filename, 'r') as infile:
            curr_dict = {}
            for line in infile: 
                d = json.loads(line)
                if curr_dict == {}:
                    curr_dict = d
                elif len(curr_dict['choices']) < 5:
                    assert d['input'] == curr_dict['input']
                    curr_dict['choices'].append(d['choices'][0])
                else: 
                    # add current dictionary to new_gens
                    new_gens.append(curr_dict)
                    # start new dictionary
                    curr_dict = d
            if curr_dict != {}: 
                new_gens.append(curr_dict)
        bad_prompts = 0
        outfile = open(outpath + filename, 'w') 
        with open(old_gen_path + filename, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                prompt = d['input']
                if '-RRB-' in prompt or '-LRB-' in prompt:
                    print(d['input'])
                    print(new_gens[bad_prompts]['input'])
                    print()
                    outfile.write(json.dumps(new_gens[bad_prompts]) + '\n')
                    bad_prompts += 1
                else: 
                    outfile.write(json.dumps(d) + '\n')
        outfile.close()
        assert bad_prompts == len(new_gens) 
        # write out non-problematic prompts to outpath
        # write redo prompts to same outpath
    
def format_for_booknlp(gen_path, outpath): 
    ''' 
    Format generated stories for bookNLP
    '''
    for filename in os.listdir(gen_path): 
        num_lines = 0
        bookname = filename.replace('.json', '')
        output_file = open(outpath + bookname, 'w') 
        with open(gen_path + filename, 'r') as infile: 
            for line in infile: 
                num_lines += 1
                d = json.loads(line)
                for j in range(len(d['choices'])): 
                    text = d['choices'][j]['text']
                    input_text = d['input']
                    all_text = input_text.strip() + ' ' + text.strip()
                    # put in a divider to discourage coref and dep parse from crossing between stories
                    output_file.write(all_text + 
                                      '\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n')
        output_file.close()

def get_prompt_char_names(): 
    names = set()
    for f in os.listdir(LOGS + 'original_prompts/'): 
        with open(LOGS + 'original_prompts/' + f, 'r') as infile: 
            reader = csv.reader(infile, delimiter='\t')
            for row in reader: 
                char_name = row[1].lower().translate(str.maketrans('', '', string.punctuation))
                names.add(char_name)
    with open(LOGS + 'prompt_char_names.txt', 'w') as outfile: 
        for n in names: 
            outfile.write(n + ' ')
        
def examine_generated_book_overlap(gen_path): 
    '''
    Carlini et al. (2021) has shown that GPT-3 can memorize books (and potentially
    copyrighted materials). This allows us to see what materials
    we should not disseminate further. 
    
    For each generated story, we break it down into 2 sentence chunks, and see if
    the original books contain those two consecutive sentences. 
    '''
    files = os.listdir(gen_path)
    lengths = []
    for filename in files: 
        bookname = filename.replace('.json', '')
        gen_sentence_pairs = []
        with open(gen_path + filename, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                lengths.append(len(d['input'].split()))
                for j in range(len(d['choices'])): 
                    text = d['choices'][j]['text']
                    sentences = sent_tokenize(text)
                    for i in range(len(sentences) - 1): 
                        gen_sentence_pairs.append((sentences[i].lower(), sentences[i + 1].lower()))
        orig_sentence_pairs = []
        with open(ROOT + 'data/originals/' + bookname + '.txt', 'r') as infile: 
            for line in infile: 
                sentences = sent_tokenize(line.strip())
                for i in range(len(sentences) - 1): 
                    orig_sentence_pairs.append((sentences[i].lower(), sentences[i + 1].lower()))
        for pair in orig_sentence_pairs: 
            if pair in gen_sentence_pairs: 
                if len(pair[0]) > 1 and len(pair[1]) > 1: 
                    print(filename, pair)
    print(sum(lengths)/len(lengths))
    print(max(lengths))
    
def examine_generated_book_overlap2(gen_path): 
    '''
    In this version, we show 5-grams from each book are
    repeated in generated text 
    
    The script passes over each book and for every 5-gram window,
    checks if it is in the generated stories for that book. 
    It prints out every 5+-gram (continuous ngram that is at least 5 tokens long that
    appears in the generated stories) and also prints the number of tokens "|#|" that
    do not appear in generated stories between these ngrams. 
    
    This is to give a sense of just how closely the ngrams are in the original text,
    in case GPT-3 is outputting sentences that are series of copied ngrams stitched
    together by small differences. 
    Cases where you see \"| |\" are where different parts of a phrase are overlapping ngrams 
    that each appear in generated stories, but the entire phrase itself that contains 
    these ngrams does not appear in generated stories
    After printing ngrams for a book, I print out the number of non-copied areas in 
    this book that have less than 5 non-copied tokens in them (aka where the # in each 
    |#| is less than 5). 
    '''
    files = os.listdir(gen_path)
    for filename in files: 
        print(filename)
        bookname = filename.replace('.json', '')
        gen_sentence_grams = set()
        with open(gen_path + filename, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                for j in range(len(d['choices'])): 
                    text = d['choices'][j]['text'].lower().split()
                    for i in range(len(text) - 4): 
                        gen_sentence_grams.add(tuple(text[i:i+5]))
        with open(ROOT + 'data/originals/' + bookname + '.txt', 'r') as infile: 
            non_frag_tokens = 0
            print_out = ''
            consec_frags = 0
            for line in infile: 
                start = 0
                prev = False
                text = line.strip().lower().split()
                for i in range(len(text) - 4): 
                    gram = tuple(text[i:i+5])
                    if gram in gen_sentence_grams: 
                        if non_frag_tokens > 5: 
                            print_out += " | " + str(non_frag_tokens - 4) + " | "
                            if non_frag_tokens - 4 < 5: 
                                consec_frags += 1
                        elif non_frag_tokens <= 5 and non_frag_tokens > 0: 
                            print_out += " | | "
                        non_frag_tokens = 0
                        prev = True
                    else: 
                        non_frag_tokens += 1
                        if prev: 
                            print_out += ' '.join(text[start:i+4])
                        prev = False
                        start = i + 1
                if prev: 
                    print_out += ' '.join(text[start:i+5])
                else: 
                    non_frag_tokens += 4
            print(print_out)
            print("NUM OF SHORT PAUSES:", consec_frags, filename)
        print()
        break

def main(): 
    #sanity_check_outputs(LOGS + 'generated_0.9/', INPUTS)
    #sanity_check_redo_outputs(LOGS + 'redo_0.9/', LOGS + 'redo_prompts/')
    #replace_bad_outputs(LOGS + 'redo_0.9/', LOGS + 'old_generated_0.9/', 
    #    LOGS + 'generated_0.9/')
    #format_for_booknlp(LOGS + 'generated_0.9/', LOGS + 'plaintext_stories_0.9/')
    #get_prompt_char_names()
    #examine_generated_book_overlap(LOGS + 'generated_0.9/')
    examine_generated_book_overlap2(LOGS + 'generated_0.9/')

if __name__ == "__main__":
    main()

