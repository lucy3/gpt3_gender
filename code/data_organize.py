import os
import json
import csv
import string

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
    
def get_stats(): 
    '''
    TODO: get # of characters, # of books with characters
    average length of prompt
    '''
    pass

def format_for_booknlp(gen_path, file_list, outpath): 
    ''' 
    Format generated stories for bookNLP
    '''
    files = []
    with open(file_list, 'r') as infile: 
        for filename in infile: 
            files.append(filename.strip())
    for i, filename in enumerate(files): 
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
        

def main(): 
    #sanity_check_outputs(LOGS + 'generated_0.9/', INPUTS)
    #format_for_booknlp(LOGS + 'generated_0.9/', LOGS + 'file_list', LOGS + 'plaintext_stories_0.9/')
    #get_stats()
    get_prompt_char_names()

if __name__ == "__main__":
    main()

