"""
Queries OpenAI
"""
import os
import openai
import csv
import json
import time
import socket

ORIGINAL = '/mnt/data0/lucy/gpt3_bias/logs/original_prompts/'
REDO = '/mnt/data0/lucy/gpt3_bias/logs/redo_prompts/'
LOGS = '/mnt/data0/lucy/gpt3_bias/logs/'

openai.api_key = # HIDDEN

def get_response(prompt, temp, n_gens):
   response = openai.Completion.create(
	  engine="davinci",
	  prompt=prompt,
	  temperature=temp,
	  max_tokens=1800,
	  top_p=1,
          n=n_gens
   )
   print("**Response complete!**")
   return response

def query_for_stories(in_folder, out_prefix, log_file_path, temp): 
   '''
   We try a temperature of 0.9.
   Due to rate limits this has been mofidied to only get 1 generation 
   per try instead of 5 generations at once.  
   '''
   already_done = set()
   with open(log_file_path, 'r') as infile: 
       for line in infile: 
           already_done.add(line.strip())
   n_gens = 5
   logfile = open(log_file_path, 'a')
   for f in os.listdir(in_folder):
       if f in already_done: continue 
       print(f)
       outfile = open(out_prefix + str(temp) + '/' + f + '.json', 'w')
       with open(in_folder + f, 'r') as infile: 
           reader = csv.reader(infile, delimiter='\t')
           for row in reader: 
               prompt = row[2]
               print("Prompt:", prompt)
               gotten_response = False
               n_gens_already = 0
               while not gotten_response: 
                   try: 
                       response = get_response(prompt, temp, 1)
                       n_gens_already += 1
                       if n_gens_already == n_gens: 
                           gotten_response = True
                   except openai.error.RateLimitError: 
                       print("Rate Limit Exceeded, will sleep a little bit and try again")
                       time.sleep(70)
                   except openai.error.APIConnectionError: 
                       print("Socket timeout, will sleep a little and try again")
                       time.sleep(70)
                   except openai.error.APIError: 
                       print("API error, will sleep a little and try again")
                       time.sleep(70)
                   response['input'] = prompt
                   response_str = json.dumps(response)
                   outfile.write(response_str + '\n')
       outfile.close()
       logfile.write(f + '\n') 
   logfile.close()
       
def main(): 
    temp = 0.9
    #in_folder = ORIGINAL
    #out_prefix = LOGS + 'generated_'
    #log_file_path = LOGS + 'already_queried_books_' + str(temp)
    in_folder = REDO
    out_prefix = LOGS + 'redo_'
    log_file_path = LOGS + 'redo_books_log_' + str(temp)
    query_for_stories(in_folder, out_prefix, log_file_path, temp)

if __name__ == "__main__":
    main()

