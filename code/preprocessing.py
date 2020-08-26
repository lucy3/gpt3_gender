import csv 
import os

BOOKLIST = '/mnt/data0/lucy/gpt3_bias/data/contemporary_litbank_booklist.csv'

def get_start_end_tokenids(): 
    book_start = {}
    book_end = {}
    with open(BOOKLIST, 'r') as csvfile: 
        reader = csv.DictReader(csvfile)
        for row in reader: 
           book = row['ID if scanned'].strip()
           if book == '': continue
           start = row['Start'].replace('\n', ' ').replace('  ', ' ')
           end = row['End'].replace('\n', ' ').replace('  ', ' ')
           book_start[book] = start
           book_end[book] = end

def which_books(): 
    '''
    Compares the books I have with the books in the spreadsheet
    '''
    booklist = set()
    with open(BOOKLIST, 'r') as csvfile: 
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['ID if scanned'].strip() != '': 
                booklist.add(row['ID if scanned'].strip())
    processed_books = set()
    processed_books.update(os.listdir('/mnt/data0/lucy/gpt3_bias/logs/tokens/'))
    print("Missing from processed:", len(booklist - processed_books))
    print("Missing from spreadsheet:", len(processed_books - booklist))

def main(): 
    #which_books()
    get_start_end_tokenids()

if __name__ == "__main__":
    main()
