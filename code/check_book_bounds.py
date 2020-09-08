import sys, re, csv

"""
Check the strings marked as the start and end strings of a book to make sure they are unique.

"""

def proc(filename, textPath):
    outPath = "/mnt/data0/lucy/gpt3_bias/data/stripped/"
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for cols in csv_reader:

            start=cols[8]
            end=cols[9].rstrip()
            bookID=cols[6]
            if bookID == "":
                continue

            if start != "" and end != "":
                data=open("%s/%s.txt" % (textPath, bookID), encoding="utf-8").read()
                start_matches=re.findall(re.escape(start), data)
                end_matches=re.findall(re.escape(end), data)
                start_idx = re.search(re.escape(start), data).start()
                end_idx = re.search(re.escape(end), data).end()
                actual_contents = data[start_idx:end_idx+1]
                with open(outPath + bookID + '.txt', 'w') as outfile: 
                    outfile.write(actual_contents + '\n')
                print("Approx # of white-space tokens:", len(actual_contents.split()))
                
                if len(start_matches) != 1 or len(end_matches) != 1:
                    print ("Problem! %s\tstart matches: %s, end matches: %s" % (bookID, len(start_matches), len(end_matches)))
                else:
                    print("ok %s" % bookID)


# arg1 = Contemporary LitBank Book List (from Google Drive) downloaded as CSV
# arg2 = Path to all OCR'd text files

# python check_book_bounds.py "Contemporary LitBank Book List - List.csv" /Users/dbamman/Dropbox/contemporary_litbank/english/text

proc('/mnt/data0/lucy/gpt3_bias/data/contemporary_litbank_booklist.csv', "/mnt/data1/corpora/contemporary_litbank/english/originals/")
