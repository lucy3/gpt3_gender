#!/bin/bash
# Runs NER on input

for filename in /mnt/data1/corpora/contemporary_litbank/english/originals/*; do
    justfile=$(basename $filename)
    name=$(echo "$justfile" | cut -f 1 -d '.')
    mkdir -p /mnt/data0/lucy/gpt3_bias/logs/ner/$name
    java -Xmx4g -cp "/mnt/data0/lucy/stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -file $filename -outputFormat conll -output.columns word,ner -outputDirectory /mnt/data0/lucy/gpt3_bias/logs/ner/$name/
done
