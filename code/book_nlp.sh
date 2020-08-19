mkdir -p /mnt/data0/lucy/gpt3_bias/logs/output/
mkdir -p /mnt/data0/lucy/gpt3_bias/logs/tokens/
for filename in /mnt/data1/corpora/contemporary_litbank/english/originals/*; do
    justfile=$(basename $filename)
    name=$(echo "$justfile" | cut -f 1 -d '.')
    /mnt/data0/lucy/book-nlp/runjava novels/BookNLP -doc $filename -printHTML -p /mnt/data0/lucy/gpt3_bias/logs/output/$name -tok /mnt/data0/lucy/gpt3_bias/logs/tokens/$name -f
done
