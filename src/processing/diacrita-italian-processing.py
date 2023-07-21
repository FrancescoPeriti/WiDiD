import argparse

parser = argparse.ArgumentParser(prog='Processing Diacrita-Italian', description='Remove pos tags')
parser.add_argument('-f', '--corpus_filename', type=str, help='Filename of the corpus to process')
parser.add_argument('-t', '--output_token', type=str, help='Output filename of the token corpus')
parser.add_argument('-l', '--output_lemma', type=str, help='Output filename of the lemma corpus')
args = parser.parse_args()

filename = args.corpus_filename
text_output = args.output_token
lemma_output = args.output_lemma

text = open(filename, mode='r', encoding='utf-8').read().split('\n\n')[:-1]
with open(text_output, mode='a', encoding='utf-8') as f:
    rows = [" ".join([t.split()[0] for t in row.split('\n')])+'\n' for row in text]
    f.writelines(rows)
    
with open(lemma_output, mode='a', encoding='utf-8') as f:
    rows = [" ".join([t.split()[2] for t in row.split('\n')])+'\n' for row in text]
    f.writelines(rows)
