import argparse
import pandas as pd
from datasets import load_dataset

parser = argparse.ArgumentParser(prog='Processing SemEval-English', description='Remove pos tags')
parser.add_argument('-d', '--dataset_folder', type=str, help='Folder of the dataset to process')
args = parser.parse_args()

def process(text):
    return text.replace('_nn', '').replace('_vb', '')

# -- Targets --
filename_targets = f'{args.dataset_folder}/targets.txt'
df_targets = pd.read_csv(filename_targets, sep='\t', names=['word'])
df_targets['word'] = [process(target) for target in df_targets['word'].values]
df_targets = df_targets.sort_values('word')
df_targets.to_csv(filename_targets, sep='\t', header=None, index=False)

# -- Truth --
for truth in ['binary', 'graded']:
    filename_truth = f'{args.dataset_folder}/truth/{truth}.txt'
    df_truth = pd.read_csv(filename_truth, sep='\t', names=['word', 'score'])
    df_truth['word'] = [process(target) for target in df_truth['word'].values]
    df_truth = df_truth.sort_values('word')
    df_truth.to_csv(filename_truth, sep='\t', header=None, index=False)

# -- Dataset --
for corpus in ['corpus1', 'corpus2']:
    filename = f'{args.dataset_folder}/{corpus}/token/{corpus}.txt'
    dataset = load_dataset('text', data_files=filename, split='train')
    dataset = dataset.map(lambda batch: {'text': [process(text) for text in batch['text']]}, batched=True)['text']
    
    with open(filename, mode='w', encoding='utf-8') as f:
        f.writelines([row+'\n' for row in dataset])