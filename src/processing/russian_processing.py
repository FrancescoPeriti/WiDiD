import json
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(prog='Processing RuShiftEval-Russian', description='Remove pos tags')
parser.add_argument('-d', '--dataset_folder', type=str, help='Folder of the processed dataset')
parser.add_argument('-r', '--row_data_folder', type=str, help='Folder of the dataset to process')
parser.add_argument('-t', '--tokenization_folder', type=str, help='Folder of the tokenization output')
args = parser.parse_args()

raw_data_folder=args.row_data_folder
dataset_folder=args.dataset_folder
output_tokenization_folder=args.tokenization_folder

# -- Targets --
filename_targets = f'{dataset_folder}/targets_test.txt'
target_words = [line.strip() for line in open(filename_targets, mode='r', encoding='utf-8').readlines()]

# container for sentences and tokenization
corpora = dict(corpus1=list(), corpus2=list())
tokens_corpora = dict(corpus1=list(), corpus2=list())

for path in Path(f'{raw_data_folder}/').glob('*.data'):
        
    # load data
    with open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)

      
    # collect data
    for row in data:
        for i in range(1, 3):
            tokens_corpora[f'corpus{i}'].append(dict(token=row['lemma'], 
                                                     lemma=row['lemma'], 
                                                     start=row[f'start{i}'], end=row[f'end{i}'], 
                                                     sent=row[f'sentence{i}']))
            corpora[f'corpus{i}'].append(row[f'sentence{i}']+'\n')
            
# store data
for i in range(1, 3):
    folder=f'{dataset_folder}/corpus{i}/token'
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(folder+f'/corpus{i}.txt', mode='w', encoding='utf-8') as f:
        f.writelines(corpora[f'corpus{i}'])
    
    folder=f'{output_tokenization_folder}/corpus{i}/token'
    Path(folder).mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(tokens_corpora[f'corpus{i}'])

    for word in target_words:
        filename = folder+f'/{word}.txt'
        df[df['token'] == word].to_json(filename, orient='records', lines=True)
        
# Emulate binary scores
filename_graded = f'{dataset_folder}/truth/graded_test.txt'
df_graded=pd.read_csv(filename_graded, sep='\t', names=['word', 'score'], encoding='utf-8').sort_values('score')

filename_binary = f'{dataset_folder}/truth/binary.txt'
df_binary=pd.DataFrame()
df_binary['word']=df_graded['word'][:20].values.tolist() + df_graded['word'][-20:].values.tolist()
df_binary['score'] = [0]*20 + [1]*20
df_binary.to_csv(filename_binary, sep='\t', index=False, header=False, encoding='utf-8')

new_df_graded=pd.DataFrame()
new_df_graded['word']=df_graded['word'][:20].values.tolist() + df_graded['word'][-20:].values.tolist()
new_df_graded['score'] = df_graded['score'][:20].values.tolist() + df_graded['score'][-20:].values.tolist()
df_graded=new_df_graded.sort_values('word')

new_filename_graded = f'{dataset_folder}/truth/graded.txt'
df_graded.to_csv(new_filename_graded, sep='\t', index=False, header=False, encoding='utf-8')

new_filename_targets = f'{dataset_folder}/targets.txt'
df_graded[['word']].to_csv(new_filename_targets, sep='\t', index=False, header=False, encoding='utf-8')