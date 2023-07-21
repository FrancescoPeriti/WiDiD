import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from extraction import WordEmbeddingExtraction

parser = argparse.ArgumentParser(prog='Embedding extraction',
                                 description='Extract the embeddeddings for each target word')
parser.add_argument('-t', '--tokenized_dataset',
                    type=str,
                    help='A string representing the directory path to a tokenized dataset for LSC detection. '
                         'This dataset should contain pre-tokenized text for the target words.')
parser.add_argument('-m', '--model',
                    type=str,
                    help='A string representing the name of the Hugging Face pre-trained model to use for embedding extraction.')
parser.add_argument('-M', '--max_length',
                    type=int, default=512,
                    help='An integer representing the maximum sequence length to use for the embedding extraction process. '
                         'Default value is 512.')
parser.add_argument('-l', '--layers',
                    type=int, default=12,
                    help='An integer representing the number of encoder layers of the pre-trained model to use for embedding extraction. '
                         'Default value is 12.')
parser.add_argument('-b', '--batch_size',
                    type=int, default=8,
                    help='An integer representing the batch size to use for the embedding extraction process. '
                         'Default value is 8.')
parser.add_argument('-o', '--output',
                    type=str,
                    help='Dirname where embeddings will be stored')
parser.add_argument('-n', '--sampling',
                    type=int, default=0,
                        help='An integer representing the number of sentences to sample randomly from the dataset. Default is 0 (no sampling).')
parser.add_argument('-T', '--targets',
                    type=str,
                    help='A string representing the directory path to a text file containing the target words to extract embeddings for.')
parser.add_argument('-s', '--single_model',
                    type=bool, default=True,
                    help='If false, embeddings are extracted from two different fine-tuned model for corpus1 and corpus2 respectively')
args = parser.parse_args()

# target words
words = [word.strip() for word in open(args.targets, mode='r', encoding='utf-8').readlines()]

if args.single_model:
    w = WordEmbeddingExtraction(args.model)

for corpus in ['corpus1', 'corpus2']:
    if not args.single_model:
        w = WordEmbeddingExtraction(args.model+'-'+corpus)

    for word in tqdm(words, desc=corpus):
        tokenization_input = f'{args.tokenized_dataset}/{corpus}/token/{word}.txt'
        embeddings_output = f'{args.output}/{args.model.replace("/", "_")}/{corpus}/token'

        # extraction
        embeddings = w.extract(dataset=tokenization_input, batch_size=args.batch_size,
                              max_length=args.max_length, agg_sub_words='mean',
                              layers=args.layers, sampling=args.sampling)

        # store embeddings
        for l in range(1, args.layers + 1):
            Path(f'{embeddings_output}/{l}/').mkdir(parents=True, exist_ok=True)

            # BERT returns nan vectors
            if 'NorDiaChange' in tokenization_input:
                embeddings[l] = embeddings[l][~torch.any(embeddings[l].isnan(), dim=1)]
            
            torch.save(embeddings[l].to('cpu'), f'{embeddings_output}/{l}/{word}.pt')
