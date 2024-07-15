from scipy.stats import spearmanr
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

import argparse
parser = argparse.ArgumentParser(prog='Print results', add_help=True)
parser.add_argument('-d', '--dataset', type=str, default='SemEval-English')
parser.add_argument('-m', '--model', type=str, default='bert-base-uncased')
parser.add_argument('-l', '--layer', type=int, default=12)
args = parser.parse_args()

try:
    gold = pd.read_csv(f'datasets/LSC/{args.dataset}/truth/graded.txt', sep='\t', names=['word', 'score'])
    pred = pd.read_csv(f'scores/LSC/{args.dataset}/{args.model}/{args.layer}/token.txt', sep='\t', names=['word', 'measure', 'score'])
    gold = gold.sort_values('word')
    pred = pred.sort_values('word')

    print('app+apdp_canberra')
    print(spearmanr(pred[pred['measure']=='app+apdp_canberra'].score.values, gold.score.values))
    print('ap+jsd')
    print(spearmanr(pred[pred['measure']=='app+jsd'].score.values, gold.score.values))
except:
    pass

try:
    gold = pd.read_csv(f'datasets/LSC/{args.dataset}/truth/binary.txt', sep='\t', names=['word', 'score'])
    pred = pd.read_csv(f'scores/LSC/{args.dataset}/{args.model}/{args.layer}/token.txt', sep='\t', names=['word', 'measure', 'score'])
    gold = gold.sort_values('word')
    pred = pred.sort_values('word')
    y_true = gold.score.values
    
    print('app+jsd')
    y = pred[pred['measure']=='app+jsd'].score.values
    fpr, tpr, thresholds = roc_curve(y_true, y)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_true, [m > thresh for m in y]))
    accuracy_scores = np.array(accuracy_scores)
    max_accuracy = accuracy_scores.max()
    max_accuracy_threshold = thresholds[accuracy_scores.argmax()]
    print(max_accuracy.round(3))

    print('app+apdp_canberra')
    y = pred[pred['measure']=='app+apdp_canberra'].score.values
    fpr, tpr, thresholds = roc_curve(y_true, y)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_true, [m > thresh for m in y]))
    accuracy_scores = np.array(accuracy_scores)
    max_accuracy = accuracy_scores.max()
    max_accuracy_threshold = thresholds[accuracy_scores.argmax()]
    print(max_accuracy.round(3))
except:
    pass
