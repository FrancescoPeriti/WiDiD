from src.visualization import WiDiDClusters
import plotly.io as pio
import pandas as pd
import os

words = set([w.strip() for w in open(target_path, mode='r').readlines() if w.strip() != ''])

with open('lemma-paisa.txt', 'r') as file:
    vocab = file.readlines()
    vocab = {e.split(',')[0].lower() for e in vocab[2:]}

if not os.path.exists('./results'):
    os.makedirs('./results/clusters')
    os.makedirs('./results/metrics')

for word in words:
    clusters = WiDiDClusters(word, minfreq=1)
    pio.write_image(clusters.cluster_plot(), f'results/clusters/{word}_cluster_no_tag.png', scale=6)
    clusters.tfidf(top_n=10, vocab=vocab)
    pio.write_image(clusters.cluster_plot(), f'results/clusters/{word}_cluster.png', scale=6)
    pio.write_image(clusters.word_plot(), f'results/clusters/{word}_word.png', scale=6)
    for m in ['prominence', 'polysemy', 'sense_shift', 'word_shift']:
        clusters.metrics[m].to_csv(f'results/metrics/{word}_{m}.csv')