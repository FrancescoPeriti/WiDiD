import json
import torch
import numpy as np
from os import path
from tqdm import tqdm
from pathlib import Path
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from collections import defaultdict, Counter
from src.extraction import WordEmbeddingExtraction
from src.cluster import APosterioriaffinityPropagation as APP

def jsd(L1: np.array, L2: np.array) -> float:
    labels = np.unique(np.concatenate([L1, L2]))

    c1 = Counter(L1)
    c2 = Counter(L2)

    L1_dist = np.array([c1[l] for l in labels])
    L2_dist = np.array([c2[l] for l in labels])

    L1_dist = L1_dist / L1_dist.sum()
    L2_dist = L2_dist / L2_dist.sum()

    m = (L1_dist + L2_dist) / 2

    return (entropy(L1_dist, m) + entropy(L2_dist, m)) / 2

######################## Parameters ########################
model = "bert-base-multilingual-cased" # hugginface model
target_path = 'targets.txt' # target words
corpus_path = 'data' # folder containing data
embs_path = 'case-study/embs' # folder where to store embeddings
clus_path = 'case-study/clustering' # folder where to store clustering results

######################## Word embeddings extraction #########################
bert = WordEmbeddingExtraction(model) # helper for embedding extraction

# function for embedding extraction
def extract(dataset, batch_size=32, max_length=512, agg_sub_words='mean', layers=12):
    return bert.extract(dataset=dataset,
                        batch_size=batch_size,
                        max_length=max_length,
                        agg_sub_words=agg_sub_words,
                        layers=layers)


# target words
words = set([w.strip() for w in open(target_path, mode='r').readlines() if w.strip() != ''])

bar = tqdm(range(1, 19), total=len(words) * 18, position=0, leave=True)

for i in bar:
    # create dir
    for layer in range(1, 13):
        Path(f'{embs_path}/corpus{i}/{layer}').mkdir(parents=True, exist_ok=True)

    # extract embs
    for word in sorted(words):
        bar.set_description(f'corpus{i}-{word}')

        dataset = f'{corpus_path}/corpus{i}/{word}.json'
        # avoid errors if no occurrences are available for that period
        if not path.exists(dataset):
            continue
        E = extract(dataset)
        for layer in range(1, 13):
            torch.save(E[layer].to('cpu'), f'{embs_path}/corpus{i}/{layer}/{word}.pt')
        bar.update(1)

######################## Clustering #########################
def get_cluster_membership(evolution, step, exemplar):
    '''Return the new idx assigned to a cluster during the incremental iteration'''
    for cluster_idx, cluster_items in evolution[step].items():
        if exemplar in cluster_items:
            return int(cluster_idx)

def tojson(memory):
    '''From array to json'''
    new_memory = defaultdict(lambda: defaultdict())

    for cluster in list(memory):
        for cluster1 in list(memory[cluster]):
            cluster, cluster1 = int(cluster), int(cluster1)
            new_memory[cluster][cluster1] = [int(id_item) for id_item in memory[cluster][cluster1]]

    return dict(new_memory)

def get_cluster_appearance(evolution):
    '''Return the new clusters (only new items) for each period'''
    all_clusters = list()

    for step in evolution:
        current_clusters = list(evolution[step].keys())

        if step == 0:
            # all clusters are new
            all_clusters.append(current_clusters)
        else:
            # filter only new clusters
            prev_clusters = list(evolution[step - 1].keys())
            current_clusters = [cluster for cluster in current_clusters
                                if len(set(evolution[step][cluster]).intersection(prev_clusters)) == 0]
            all_clusters.append(current_clusters)

    return all_clusters

# new clusters idx for each period
cluster_appearance = dict()

# how a cluster change from a period to another
cluster_shift = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# number of active clusters in each time step
n_clusters = defaultdict(list)

# semantic shift of words
ssd = defaultdict(list)

bar = tqdm(list(words))
for word in bar:
    bar.set_description(word)
    app = APP(affinity='cosine', damping=0.9, random_state=42, th_gamma=0)

    # let's store the last labels (last clustering result)
    last_labels = np.array([])

    # let's store the last centroids
    last_centroids = np.array([])
    last_embs = None

    ignore = list()

    # Pay attention: different kind of index used (from 0 to 17 and from 1 to 18)
    for step in range(0, 18):
        try:
            # load embeddings
            pt_embs = torch.load(f"{embs_path}/corpus{step + 1}/12/{word}.pt")
        except FileNotFoundError:
            # update memory
            if step > 0:
                app.memory_.update({app.step_: app.memory_[app.step_ - 1]})
            else:
                app.memory_.update({app.step_: {}})
                app.cluster_centers_ = list()
            ignore.append(f'{step}\n')

            # update step
            app.step_ += 1

            # compute shift -> 0 shift
            if step > 0:
                # no differences introduced for each cluster
                for cluster in app.memory_[step - 1]:
                    # print(word, step, (cluster, cluster1))
                    cluster1 = cluster
                    cluster_shift[word][(step - 1, step)][(cluster, cluster1)] = 0

                # no differences introduced for the target word
                ssd[word].append(0)
                # clusters are the same as before
                n_clusters[word].append(np.unique(last_labels).shape[0])
            else:
                # no clusters
                n_clusters[word].append(0)

        # no exception
        else:
            # clustering
            app.fit(pt_embs)

            # update labels
            last_labels = app.labels_

            # no history with step == 0
            if step > 0:
                for cluster in app.memory_[step - 1]:
                    # cluster map from step-1 to step
                    cluster1 = get_cluster_membership(app.memory_, step, cluster)
                    cluster = int(cluster)
                    dist = cosine(last_centroids[cluster], app.cluster_centers_[cluster1])
                    # shift of the word from step-1 to step. cluster-cluster1 is the mapping of the cluster.
                    cluster_shift[word][(step - 1, step)][(cluster, cluster1)] = dist

                n1, n2 = last_embs.shape[0], pt_embs.shape[0]
                L1 = last_labels[-n1 - n2:-n2]
                L2 = last_labels[-n2:]
                # print((last_embs.shape[0]+pt_embs.shape[0]), last_embs.shape[0], L1.shape[0], L2.shape[0])
                ssd[word].append(jsd(L1, L2))

            n_clusters[word].append(np.unique(last_labels).shape[0])

            # update centroids
            last_centroids = app.cluster_centers_

            # update embs
            last_embs = pt_embs

        # create dir
        Path(f"{clus_path}/{word}/corpus{step + 1}/").mkdir(exist_ok=True, parents=True)

        # store results
        np.save(open(f"{clus_path}/{word}/corpus{step + 1}/labels.npy", mode='wb'), last_labels)
        json.dump(tojson(app.memory_), open(f"{clus_path}/{word}/corpus{step + 1}/evolution.json", mode='w'))

        # store time periods to ignore
    open(f"{clus_path}/{word}/ignore.txt", mode='w').writelines(ignore)

    cluster_appearance[word] = get_cluster_appearance(app.memory_)

######################## Sense Shift #########################
import numpy as np
import pandas as pd

max_shift = 0
sense_shift = defaultdict(lambda: defaultdict(list))
for word in tqdm(list(words)):

    # final labels at time 18
    final_labels = np.load(f"{clus_path}/{word}/corpus18/labels.npy")
    final_labels = final_labels.astype(int)

    # number of documents per time period
    cum_docs = np.array([np.load(f"{clus_path}/{word}/corpus{i}/labels.npy").shape[0]
                         for i in range(1, 19)])

    # active labels at time 18
    clusters_id = np.unique(final_labels)
    clusters_id = clusters_id.astype(int)

    # labels per each time period
    periodlabels = np.split(final_labels, cum_docs)

    for c in clusters_id:
        last_embs = None
        last_labels = None

        for step in range(1, 18):
            prev_labels = periodlabels[step - 1] == c
            prev_embs = torch.load(f"{embs_path}/corpus{step}/12/{word}.pt")

            curr_embs = torch.load(f"{embs_path}/corpus{step + 1}/12/{word}.pt")
            curr_labels = periodlabels[step] == c

            if any(prev_labels):

                prev_embs = prev_embs[prev_labels].mean(axis=0)
                last_embs = prev_embs
                last_labels = prev_labels
            elif last_embs is None:
                sense_shift[word][c].append(np.nan)
                continue

            curr_embs = curr_embs[curr_labels].mean(axis=0)

            distance = cosine(last_embs, curr_embs)
            sense_shift[word][c].append(distance)

            if distance == distance:
                max_shift = max(max_shift, distance)

# normalize shift for the maximum shift
sense_shift_df = list()
for word in sense_shift:
    for c in sense_shift[word]:
        record = dict(word=word, cluster=c)
        for i, v in enumerate(sense_shift[word][c]):
            record[i] = v / max_shift
        sense_shift_df.append(record)

# store results
pd.DataFrame(sense_shift_df).to_csv('sense_shift.tsv', sep='\t', index=False)


######################## Semantic Shift #########################
max_shift = 0
semantic_shift = defaultdict(list)
for word in tqdm(list(words)):
    # final labels at time 18
    final_labels = np.load(f"{clus_path}/{word}/corpus18/labels.npy")
    final_labels = final_labels.astype(int)

    # number of documents per time period
    cum_docs = np.array([np.load(f"{clus_path}/{word}/corpus{i}/labels.npy").shape[0]
                         for i in range(1, 19)])

    # active labels at time 18
    clusters_id = np.unique(final_labels)
    clusters_id = clusters_id.astype(int)

    # labels per each time period
    periodlabels = np.split(final_labels, cum_docs)

    for step in range(1, 18):
        prev_labels = periodlabels[step - 1]
        last_labels = prev_labels
        curr_labels = periodlabels[step]

        distance = jsd(last_labels, curr_labels)
        semantic_shift[word].append(distance)

        if distance == distance:
            max_shift = max(max_shift, distance)

# normalize shift for the maximum shift
semantic_shift_df = list()
for word in semantic_shift:
    record = dict(word=word)
    for i, v in enumerate(semantic_shift[word]):
        record[i] = v / max_shift
    semantic_shift_df.append(record)

# store results
pd.DataFrame(semantic_shift_df).to_csv('semantic_shift.tsv', sep='\t', index=False)
