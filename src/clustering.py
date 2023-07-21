import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import AffinityPropagation
from cluster import APosterioriaffinityPropagation
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


class Clustering:
    """
    The Clustering class is designed for clustering text embeddings. It uses Affinity Propagation (AP) or A Posteriori Affinity Propagation (APPA) clustering algorithm for clustering.
    This class takes text embeddings from two corpora and generates cluster labels for each embedding of each corpus. It saves the labels to output directories.
    """

    def __init__(self, algo: str, affinity: str = 'cosine',
                 damping: float = 0.9, max_iter: int = 200,
                 convergence_iter: int = 15, copy: bool = True, preference: bool = None,
                 verbose: bool = False, random_state: int = 42, th_gamma: int = 0,
                 pack='mean', singleton='one'):
        """
        Args:
            algo(str, default='ap'): The clustering algorithm to use, can be 'ap' for Affinity Propagation or 'app' for APosteriori Affinity Propagation.
            affinity(str, default='cosine'): The distance metric to use, can be 'euclidean' or 'cosine'.
            damping(float, default=0.9): The damping factor, which is used to avoid numerical oscillations when updating the messages, must be between 0.5 and 1.
            max_iter(int, default=200): The maximum number of iterations.
            convergence_iter(int, default=15): The number of iterations with no change in the number of estimated clusters before the algorithm stops.
            copy(int, default=15): Whether or not to make a copy of the input data.
            preference(bool, default=None): The preference parameter used to determine the number of clusters, if not specified, it will be set to the median of the input similarities.
            verbose(bool, default=False): Whether or not to print progress messages during fitting.
            random_state(int, default=42): The random seed used to initialize the centers.
            th_gamma(int, default=0): The threshold value for gamma, the minimum number of samples required to form a cluster.
            pack(str, default='mean'): How to pack the embeddings, can be 'mean' for mean packing, 'centroid' for considering the exemplars of clusters, or 'most_similar' for considering the most similar embedding to the mean.
            singleton(str, default='one'): How to handle singleton clusters, can be 'one' to assign them to a unique cluster, or 'all' to assign them to different singleton clusters.
        """

        self.algo = algo
        self.affinity = affinity
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.preference = preference
        self.verbose = verbose
        self.random_state = random_state
        self.th_gamma = th_gamma
        self.pack = pack
        self.singleton = singleton

    def fit_russian(self, embs1: str, embs2: str, embs3: str, out1: str, out2: str, out3: str, out4:str, out5: str, out6:str):
        pt_embs1 = torch.load(embs1).numpy()
        pt_embs2 = torch.load(embs2).numpy()
        pt_embs3 = torch.load(embs3).numpy()

        self.ap = APosterioriaffinityPropagation(affinity=self.affinity, damping=self.damping,
                                                     max_iter=self.max_iter,
                                                     convergence_iter=self.convergence_iter,
                                                     copy=self.copy,
                                                     preference=self.preference, verbose=self.verbose,
                                                     random_state=self.random_state,
                                                     th_gamma=self.th_gamma, pack=self.pack,
                                                     singleton=self.singleton)
        self.ap.fit(pt_embs1)
        self.ap.fit(pt_embs2)
        self.ap.fit(pt_embs3)
        L = self.ap.labels_
        n1, n2, n3 = pt_embs1.shape[0], pt_embs2.shape[0], pt_embs3.shape[0]
        L1, L2, L3 = L[:n1], L[n1:n1+n2], L[n1+n2:]
        np.save(out1, L1) #Russian12-1
        np.save(out2, L2) #Russian12-2
        np.save(out3, L2) #Russian23-2
        np.save(out4, L3) #Russian23-3
        np.save(out5, L1) #Russian13-1
        np.save(out6, L3) #Russian13-3
        
    def fit(self, embs1: str, embs2: str, out1: str, out2: str):
        """
        This method fits the clustering model to the input data.
        Args:
            embs1(str): The file path to the numpy array containing the embeddings for corpus1.
            embs2 (str): The file path to the numpy array containing the embeddings for corpus2.
            out1 (str): The file path to save the cluster labels for corpus1.
            out2 (str): The file path to save the cluster labels for corpus2.
        """

        pt_embs1 = torch.load(embs1).numpy()
        pt_embs2 = torch.load(embs2).numpy()
        
        if self.algo == 'app':
            self.ap = APosterioriaffinityPropagation(affinity=self.affinity, damping=self.damping,
                                                     max_iter=self.max_iter,
                                                     convergence_iter=self.convergence_iter,
                                                     copy=self.copy,
                                                     preference=self.preference, verbose=self.verbose,
                                                     random_state=self.random_state,
                                                     th_gamma=self.th_gamma, pack=self.pack,
                                                     singleton=self.singleton)
            self.ap.fit(pt_embs1)
            self.ap.fit(pt_embs2)
        else:
            self.ap = AffinityPropagation(affinity='precomputed', damping=self.damping,
                                          max_iter=self.max_iter,
                                          convergence_iter=self.convergence_iter,
                                          copy=self.copy,
                                          preference=self.preference, verbose=self.verbose,
                                          random_state=self.random_state)
            sim = cosine_similarity(np.concatenate([pt_embs1, pt_embs2], axis=0))
            self.ap.fit(sim)

        L = self.ap.labels_
        L1, L2 = L[:pt_embs1.shape[0]], L[pt_embs1.shape[0]:]

        np.save(out1, L1)
        np.save(out2, L2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='clustering', add_help=True)
    parser.add_argument('-e', '--embeddings', type=str,
                        help='The directory path to the embeddings for both corpus1 and corpus2.')
    parser.add_argument('-l', '--layers', type=int, default=12,
                        help='The number of layers in the language model used to generate the embeddings.')
    parser.add_argument('-o', '--output', type=str,
                        help='The directory path to save the cluster labels for both corpus1 and corpus2.')
    parser.add_argument('-t', '--targets', type=str,
                        help='The directory path to save the cluster labels for both corpus1 and corpus2.')
    parser.add_argument('-a', '--algo', type=str,
                        help='The clustering algorithm to use.')
    args = parser.parse_args()

    # Targets
    words = [word.strip() for word in open(args.targets, mode='r', encoding='utf-8').readlines()]

    for word in tqdm(words):
        for l in range(1, args.layers + 1):
            if args.algo == 'app_incremental':
                algo = 'app'
                Path(f'{args.output}/{algo}/corpus1/token/{l}/').mkdir(parents=True, exist_ok=True)
                Path(f'{args.output}/{algo}/corpus2/token/{l}/').mkdir(parents=True, exist_ok=True)
                Path(f'{args.output.replace("12", "23")}/{algo}/corpus1/token/{l}/').mkdir(parents=True, exist_ok=True)
                Path(f'{args.output.replace("12", "23")}/{algo}/corpus2/token/{l}/').mkdir(parents=True, exist_ok=True)
                Path(f'{args.output.replace("12", "13")}/{algo}/corpus1/token/{l}/').mkdir(parents=True, exist_ok=True)
                Path(f'{args.output.replace("12", "13")}/{algo}/corpus2/token/{l}/').mkdir(parents=True, exist_ok=True)

                c = Clustering(algo=algo)
                c.fit_russian(f'{args.embeddings}/corpus1/token/{l}/{word}.pt', #embs1
                              f'{args.embeddings}/corpus2/token/{l}/{word}.pt', #embs2
                              f'{args.embeddings.replace("12", "23")}/corpus2/token/{l}/{word}.pt', #embs3
                              f'{args.output}/{algo}/corpus1/token/{l}/{word}', # out1
                              f'{args.output}/{algo}/corpus2/token/{l}/{word}', # out2
                              f'{args.output.replace("12", "23")}/{algo}/corpus1/token/{l}/{word}', #out3
                              f'{args.output.replace("12", "23")}/{algo}/corpus2/token/{l}/{word}', #out4
                              f'{args.output.replace("12", "13")}/{algo}/corpus1/token/{l}/{word}', #out5
                              f'{args.output.replace("12", "13")}/{algo}/corpus2/token/{l}/{word}') #out6          
            else:
                Path(f'{args.output}/{args.algo}/corpus1/token/{l}/').mkdir(parents=True, exist_ok=True)
                Path(f'{args.output}/{args.algo}/corpus2/token/{l}/').mkdir(parents=True, exist_ok=True)
                
                c = Clustering(algo=args.algo)
                c.fit(f'{args.embeddings}/corpus1/token/{l}/{word}.pt',
                      f'{args.embeddings}/corpus2/token/{l}/{word}.pt',
                      f'{args.output}/{args.algo}/corpus1/token/{l}/{word}',
                      f'{args.output}/{args.algo}/corpus2/token/{l}/{word}')
