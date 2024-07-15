import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.stats import entropy
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import directed_hausdorff

class SemanticShiftDetection:
    """
    Class for detecting semantic shift in a given dataset.
    """

    def form_based_fit(self, word:str, E1: np.array, E2: np.array) -> list:
        """
        Calculates semantic shift scores between two sets of word embeddings without taking into account any
        sense-level information.

        Args:
            word(str): a target word.
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.

        Returns:
            scores(list): A list of dictionaries, each representing the scores calculated for a specific measure.
            Each dictionary has the following keys:
                'measure' : str
                    The name of the measure used to calculate the score.
                'score' : float
                    The score calculated for the given word, layer and measure.
        """
        # result container
        scores = list()
        
        for measure in ['hd', 'prt',
                        'apd_cosine', 'apd_euclidean', 'apd_canberra',
                        'div', 'cd']:
            
            scores.append(dict(word=word, measure=measure, score=getattr(self, measure)(None, None, E1.copy(), E2.copy())))

        return scores

    def sense_based_fit(self, word:str, algo:str, E1: np.array, E2: np.array, L1: np.array, L2: np.array):
        """
         Calculates semantic shift scores between two sets of word embeddings while taking into account sense-level
        information.

        Args:
            word(str): A target word.
            algo(str): The name of clustering algorithm used.
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            scores(list): A list of dictionaries, each representing the scores calculated for a specific measure.
            Each dictionary has the following keys:
                'measure' : str
                    The name of the measure used to calculate the score.
                'score' : float
                    The score calculated for the given word, layer and measure.
        """
        # container score
        scores = list()

        for measure in ["jsd", "pdis", "pdiv", "apdp_euclidean", "apdp_cosine", "apdp_canberra", "mns", "wd", "ms", "csc", "cdcd"]:
            scores.append(dict(word=word, measure=f'{algo}+{measure}', score=getattr(self, measure)(L1.copy(), L2.copy(), E1.copy(), E2.copy())))

        return scores

    @staticmethod
    def jsd(L1: np.array, L2: np.array, E1: np.array, E2: np.array) -> float:
        """Returns the Jensen Shannon Divergence between the distributions of senses in two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        labels = np.unique(np.concatenate([L1, L2]))

        c1 = Counter(L1)
        c2 = Counter(L2)

        L1_dist = np.array([c1[l] for l in labels])
        L2_dist = np.array([c2[l] for l in labels])

        L1_dist = L1_dist / L1_dist.sum()
        L2_dist = L2_dist / L2_dist.sum()

        m = (L1_dist + L2_dist) / 2

        return (entropy(L1_dist, m) + entropy(L2_dist, m)) / 2

    @staticmethod
    def pdis(L1: np.array, L2: np.array, E1: np.array, E2: np.array) -> float:
        """Returns the cosine distance between the prototype embeddings of two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        # cluster centroids
        mu_E1 = np.array([E1[L1 == label].mean(axis=0) for label in np.unique(L1)])
        mu_E2 = np.array([E2[L2 == label].mean(axis=0) for label in np.unique(L2)])

        return cosine(mu_E1.mean(axis=0), mu_E2.mean(axis=0))

    @staticmethod
    def div(L1: np.array, L2: np.array, prev_features: np.array, features: np.array) -> float:
        """Returns the Difference Between Diversities (DIV) between two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        # centroids
        mu_E1, mu_E2 = E1.mean(axis=0), E2.mean(axis=0)

        # diversities
        div_E1 = np.array([cosine(x, mu_E1) for x in prev_features])
        div_E2 = np.array([cosine(y, mu_E2) for y in features])

        return abs(div_E1.mean(axis=0) - div_E2.mean(axis=0))

    @staticmethod
    def pdiv(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the difference between prototype embedding diversities between two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        # cluster centroids
        mu_E1 = np.array([E1[L1 == label].mean(axis=0) for label in np.unique(L1)])
        mu_E2 = np.array([E2[L2 == label].mean(axis=0) for label in np.unique(L2)])

        return SemanticShiftDetection.div(L1, L2, mu_E1, mu_E2)

    @staticmethod
    def cd(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the cosine distance between the centroid embeddings of two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        # cluster centroids
        mu_E1 = E1.mean(axis=0)
        mu_E2 = E2.mean(axis=0)

        return cosine(mu_E1, mu_E2)

    @staticmethod
    def prt(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the Inverse Cosine distance between the centroid embeddings of two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        # cluster centroids
        mu_E1 = E1.mean(axis=0)
        mu_E2 = E2.mean(axis=0)

        return 1 / (1 - cosine(mu_E1, mu_E2))

    @staticmethod
    def apd_euclidean(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the average pairwise Euclidean distance between two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """
        return np.mean(cdist(E1, E2, metric='euclidean'))

    @staticmethod
    def apdp_euclidean(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the average pairwise Euclidean distance between the cluster centroids of two set of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        # cluster centroids
        mu_E1 = np.array([E1[L1 == label].mean(axis=0) for label in np.unique(L1)])
        mu_E2 = np.array([E2[L2 == label].mean(axis=0) for label in np.unique(L2)])

        return SemanticShiftDetection.apd_euclidean(L1, L2, mu_E1, mu_E2)

    @staticmethod
    def apdp_cosine(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the average pairwise Cosine distance between the cluster centroids of two set of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """
        # cluster centroids
        mu_E1 = np.array([E1[L1 == label].mean(axis=0) for label in np.unique(L1)])
        mu_E2 = np.array([E2[L2 == label].mean(axis=0) for label in np.unique(L2)])
        return SemanticShiftDetection.apd_cosine(L1, L2, mu_E1, mu_E2)

    @staticmethod
    def apdp_canberra(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the average pairwise canberra distance between the cluster centroids of two set of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """
        # cluster centroids
        mu_E1 = np.array([E1[L1 == label].mean(axis=0) for label in np.unique(L1)])
        mu_E2 = np.array([E2[L2 == label].mean(axis=0) for label in np.unique(L2)])
        return SemanticShiftDetection.apd_canberra(L1, L2, mu_E1, mu_E2)

    @staticmethod
    def apd_canberra(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the average pairwise canberra distance between two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """
        return np.mean(cdist(E1, E2, metric='canberra'))

    @staticmethod
    def apd_cosine(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the average pairwise canberra distance between two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """
        return np.mean(cdist(E1, E2, metric='cosine'))

    @staticmethod
    def mns(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the Maximum novelty scores between two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        labels = np.unique(np.concatenate([L1, L2]))
        epsilon = 0.001

        c1 = Counter(L1)
        c2 = Counter(L2)

        L1_dist = np.array([c1[l] + epsilon for l in labels])
        L2_dist = np.array([c2[l] + epsilon for l in labels])

        L1_dist = L1_dist / L1_dist.sum()
        L2_dist = L2_dist / L2_dist.sum()

        return max(L1_dist / L2_dist)

    @staticmethod
    def wd(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the Wassertein Distance between two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """
        labels = np.unique(np.concatenate([L1, L2]))

        c1 = Counter(L1)
        c2 = Counter(L2)

        L1_dist = np.array([c1[l] for l in labels])
        L2_dist = np.array([c2[l] for l in labels])

        L1_dist = L1_dist / L1_dist.sum()
        L2_dist = L2_dist / L2_dist.sum()

        mu_E1 = np.array([E1[L1 == label].mean(axis=0) if label in L1
                                     else np.zeros(E1[0].shape)
                                     for label in labels])
        mu_E2 = np.array([E2[L2 == label].mean(axis=0) if label in L2
                                else np.zeros(E2[0].shape)
                                for label in labels])

        M = np.nan_to_num(np.array([cdist(mu_E1, mu_E2, metric='cosine')])[0], nan=1)

        return wasserstein_distance(L1_dist, L2_dist, M)

    @staticmethod
    def ms(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the Maximum Square two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        labels = np.unique(np.concatenate([L1, L2]))

        c1 = Counter(L1)
        c2 = Counter(L2)

        L1_dist = np.array([c1[l] for l in labels])
        L2_dist = np.array([c2[l] for l in labels])

        L1_dist = L1_dist / L1_dist.sum()
        L2_dist = L2_dist / L2_dist.sum()

        return max((L1_dist - L2_dist) ** 2)

    @staticmethod
    def csc(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the Coefficient of Semantic Change between two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        labels = np.unique(np.concatenate([L1, L2]))
        epsilon = 0.001

        c1 = Counter(L1)
        c2 = Counter(L2)

        L1_dist = np.array([c1[l] + epsilon for l in labels])
        L2_dist = np.array([c2[l] + epsilon for l in labels])

        L1_dist = L1_dist / L1_dist.sum()
        L2_dist = L2_dist / L2_dist.sum()

        L1_n = L1_dist.sum()
        L2_n = L2_dist.sum()

        return (L2_n * L1_dist - L1_n * L2_dist).sum() / (L1_n * L2_n)

    @staticmethod
    def cdcd(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the Cosine distance between labels of two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """

        labels = np.unique(np.concatenate([L1, L2]))

        c1 = Counter(L1)
        c2 = Counter(L2)

        L1_dist = np.array([c1[l] for l in labels])
        L2_dist = np.array([c2[l] for l in labels])

        L1_dist = L1_dist / L1_dist.sum()
        L2_dist = L2_dist / L2_dist.sum()

        return cosine(L1_dist, L2_dist)

    @staticmethod
    def hd(L1:np.array, L2:np.array, E1:np.array, E2:np.array) -> float:
        """
        Returns the Hausdorff distance between two sets of data.

        Args:
            E1(np.array): A numpy array representing the word embeddings of the first set of words.
            E2(np.array): A numpy array representing the word embeddings of the second set of words.
            L1(np.array): A numpy array representing the sense labels for the words in the first set.
            L2(np.array): A numpy array representing the sense labels for the words in the second set.

        Returns:
            score(float)
        """
        return directed_hausdorff(E1, E2)[0]

if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(prog='clustering', add_help=True)
    parser.add_argument('-e', '--embeddings',
                        type=str,
                        help='A string representing the directory path to the stored embeddings for the benchmark dataset for LSC detection.')
    parser.add_argument('-L', '--labels',
                        type=str,
                        help='A string representing the directory path to the stored labels for the benchmark dataset for LSC detection.')
    parser.add_argument('-t', '--targets',
                        type=str,
                        help='A string representing the directory path to a text file containing the target words.')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='A string representing the output file.')
    parser.add_argument('-l', '--layers',
                        type=int, default=12,
                        help='An integer representing the number of encoder layers of the pre-trained model used for embedding extraction. '
                         'Default value is 12.')
    parser.add_argument('-m', '--model',
                        type=str, default='bert-base-uncased',
                        help='A string representing the name of the Hugging Face pre-trained model used for attention extraction.')
    args = parser.parse_args()

    # target words
    words = [word.strip() for word in open(args.targets, mode='r', encoding='utf-8').readlines()]

    # model name
    model=args.model.replace("/", "_")

    s = SemanticShiftDetection()
    for l in tqdm(list(range(1, args.layers+1))):
        scores=list()
        for word in words:
            # embeddings
            E1_path = f'{args.embeddings.replace("_random", "")}/{model}/corpus1/token/{l}/{word}.pt'
            E2_path = f'{args.embeddings}/{model}/corpus2/token/{l}/{word}.pt'
            E1 = torch.load(E1_path).numpy()
            E2 = torch.load(E2_path).numpy()

            # fit - form based
            #form_based_scores = s.form_based_fit(word, E1, E2)
            #scores.extend(form_based_scores)

            # clustering algorithm
            for algo in ['app']:
                L1_path = f'{args.labels}/{model}/{algo}/corpus1/token/{l}/{word}.npy'
                L2_path = f'{args.labels}/{model}/{algo}/corpus2/token/{l}/{word}.npy'
                L1 = np.load(L1_path)
                L2 = np.load(L2_path)

            #    # fit - sense based
                sense_based_scores = s.sense_based_fit(word, algo, E1, E2, L1, L2)
                scores.extend(sense_based_scores)


        Path(os.path.dirname(f'{args.output}/{l}/token.txt')).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scores).to_csv(f'{args.output}/{l}/token.txt', sep='\t', index=False, header=None)
