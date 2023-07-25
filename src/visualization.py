from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import math
import pymannkendall as mk
import plotly.graph_objects as go
import colorcet as cc

pd.set_option("display.precision", 3)
clus_path = 'case-study/clustering'
corpus_path = 'data'

class WiDiDClusters:
    
    def __init__(self, word, start=1, end=18, minfreq=1):
        self.word = word
        self.start = start
        self.end = end
        self.minfreq = minfreq
        self.periods = list(set(range(self.start, self.end+1)))
        
        self.labels = np.load(f'../{clus_path}/{self.word}/corpus{self.end}/labels.npy')
        try:
            self.labels[self.labels == -1] = int(max(self.labels)) + 1
        except:
            raise Exception(f'{word} does not appear for the selected periods.')
            
        self.clusters = self._docs_per_cluster(minfreq)
        if not self.clusters:
            raise Exception(f'{word} does not appear for the selected periods.')
        
        self.texts = defaultdict(list)
        self.sentences = []
        for i in range(self.start, self.end+1):
            try:
                with open(f'../{corpus_path}/corpus{i}/{word}.json', 'r') as file:
                    sent_i = file.readlines()
                    self.sentences.extend([eval(s)['sent'] for s in sent_i][:100])
            except:
                pass
        for l, s in zip(self.labels, self.sentences):
            self.texts[l].append(s)
        
        
    def _docs_per_cluster(self, minfreq):
        cum_docs = np.array([len(np.load(f'../{clus_path}/{self.word}/corpus{i}/labels.npy')) for i in range(self.start, self.end+1)][:-1])
        clusters_id = np.unique(self.labels)
        periodlabels = np.split(self.labels, cum_docs)
        
        clusters = defaultdict(list)
        for i in range(self.start, self.end+1):
            for c in clusters_id:
                n = sum(periodlabels[i-1]==c)
                #if n >= minfreq:
                #    clusters[int(c)].append(n)
                #else:
                #    clusters[int(c)].append(0)
                clusters[int(c)].append(n)
                    
        to_pop = []
        for c, d in clusters.items():
            #if np.array(d).sum() == 0:
            if np.array(d).sum() <= minfreq:
                to_pop.append(c)
        for c in to_pop:
            clusters.pop(c)
                
        return clusters
    
    def cluster_plot(self):
        data = {'x': [], 'y': [], 'color': [], 'docs': [], 'size': [], 'shift': []}
        sense_shift = self.metrics['sense_shift']
        for i, (cluster, docs) in enumerate(self.clusters.items()):
            occurrences = np.nonzero(np.array(docs))[0] + 1
            
            # Add time periods
            data['x'].extend(occurrences)
            data['x'].append(None)

            # Add occurrences
            data['y'].extend([i]*len(occurrences))
            data['y'].append(None)

            # Generate color
            colors = list(np.random.choice(range(256), size=3))
            #rgb = f'rgb({colors[0]},{colors[1]},{colors[2]})'
            values = ','.join([str(int(v*255)) for v in cc.glasbey_category10[i]])
            rgb = f"rgb({values})"
            data['color'].extend([rgb]*(len(occurrences)+1))

            # N of documents
            data['docs'].extend([n for n in docs if n != 0])
            data['docs'].append(None)

            # Add marker size
            data['size'].extend([max(8, math.log(1+n)*5) for n in docs if n != 0])
            data['size'].append(0)
            
            # Add shift values for lines
            c_shift = sense_shift.loc[cluster]
            c_shift = c_shift[c_shift.notna()]
            data['shift'].extend(list(c_shift))
            data['shift'].append(0)
            data['shift'].append(0)
            
        height = 208 + 40*len(self.clusters)
        width = 1200
        
        try:
            ylabels = list(self.keywords.values())
        except:
            ylabels = list(self.clusters)
        
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=data["x"],
                    y=data["y"],
                    mode="lines+markers",
                    line=dict(
                        color='grey'
                    ),
                    marker=dict(
                        color=data["color"],
                        size=data['size'],
                        opacity=1,
                        line=dict(
                            width=0
                        )
                    ),
                    hovertemplate="""Cluster: %{y} <br> Time Period: %{x} <br> Documents: %{text}<extra></extra>""",
                    text=data['docs']
                ),
            ]
        )

        fig.update_layout(
            title=f'{self.word}',
            height=height,
            width=width,
            template='plotly_white',
            showlegend=False,
            xaxis = dict(
                tickmode = 'array',
                tickvals = list(self.periods),
                range=[self.periods[0]-0.5, self.periods[-1]+0.5],
            ),
            yaxis = dict(
                tickmode = 'array',
                tickvals = list(list(range(len(self.clusters)))),
                ticktext = ylabels,
                #tickfont = dict(size=10)
            ),
        )

        fig.update_yaxes(autorange="reversed")
        
        for x, y, shift in list(zip(data['x'], data['y'], data['shift'])):
            if x is not None and y is not None and shift:
                fig.add_annotation(x=x+.5, y=y, text=format(shift, '.2f'), showarrow=False, yshift=10)
                
        fig.update_annotations(
            font=dict(size=12)
        )
        
        return fig
    
    def word_plot(self, yrange=None):
        polysemy = np.array(self.metrics['polysemy'].loc[0][:-1].astype(int))
        shift = np.array(self.metrics['word_shift'])[0]
        height = 300
        width = 1200
        
        colors = list(np.random.choice(range(256), size=3))
        rgb = 'rgb(0,0,0)'
        #rgb = f'rgb({colors[0]},{colors[1]},{colors[2]})'
        
        fig = go.Figure(data=go.Scatter(x=self.periods,
                                        y=np.concatenate([np.array([0]), shift]),
                                        #y=[0] * len(self.periods),
                                        mode='lines+markers+text',
                                        text=polysemy,
                                        hovertemplate="""Clusters: %{text} <br> Shift: %{y}<extra></extra>""",
                                        textposition="top center",
                                        fill='tozeroy',
                                        marker=dict(
                                            size=[min(p*2.5, 40) for p in polysemy],
                                            color=rgb,
                                            opacity=1,
                                            symbol='square',
                                            line=dict(
                                                width=0,
                                            )),
                                        line=dict(
                                            color=rgb
                                        )))

        fig.update_layout(
            title=f'{self.word}',
            height=height,
            width=width,
            template='plotly_white',
            showlegend=False,
            xaxis = dict(
                tickmode='array',
                tickvals=list(self.periods),
                range=[self.periods[0]-0.5, self.periods[-1]+0.5],
                #ticktext = [f'{str(c-1)}-{str(c)}' for c in list(self.periods)]
            ),
            yaxis = dict(
                range=[0, max(polysemy)+0.1]
        ))
        
        #for period, shift in enumerate(self.metrics['word_shift'].values[0]):
        #    fig.add_annotation(x=period+1.5, y=0, text=format(shift, '.2f'), showarrow=False, yshift=10)
            
        fig.update_annotations(
            font=dict(size=12)
        )
        
        if yrange == None:
            fig.update_yaxes(range=(0, shift.max() + 0.1))
        else:
            fig.update_yaxes(range=(0, yrange))
        
        return fig

    def tfidf(self,
              nlp=spacy.load('it_core_news_lg'), 
              pos=['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB'],
              window=5,
              vocab=None,
              stop_words=stopwords.words('italian'),
              top_n=5,
              ngram_range=(1,1)):
        
        vocab = set(vocab)
        
        # Lemmatize Texts
        texts_lemma = {}
        for c in self.texts:
            processed = list(nlp.pipe(self.texts[c]))
            lemmatized = [' '.join([token.lemma_ if token.pos_ in pos and token.text != self.word else token.text for token in p]) for p in processed]

            if window:
                windowed = []
                for l in lemmatized:
                    try:
                        i = l.index(self.word)
                        first = l[:i].split()[-window:]
                        last = l[i+len(self.word):].split()[:window]
                        first.extend(last)
                        windowed.append(' '.join(first))
                    except:
                        windowed.append(l)
                texts_lemma[c] = ''.join(windowed).replace('CLS', '').replace('SEP', '').replace('UNK', '')
            else:
                texts_lemma[c] = ''.join(lemmatized).replace('CLS', '').replace('SEP', '').replace('UNK', '')
                    
        texts_lemma = dict(sorted(texts_lemma.items()))
        
        vectorizer = TfidfVectorizer(stop_words=stop_words+[self.word], ngram_range=ngram_range)
        X = vectorizer.fit_transform(texts_lemma.values())
        features = vectorizer.get_feature_names_out()

        self.keywords = dict()
        for n, a in zip(texts_lemma, np.argsort(X.toarray())):
            if vocab:
                top_w = [w for w in features[a[::-1]] if w in vocab][:top_n]
            else:
                top_w = [w for w in features[a[::-1]][:top_n]]
            self.keywords[n] = top_w
    
    @property
    def metrics(self):
        prominence = pd.DataFrame(self.clusters).T.apply(lambda x: x / sum(x))
        prominence.fillna(0, inplace=True)
        polysemy = prominence.astype(bool).sum(axis=0)
        
        prominence.columns = self.periods
        polysemy.index = self.periods
        
        prominence['trend'] = prominence.apply(lambda x: self._trend_identifier(x), axis=1)
        polysemy['trend'] = self._trend_identifier(polysemy)
        
        polysemy = polysemy.to_frame().T
        
        sense_shift = pd.read_csv('../sense_shift.tsv', delimiter='\t')
        sense_shift = sense_shift.loc[sense_shift['word'] == self.word, :].drop('word', axis=1)
        sense_shift = sense_shift.loc[sense_shift['cluster'].isin(self.clusters.keys()), :].set_index('cluster')
        sense_shift.columns = self.periods[:-1]
        
        word_shift = pd.read_csv('../semantic_shift.tsv', delimiter='\t')
        word_shift = word_shift.loc[word_shift['word'] == self.word, :].drop('word', axis=1)
        word_shift.columns = self.periods[:-1]
        
        return {'prominence': prominence,
                'polysemy': polysemy,
                'sense_shift': sense_shift,
                'word_shift': word_shift}
    
    def _trend_identifier(self, values):
        try:
            return mk.original_test(np.trim_zeros(np.array(values)), alpha=0.1).trend
        except:
            return 'no trend'