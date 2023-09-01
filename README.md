# WiDiD
Studying Word Meaning Evolution through Incremental Semantic Shift Detection: a Case-Study on the Italian Parliamentary Speeches

## Evaluation
<b> Download data </b>
```
bash sbatch/semeval-english-download.sh datasets
bash sbatch/semeval-swedish-download.sh datasets
bash sbatch/semeval-latin-download.sh datasets
bash sbatch/semeval-german-download.sh datasets
bash sbatch/rushifteval-russian-download.sh datasets tokenization
bash sbatch/diacrita-italian-download.sh datasets
bash sbatch/lscdiscovery-spanish-download.sh datasets
```
<b> Tokenization </b>
```
bash sbatch/semeval-english-tokenization.sh datasets tokenization
bash sbatch/semeval-german-tokenization.sh datasets tokenization
bash sbatch/semeval-latin-tokenization.sh datasets tokenization
bash sbatch/semeval-swedish-tokenization.sh datasets tokenization
bash sbatch/diacrita-italian-tokenization.sh datasets tokenization
bash sbatch/lsdiscovery-spanish-tokenization.sh datasets tokenization
```
<b> Embedding extraction </b>
```
bash sbatch/semeval-english-extraction.sh tokenization embeddings datasets
bash sbatch/semeval-german-extraction.sh tokenization embeddings datasets
bash sbatch/semeval-latin-extraction.sh tokenization embeddings datasets
bash sbatch/semeval-swedish-extraction.sh tokenization embeddings datasets
bash sbatch/diacrita-italian-extraction.sh tokenization embeddings datasets
bash sbatch/lscdiscovery-spanish-extraction.sh tokenization embeddings datasets
bash sbatch/rushifteval-russian-extraction.sh tokenization embeddings datasets
```
<b> Clustering </b>
```
bash sbatch/semeval-english-clustering.sh embeddings labels datasets
bash sbatch/diacrita-italian-clustering.sh embeddings labels datasets
bash sbatch/rushifteval-russian-clustering.sh embeddings labels datasets
bash sbatch/semeval-german-clustering.sh embeddings labels datasets
bash sbatch/semeval-latin-clustering.sh embeddings labels datasets
bash sbatch/semeval-swedish-clustering.sh embeddings labels datasets
bash sbatch/lscdiscovery-spanish-clustering.sh embeddings labels datasets
```
<b> LSC measuring </b>
```
bash sbatch/semeval-english-lsc-measuring.sh embeddings labels scores datasets
bash sbatch/diacrita-italian-lsc-measuring.sh embeddings labels scores datasets
bash sbatch/rushifteval-russian-measuring.sh" embeddings labels scores datasets
bash sbatch/semeval-german-lsc-measuring.sh embeddings labels scores datasets
bash sbatch/semeval-latin-lsc-measuring.sh embeddings labels scores datasets
bash sbatch/semeval-swedihs-lsc-measuring.sh embeddings labels scores datasets
bash sbatch/lscdiscovery-spanish-lsc-measuring.sh embeddings labels scores datasets
```

## Case Study
```
unzip data.zip
```

```
python case-study-application.py
```

```
python case-study-visualization.py
```

