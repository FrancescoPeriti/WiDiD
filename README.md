# WiDiD
Word Meaning Evolution through Incremental Semantic Shift Detection: a Case-Study on the Italian Parliamentary Speeches

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
sbatch sbatch/tokenization/semeval-english-tokenization.sh datasets tokenization
sbatch sbatch/tokenization/semeval-german-tokenization.sh datasets tokenization
sbatch sbatch/tokenization/semeval-latin-tokenization.sh datasets tokenization
sbatch sbatch/tokenization/semeval-swedish-tokenization.sh datasets tokenization
sbatch sbatch/tokenization/diacrita-italian-tokenization.sh datasets tokenization
sbatch sbatch/tokenization/lsdiscovery-spanish-tokenization.sh datasets tokenization
```


## Case Study
```
unzip data.zip
```

```
python case-study-application.py
```
