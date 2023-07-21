#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/LSC"
current_folder="$(pwd)"
mkdir -p "${dataset_folder}/"
dataset_folder="$(realpath ${dataset_folder})"
spanish=LSCDiscovery-Spanish

# --- Processing ---
declare -a corpora=("corpus1" "corpus2")
declare -a texts=("lemma" "token")

# -- LSCDiscovery --
# - make folders
for corpus in "${corpora[@]}"
do
   for text in "${texts[@]}"
    do 
       mkdir -p "${dataset_folder}/${spanish}/${corpus}/${text}/"
    done
done

# - download old corpus
cd "${dataset_folder}/${spanish}/corpus1/"
wget https://users.dcc.uchile.cl/~fzamora/old_corpus.tar.bz2
tar -xvjf old_corpus.tar.bz2
rm old_corpus.tar.bz2
mv dataset_XIX_lemmatized.txt corpus1.txt
mv corpus1.txt lemma
mv dataset_XIX_raw.txt corpus1_raw.txt
mv corpus1_raw.txt token
mv dataset_XIX_tokenized.txt corpus1.txt
mv corpus1.txt token
rm *.txt

# - download modern corpus
cd "${dataset_folder}/${spanish}/corpus2/"
wget https://users.dcc.uchile.cl/~fzamora/modern_corpus.tar.bz2
tar -xvjf modern_corpus.tar.bz2
rm modern_corpus.tar.bz2
mv modern_corpus_lemmatized.txt corpus2.txt
mv corpus2.txt lemma
mv modern_corpus_raw.txt corpus2_raw.txt
mv corpus2_raw.txt token
mv modern_corpus_tokenized.txt corpus2.txt
mv corpus2.txt token
rm *.txt

# - download targets and scores
cd "${dataset_folder}/${spanish}"
wget https://zenodo.org/record/6433203/files/dwug_es.zip?download=1 -O dwug_es_development.zip
unzip "dwug_es_development.zip" -d "dwug_es_development" && rm dwug_es_development.zip
targets_development="${dataset_folder}/${spanish}/dwug_es_development/dwug_es/stats/opt/stats_groupings.csv"
cut -f1 "${targets_development}" | tail -n+2 > targets_development.txt
wget https://zenodo.org/record/6433398/files/dwug_es.zip?download=1 -O dwug_es_test.zip
unzip "dwug_es_test.zip" -d "dwug_es_test" && rm dwug_es_test.zip
targets_test="${dataset_folder}/${spanish}/dwug_es_test/dwug_es/stats/opt/stats_groupings.csv"
cut -f1 "${targets_test}" | tail -n+2 > targets_test.txt
cp targets_development.txt targets.txt
cut -f1 "${targets_test}" | tail -n+2 >> targets.txt
mkdir truth
cut -f1,12 "${targets_test}" | tail -n+2 > "${dataset_folder}/${spanish}/truth/binary_test.txt"
cut -f1,15 "${targets_test}" | tail -n+2 > "${dataset_folder}/${spanish}/truth/graded_test.txt"
cut -f1,12 "${targets_development}" | tail -n+2 > "${dataset_folder}/${spanish}/truth/binary_development.txt"
cut -f1,15 "${targets_development}" | tail -n+2 > "${dataset_folder}/${spanish}/truth/graded_development.txt"
cp "${dataset_folder}/${spanish}/truth/binary_development.txt" "${dataset_folder}/${spanish}/truth/binary.txt"
cp "${dataset_folder}/${spanish}/truth/graded_development.txt" "${dataset_folder}/${spanish}/truth/graded.txt"
cut -f1,12 "${targets_test}" | tail -n+2 >> "${dataset_folder}/${spanish}/truth/binary.txt"
cut -f1,15 "${targets_test}" | tail -n+2 >> "${dataset_folder}/${spanish}/truth/graded.txt"
rm -rf "${dataset_folder}/${spanish}/dwug_es_development/"
rm -rf "${dataset_folder}/${spanish}/dwug_es_test/"
cd "${current_folder}"
