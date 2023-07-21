#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/LSC"
mkdir -p "${dataset_folder}/"
dataset_folder="$(realpath ${dataset_folder})"
current_folder="$(pwd)"
english=SemEval-English

# --- Download SemEval-English ---
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip
unzip semeval2020_ulscd_eng.zip
mv "semeval2020_ulscd_eng" "${dataset_folder}/${english}"
rm semeval2020_ulscd_eng.zip

# --- Processing ---
declare -a corpora=("corpus1" "corpus2")
declare -a texts=("lemma" "token")

n=1
for corpus in "${corpora[@]}"
do
   for text in "${texts[@]}"
    do
       cd "${dataset_folder}/${english}/${corpus}/${text}/"
       gunzip "ccoha${n}.txt.gz"
       mv "ccoha${n}.txt" "corpus${n}.txt"
    done
    n=$(($n+1))
done

python "${current_folder}/src/processing/semeval-english-processing.py" -d "${dataset_folder}/${english}"
cd "${current_folder}"
