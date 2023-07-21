#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/LSC"
current_folder="$(pwd)"
mkdir -p "${dataset_folder}/"
dataset_folder="$(realpath ${dataset_folder})"
swedish=SemEval-Swedish

# -- SemEval-Swedish --
wget https://zenodo.org/record/3730550/files/semeval2020_ulscd_swe.zip?download=1 -O semeval2020_ulscd_swe.zip
unzip semeval2020_ulscd_swe.zip
rm semeval2020_ulscd_swe.zip
mv semeval2020_ulscd_swe "${dataset_folder}/${swedish}/"

# --- Processing ---
declare -a corpora=("corpus1" "corpus2")
declare -a texts=("lemma" "token")

n=1
for corpus in "${corpora[@]}"
do
   for text in "${texts[@]}"
    do 
       cd "${dataset_folder}/${swedish}/${corpus}/${text}/"
       if [[ "$n" == 1 ]]; then
          gunzip "kubhist2a.txt.gz"
          mv kubhist2a.txt corpus1.txt
       else
          gunzip "kubhist2b.txt.gz"
          mv kubhist2b.txt corpus2.txt
       fi
    done
    n=$(($n+1))
done

rm "${dataset_folder}/${swedish}/README.html" "${dataset_folder}/${swedish}/README.md"
cd "${current_folder}"