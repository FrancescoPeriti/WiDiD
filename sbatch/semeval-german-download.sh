#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/LSC"
current_folder="$(pwd)"
mkdir -p "${dataset_folder}/"
dataset_folder="$(realpath ${dataset_folder})"
german=SemEval-German

# -- SemEval-German --
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip
unzip semeval2020_ulscd_ger.zip
rm semeval2020_ulscd_ger.zip
mv semeval2020_ulscd_ger "${dataset_folder}/${german}"

# --- Processing ---
declare -a corpora=("corpus1" "corpus2")
declare -a texts=("lemma" "token")

n=1
for corpus in "${corpora[@]}"
do
   for text in "${texts[@]}"
    do 
       cd "${dataset_folder}/${german}/${corpus}/${text}/"
       if [[ "$n" == 1 ]]; then
          gunzip "dta.txt.gz"
          mv dta.txt corpus1.txt
       else
          gunzip "bznd.txt.gz"
          mv bznd.txt corpus2.txt
       fi
    done
    n=$(($n+1))
done

rm "${dataset_folder}/${german}/README.html" "${dataset_folder}/${german}/README.md"
cd "${current_folder}"
