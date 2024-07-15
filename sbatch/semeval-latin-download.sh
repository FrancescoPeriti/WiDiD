#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/LSC"
current_folder="$(pwd)"
mkdir -p "${dataset_folder}/"
dataset_folder="$(realpath ${dataset_folder})"
latin=SemEval-Latin

# -- SemEval-Latin --
wget https://zenodo.org/record/3992738/files/semeval2020_ulscd_lat.zip?download=1 -O semeval2020_ulscd_lat.zip
unzip semeval2020_ulscd_lat.zip
rm semeval2020_ulscd_lat.zip
mv semeval2020_ulscd_lat "${dataset_folder}/${latin}/"

# --- Processing ---
declare -a corpora=("corpus1" "corpus2")
declare -a texts=("lemma" "token")

n=1
for corpus in "${corpora[@]}"
do
   for text in "${texts[@]}"
    do 
       cd "${dataset_folder}/${latin}/${corpus}/${text}/"
       if [[ "$n" == 1 ]]; then
          gunzip "LatinISE1.txt.gz"
          mv LatinISE1.txt corpus1.txt
       else
          gunzip "LatinISE2.txt.gz"
          mv LatinISE2.txt corpus2.txt
       fi
    done
    n=$(($n+1))
done

rm "${dataset_folder}/${latin}/README.html" "${dataset_folder}/${latin}/README.md"
cd "${current_folder}"

# install spacy model
pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl
