#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/LSC"
current_folder="$(pwd)"
mkdir -p "${dataset_folder}/"
dataset_folder="$(realpath ${dataset_folder})"
italian=DiacrIta-Italian

# --- Create dir ---
mkdir -p "${dataset_folder}/${italian}/truth"

# -- DiacrIta-Italian --
echo "DiacrIta: download data from here -> https://diacr-ita.github.io/DIACR-Ita/ and rename the zip as 'diacrita.zip'"
wget https://www.dropbox.com/sh/i2tcpa390kywow0/AACjC6hnxfFuNGiDlT8ht8U8a?dl=0 -O diacrita.zip
unzip diacrita.zip
rm diacrita.zip
gunzip T0.txt.gz
gunzip T1.txt.gz

# --- Processing ---
declare -a corpora=("corpus1" "corpus2")
declare -a texts=("lemma" "token")

for corpus in "${corpora[@]}"
do 
    for text in "${texts[@]}"
    do
        mkdir -p "${dataset_folder}/${italian}/${corpus}/${text}"
    done
done

python "${current_folder}/src/processing/diacrita-italian-processing.py" T0.txt "${dataset_folder}/${italian}/corpus1/token/corpus1.txt" "${dataset_folder}/${italian}/corpus1/lemma/corpus1.txt" 
python "${current_folder}/src/processing/diacrita-italian-processing.py" T1.txt "${dataset_folder}/${italian}/corpus2/token/corpus2.txt" "${dataset_folder}/${italian}/corpus2/lemma/corpus2.txt" 
rm T0.txt T1.txt
mkdir "${dataset_folder}/${italian}/truth"
wget https://raw.githubusercontent.com/diacr-ita/data/master/test/gold.txt -O binary.txt
mv binary.txt "${dataset_folder}/${italian}/truth"
awk -v OFS="\t" '$1=$1' "${dataset_folder}/${italian}/truth/binary.txt" > tmp.out.it
rm "${dataset_folder}/${italian}/truth/binary.txt"
mv tmp.out.it "${dataset_folder}/${italian}/truth/binary.txt"
cut -d ' ' -f1 "${dataset_folder}/${italian}/truth/binary.txt" > "${dataset_folder}/${italian}/targets.txt"

cd "${current_folder}"
