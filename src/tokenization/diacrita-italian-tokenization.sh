#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 2-00:00:00

dataset="$1/LSC/DiacrIta-Italian"
model="it_core_news_sm"
sampling=0
output="$2/LSC/DiacrIta-Italian"
tokenization_class="ItalianSpacyTokenization"

python3 -m spacy download "${model}"
python3 "src/tokenization.py" -d "${dataset}" -m "${model}" -n "${sampling}" -o "${output}" -t "${tokenization_class}"
