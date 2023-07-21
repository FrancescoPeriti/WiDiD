#!/bin/sh
#SBATCH -A NAISS2023-22-226 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 2-00:00:00

dataset="$1/LSC/LSCDiscovery-Spanish"
sampling=0
output="$2/LSC/LSCDiscovery-Spanish"
tokenization_class="StandardSpacyTokenization"
model="es_core_news_sm"

python3 -m spacy download "${model}"
python3 "src/tokenization.py" -d "${dataset}" -n "${sampling}" -o "${output}" -t "${tokenization_class}" -m "${model}"
