#!/bin/bash

# --- Create dataset folder ---
dataset_folder="$1/LSC"
current_folder="$(pwd)"
tokenization_folder="$2/LSC"
mkdir -p "${dataset_folder}/"
mkdir -p "${tokenization_folder}/"
dataset_folder="$(realpath ${dataset_folder})"
tokenization_folder="$(realpath ${tokenization_folder})"
russian12=GlossReader12-Russian
russian23=GlossReader23-Russian
russian13=GlossReader13-Russian

# --- Create dir ---
mkdir -p "${dataset_folder}/${russian12}/truth"
mkdir -p "${dataset_folder}/${russian23}/truth"
mkdir -p "${dataset_folder}/${russian13}/truth"

# --- Download gold scores ---
wget https://raw.githubusercontent.com/akutuzov/rushifteval_public/main/annotated_devset.tsv -O targets_development.txt
wget https://raw.githubusercontent.com/akutuzov/rushifteval_public/main/annotated_testset.tsv -O targets_test.txt
cut -f1 targets_development.txt > "${dataset_folder}/${russian12}/targets_development.txt"
cut -f1 targets_development.txt > "${dataset_folder}/${russian23}/targets_development.txt"
cut -f1 targets_development.txt > "${dataset_folder}/${russian13}/targets_development.txt"
cut -f1 targets_test.txt > "${dataset_folder}/${russian12}/targets_test.txt"
cut -f1 targets_test.txt > "${dataset_folder}/${russian23}/targets_test.txt"
cut -f1 targets_test.txt > "${dataset_folder}/${russian13}/targets_test.txt"

cut -f1 targets_development.txt > "${dataset_folder}/${russian12}/targets_full.txt"
cut -f1 targets_development.txt > "${dataset_folder}/${russian23}/targets_full.txt"
cut -f1 targets_development.txt > "${dataset_folder}/${russian13}/targets_full.txt"
cut -f1 targets_test.txt >> "${dataset_folder}/${russian12}/targets_full.txt"
cut -f1 targets_test.txt >> "${dataset_folder}/${russian23}/targets_full.txt"
cut -f1 targets_test.txt >> "${dataset_folder}/${russian13}/targets_full.txt"

cut -f1,2 targets_development.txt > "${dataset_folder}/${russian12}/truth/graded_development.txt"
cut -f1,3 targets_development.txt > "${dataset_folder}/${russian23}/truth/graded_development.txt"
cut -f1,4 targets_development.txt > "${dataset_folder}/${russian13}/truth/graded_development.txt"
cut -f1,2 targets_test.txt > "${dataset_folder}/${russian12}/truth/graded_test.txt"
cut -f1,3 targets_test.txt > "${dataset_folder}/${russian23}/truth/graded_test.txt"
cut -f1,4 targets_test.txt > "${dataset_folder}/${russian13}/truth/graded_test.txt"

cut -f1,2 targets_development.txt > "${dataset_folder}/${russian12}/truth/graded_full.txt"
cut -f1,3 targets_development.txt > "${dataset_folder}/${russian23}/truth/graded_full.txt"
cut -f1,4 targets_development.txt > "${dataset_folder}/${russian13}/truth/graded_full.txt"
cut -f1,2 targets_test.txt >> "${dataset_folder}/${russian12}/truth/graded_full.txt"
cut -f1,3 targets_test.txt >> "${dataset_folder}/${russian23}/truth/graded_full.txt"
cut -f1,4 targets_test.txt >> "${dataset_folder}/${russian13}/truth/graded_full.txt"

rm targets_development.txt
rm targets_test.txt

# --- Processing ---
unzip data.zip

python "${current_folder}/src/processing/russian_processing.py" -r "data/RuShiftEval/epoch-12/samples" -d "${dataset_folder}/${russian12}" -t "${tokenization_folder}/${russian12}" 
python "${current_folder}/src/processing/russian_processing.py" -r "data/RuShiftEval/epoch-23/samples" -d "${dataset_folder}/${russian23}" -t "${tokenization_folder}/${russian23}"
python "${current_folder}/src/processing/russian_processing.py" -r "data/RuShiftEval/epoch-13/samples" -d "${dataset_folder}/${russian13}" -t "${tokenization_folder}/${russian13}"

rm -rf data
#rm -rf data.zip
cd "${current_folder}"