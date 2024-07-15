#!/bin/sh
#SBATCH -A NAISS2023-5-119 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 6:00:00

embedding_folder="$1/LSC/SemEval-English"
label_folder="$2/LSC/SemEval-English"
score_folder="$3/LSC/SemEval-English"
dataset_folder="$4/LSC/SemEval-English"

declare -a models=("bert-base-uncased" "xlm-roberta-base" "bert-base-multilingual-cased")

for model in "${models[@]}"
do
   python src/lsc_measuring.py -e "${embedding_folder}" -m "${model}" -L "${label_folder}" -o "${score_folder}/${model}" -t "${dataset_folder}/targets.txt"
done
