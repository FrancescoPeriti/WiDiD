#!/bin/sh
#SBATCH -A NAISS2023-5-119 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 5:00:00

tokenized_dataset="$1/LSC/SemEval-German"
max_length=512
layers=12
batch_size=64
output="$2/LSC/SemEval-German"
sampling=250
targets="$3/LSC/SemEval-German/targets.txt"
agg_sub_words='mean'

declare -a models=("xlm-roberta-base" "bert-base-multilingual-cased" "bert-base-german-cased")

for model in "${models[@]}"
do
   python src/embeddings.py -t "${tokenized_dataset}" -m "${model}" -M "${max_length}" -l "${layers}" -b "${batch_size}" -o "${output}" -n "${sampling}" -T "${targets}"
done
