#!/bin/sh
#SBATCH -A NAISS2023-5-119 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 5:00:00

declare -a models=("xlm-roberta-base" "bert-base-multilingual-cased" "DeepPavlov/rubert-base-cased")
declare -a tokenized_datasets=("$1/LSC/GlossReader12-Russian" "$1/LSC/GlossReader23-Russian" "$1/LSC/GlossReader13-Russian")
declare -a outputs=("$2/LSC/GlossReader12-Russian" "$2/LSC/GlossReader23-Russian" "$2/LSC/GlossReader13-Russian")
declare -a targets=("$3/LSC/GlossReader12-Russian/targets_test.txt" "$3/LSC/GlossReader23-Russian/targets_test.txt" "$3/LSC/GlossReader13-Russian/targets_test.txt")

max_length=512
layers=12
batch_size=64
sampling=0
agg_sub_words='mean'

i=0
for tokenized_dataset in "${tokenized_datasets[@]}"
do
   for model in "${models[@]}"
   do
      python src/embeddings.py -t "${tokenized_dataset}" -m "${model}" -M "${max_length}" -l "${layers}" -b "${batch_size}" -o "${outputs[i]}" -n "${sampling}" -T "${targets[i]}"
   done
    i=$(($i+1))
done
