#!/bin/sh
#SBATCH -A NAISS2023-5-119 -p alvis
#SBATCH -C NOGPU
#SBATCH -t 3:00:00

layers=12

declare -a embeddings_folders=("$1/LSC/GlossReader12-Russian" "$1/LSC/GlossReader23-Russian" "$1/LSC/GlossReader13-Russian")
declare -a labels_folders=("$2/LSC/GlossReader12-Russian" "$2/LSC/GlossReader23-Russian" "$2/LSC/GlossReader13-Russian")
declare -a dataset_folders=("$3/LSC/GlossReader12-Russian" "$3/LSC/GlossReader23-Russian" "$3/LSC/GlossReader13-Russian")
declare -a algorithms=("app")
declare -a models=("xlm-roberta-base" "bert-base-multilingual-cased" "DeepPavlov_rubert-base-cased")


for model in "${models[@]}"
do
    python src/clustering.py -a "app_incremental" -e "$1/LSC/GlossReader12-Russian/${model}" -l "${layers}" -o "$2/LSC/GlossReader12-Russian/${model}" -t "$3/LSC/GlossReader12-Russian/targets_test.txt"
done

#for model in "${models[@]}"
#do
#    n=0
#    for embeddings_folder in "${embeddings_folders[@]}"
#    do
#	for algo in "${algorithms[@]}"
#	do
#	    python src/clustering.py -a "${algo}" -e "${embeddings_folder}/${model}" -l "${layers}" -o "${labels_folders[n]}/${model}" -t "${dataset_folders[n]}/targets_test.txt"
#	done
#     n=$(($n+1))
#    done
#done
