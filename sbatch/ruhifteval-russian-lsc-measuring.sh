#!/bin/sh
#SBATCH -A NAISS2023-5-119 -p alvis
#SBATCH -C NOGPU #N 1 --gpus-per-node=T4:4 # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 6:00:00

declare -a models=("xlm-roberta-base") #"bert-base-multilingual-cased" "DeepPavlov_rubert-base-cased")
declare -a embeddings_folder=("$1/LSC/GlossReader12-Russian" "$1/LSC/GlossReader23-Russian" "$1/LSC/GlossReader13-Russian")
declare -a labels_folder=("$2/LSC/GlossReader12-Russian" "$2/LSC/GlossReader23-Russian" "$2/LSC/GlossReader13-Russian")
declare -a scores_folder=("$3/LSC/GlossReader12-Russian" "$3/LSC/GlossReader23-Russian" "$3/LSC/GlossReader13-Russian")
declare -a targets_filename=("$4/LSC/GlossReader12-Russian/targets_test.txt" "$4/LSC/GlossReader23-Russian/targets_test.txt" "$4/LSC/GlossReader13-Russian/targets_test.txt")
declare -a targets_filename_40=("$4/LSC/GlossReader12-Russian/targets.txt" "$4/LSC/GlossReader23-Russian/targets.txt" "$4/LSC/GlossReader13-Russian/targets.txt")


i=0
for embedding_folder in "${embeddings_folder[@]}"
do
   for model in "${models[@]}"
   do
       python src/lsc_measuring.py -e "${embedding_folder}" -m "${model}" -L "${labels_folder[i]}" -o "${scores_folder[i]}/${model}" -t "${targets_filename[i]}"
   done
    i=$(($i+1))
done


