#!/bin/sh
declare -a MODELS=(bert,bert-large-cased roberta,roberta-large xlnet,xlnet-large-cased)
declare -a TASKS=(physicaliqa socialiqa)
declare -a DATA=(train dev)
declare -a EMBEDDERS=(ai2 st)

OLDIFS=$IFS
tmux set-option -g remain-on-exit on

IFS=','
for task in "${TASKS[@]}"; do
    for data in "${DATA[@]}"; do
        for embedder in "${EMBEDDERS[@]}"; do
            for i in "${MODELS[@]}"; do
                set -- $i
                tmux kill-session -t "$task-$data-$embedder-$2"
                tmux new-session -d -s "$task-$data-$embedder-$2" "srun --partition=isi --mem=48GB --time=1200 --core-spec=8 --gres=gpu:p100:1 /bin/sh embed.sh $1 $2 $task $data $embedder"
            done
        done
    done
done

IFS=$OLDIFS
