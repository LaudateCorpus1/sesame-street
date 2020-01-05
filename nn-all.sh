#!/bin/sh
declare -a MODELS=(bert,bert-large-cased roberta,roberta-large xlnet,xlnet-large-cased)
declare -a TASKS=(alphanli hellaswag physicaliqa socialiqa)
declare -a EMBEDDERS=(ai2 st)

OLDIFS=$IFS
tmux set-option -g remain-on-exit on

IFS=','
for task in "${TASKS[@]}"; do
    for embedder in "${EMBEDDERS[@]}"; do
        for i in "${MODELS[@]}"; do
            set -- $i
            tmux kill-session -t "$task-$embedder-$1"
            tmux new-session -d -s "$task-$embedder-$1" "srun --partition=isi --mem=48GB --time=1200 --core-spec=8 /bin/sh nn.sh $1 $task $embedder"
        done
    done
done

IFS=$OLDIFS
