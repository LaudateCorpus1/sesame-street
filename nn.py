import pandas as pd
from skhubness import Hubness, neighbors
import numpy as np
import os
from tqdm import tqdm
from config import *
# pylint: disable=no-member



def load_dataset(task_name, order=None):
    assert task_name in datasets, "task not found"

    with open(datasets[task_name]["labels"]["train"]) as tf, open(datasets[task_name]["labels"]["dev"]) as df:

        train_labels = list(map(int, tf.readlines()))
        dev_labels = list(map(int, df.readlines()))

    train_dataset = pd.read_json(datasets[task_name]["data"]["train"], orient="records", lines=True)
    dev_dataset = pd.read_json(datasets[task_name]["data"]["dev"], orient="records", lines=True)

    train_dataset = train_dataset.to_dict("records")
    dev_dataset = dev_dataset.to_dict("records")

    def process(args):
        d, i = args
        dd = {
            "ctx": datasets[task_name]["ctx"](d),
            "choices": datasets[task_name]["choices"](d)
        }

        dd["choices"][i - datasets[task_name]["offset"]]["correct"] = True

        return dd

    train_dataset = list(map(process, zip(train_dataset, train_labels)))
    dev_dataset = list(map(process, zip(dev_dataset, dev_labels)))

    if order is not None:
        dev_dataset = [dev_dataset[i] for i in order]

    return train_dataset, dev_dataset

def load_predictions(path, order=None):

    with open(os.path.join(path, "dev-predictions.lst")) as predfile, \
        open(os.path.join(path, "dev-probabilities.lst")) as probafile, \
            open(os.path.join(path, "dev-labels.lst")) as labelfile:

            labels = list(map(int, labelfile.readlines()))
            predictions = list(map(int, predfile.readlines()))
            probabilities = list(map(lambda l: list(map(float, l.split('\t'))), probafile.readlines()))

    if order is not None:
        labels = [labels[i] for i in order]
        predictions = [predictions[i] for i in order]
        probabilities = [probabilities[i] for i in order]

    return predictions, probabilities, labels

def load_embeddings(path):

    embeddings = []
    with open(path) as f:
        for line in tqdm(f.readlines()[1:]):
            *_, embedding = line.split('\t')
            embedding = np.asarray(eval(embedding))
            embeddings.append(embedding)
            if len(embeddings) > 1:
                assert embeddings[-1].shape == embeddings[-2].shape

    return np.asarray(embeddings)



class EmbeddingAnalyzer:


    def __init__(self, k=10, metric="Cosine", algorithm="nng", algorithm_params={"optimize": True, "n_candidates": 5}):
        self.neighbor = neighbors.NearestNeighbors(
            n_neighbors=5,
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            hubness="mutual_proximity",
            leaf_size=30, metric=metric, n_jobs=8)

    def closest(self, training_embedding, dev_embedding, output):

        self.neighbor.fit(training_embedding)
        distances, candidates = self.neighbor.kneighbors(dev_embedding)
        np.savetxt(output, candidates, delimiter="\t")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Nearest Neighbor Tool")
    parser.add_argument("--model", choices=["roberta", "xlnet", "bert"], required=True)
    parser.add_argument("--task", choices=["alphanli", "hellaswag", "physicaliqa", "socialiqa"], required=True)
    parser.add_argument("--embedder", choices=["ai2", "st"], required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    train_dataset, dev_dataset = load_dataset(args.task)
    predictions, probablities, labels = load_predictions(predictions[args.model][args.task])

    dev_embeddings = load_embeddings(embeddings[args.model][args.task]["dev"][args.embedder])
    train_embeddings = load_embeddings(embeddings[args.model][args.task]["train"][args.embedder])

    analyzer = EmbeddingAnalyzer()

    analyzer.closest(train_embeddings, dev_embeddings, args.output)






