#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-01-03 09:25:19
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Description: embed a dataset with a model


import os
import sys
import json
import pickle
from glob import glob

import torch
import numpy as np
import yaml

from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser
from torch.utils.data import DataLoader, RandomSampler
from huggingface import HuggingFaceClassifier
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LightningModuleEmbedder:

    def __init__(self, hparams):
        self.model = HuggingFaceClassifier.load_from_metrics(
            hparams=hparams,
            weights_path=hparams.weights_path,
            tags_csv=hparams.tags_csv,
            on_gpu=torch.cuda.is_available(),
            map_location=None
        )
        self.model.to(device)

    def embed(self, dataset, output, collate_fn):

        data = []
        count = 0


        for i, batch in tqdm(enumerate(DataLoader(dataset, shuffle=False, batch_size=4, collate_fn=collate_fn))):

            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch["token_type_ids"] = batch["token_type_ids"].to(device)
            batch["y"] = batch["y"].to(device)

            batch_size, num_choice, seq = batch["input_ids"].shape

            with torch.no_grad():

                embeds = self.model.intermediate(
                    batch["input_ids"].reshape(-1, seq),
                    batch["token_type_ids"].reshape(-1, seq),
                    batch["attention_mask"].reshape(-1, seq))

                embeds = embeds.reshape(batch_size, num_choice, -1)
                embeds = embeds.cpu().detach().tolist()

                for x in range(len(embeds)):
                    for y in range(len(embeds[x])):
                        data.append({
                            "index": count,
                            "embedding": embeds[x][y]
                        })

                        count += 1

        df = pd.DataFrame(data)
        df.to_csv(output, sep='\t', index=False)


class SentenceTransformerEmbedder:

    def __init__(self, hparams):
        self.model = SentenceTransformer(f'{hparams.model_type}-large-nli-stsb-mean-tokens')
        self.model.to(device)

    def embed(self, dataset, output, collate_fn, detok):

        count = 0
        data = []


        for _, batch in tqdm(enumerate(DataLoader(dataset, shuffle=False, batch_size=4, collate_fn=collate_fn))):


            sentences = []
            for i in range(len(batch['tokens'])):
                sentences.extend([detok(batch['tokens'][i][j]).replace("<s>", "").replace("</s>", " ").replace("[CLS]", "").replace("[SEP]", " ") for j in range(len(batch['tokens'][i]))])

            with torch.no_grad():

                embeds = [self.model.encode([e])[0].tolist() for e in sentences]

                for embed in embeds:

                    data.append({
                        "index": count,
                        "embedding": embed
                    })

                    count += 1

        df = pd.DataFrame(data)
        df.to_csv(output, sep='\t', index=False)


def main(hparams):

    ai2model = LightningModuleEmbedder(hparams)
    if hparams.embedder == "ai2":
        if hparams.dataset == "train":
            ai2model.embed(ai2model.model.train_dataloader.dataset, hparams.output, ai2model.model.collate_fn)
        elif hparams.dataset == "dev":
            ai2model.embed(ai2model.model.val_dataloader.dataset, hparams.output, ai2model.model.collate_fn)
    elif hparams.embedder == "st":
        embedder = SentenceTransformerEmbedder(hparams)
        if hparams.dataset == "train":
            embedder.embed(ai2model.model.train_dataloader.dataset, hparams.output, ai2model.model.collate_fn, ai2model.model.tokenizer.tokenizer.convert_tokens_to_string)
        elif hparams.dataset == "dev":
            embedder.embed(ai2model.model.val_dataloader.dataset, hparams.output, ai2model.model.collate_fn, ai2model.model.tokenizer.tokenizer.convert_tokens_to_string)


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]

    # for huggingface model only
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=True)
    add_default_args(parent_parser, root_dir)
    parser = HuggingFaceClassifier.add_model_specific_args(parent_parser)

    # general arguments
    parser.add_argument("--dataset", type=str, choices=["train", "dev"])
    parser.add_argument("--output", type=str)
    parser.add_argument("--embedder", type=str, choices=['ai2', 'st'], help="Choice from fine-tuned ai2 models(ai2) or sentence transfomers(st)")

    hyperparams = parser.parse_args()
    main(hyperparams)


