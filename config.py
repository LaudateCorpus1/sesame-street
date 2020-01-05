predictions = {
    'roberta': {
        "alphanli": "output/roberta-roberta-large-alphanli-pred",
        "hellaswag": "output/roberta-roberta-large-hellaswag-pred",
        "piqa": "output/roberta-roberta-large-physicaliqa-pred",
        "siqa": "output/roberta-roberta-large-socialiqa-pred"
    },
    'bert': {
        "alphanli":         "output/bert-bert-large-cased-alphanli-pred",
        "hellaswag":    "output/bert-bert-large-cased-hellaswag-pred",
        "piqa":         "output/bert-bert-large-cased-physicaliqa-pred",
        "siqa":         "output/bert-bert-large-cased-socialiqa-pred"
    },
    'xlnet': {
        "alphanli":         "output/xlnet-xlnet-large-cased-alphanli-pred",
        "hellaswag":    "output/xlnet-xlnet-large-cased-hellaswag-pred",
        "piqa":         "output/xlnet-xlnet-large-cased-physicaliqa-pred",
        "siqa":         "output/xlnet-xlnet-large-cased-socialiqa-pred"
    }
}

datasets = {
    "alphanli": {
        "labels":   {
            "train": "cache/alphanli-train-dev/train-labels.lst",
            "dev": "cache/alphanli-train-dev/dev-labels.lst",
        },
        "data":     {
            "train": "cache/alphanli-train-dev/train.jsonl",
            "dev": "cache/alphanli-train-dev/dev.jsonl"
        },
        "offset": 1,
        "ctx": lambda x: x['obs1'] + "\n" + x["obs2"],
        "choices": lambda x: [
            {"text": x["hyp1"], "models": [], "correct": False},
            {"text": x["hyp2"], "models": [], "correct": False},
        ],
        "num_choices": 2
    },
    "hellaswag": {
        "labels": {
            "train": "cache/hellaswag-train-dev/hellaswag-train-dev/train-labels.lst",
            "dev": "cache/hellaswag-train-dev/hellaswag-train-dev/valid-labels.lst",
        },
        "data": {
            "train": "cache/hellaswag-train-dev/hellaswag-train-dev/train.jsonl",
            "dev": "cache/hellaswag-train-dev/hellaswag-train-dev/valid.jsonl"
        },
        "offset": 0,
        "ctx": lambda x: x["ctx"],
        "choices": lambda x: [
            {"text": y, "models": [], "correct": False} for y in x["ending_options"]
        ],
        "num_choices": 4
    },
    "piqa": {
        "labels": {
            "train": "cache/physicaliqa-train-dev/physicaliqa-train-dev/train-labels.lst",
            "dev": "cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst",
        },
        "data": {
            "train": "cache/physicaliqa-train-dev/physicaliqa-train-dev/train.jsonl",
            "dev": "cache/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl",
        },
        "offset": 0,
        "ctx": lambda x: x['goal'],
        "choices": lambda x: [
           {"text": x["sol1"], "models": [], "correct": False},
           {"text": x["sol2"], "models": [], "correct": False},
        ],
        "num_choices": 2
    },
    "siqa": {
        "labels": {
            "train": "cache/socialiqa-train-dev/socialiqa-train-dev/train-labels.lst",
            "dev": "cache/socialiqa-train-dev/socialiqa-train-dev/dev-labels.lst",
        },
        "data": {
            "train": "cache/socialiqa-train-dev/socialiqa-train-dev/train.jsonl",
            "dev": "cache/socialiqa-train-dev/socialiqa-train-dev/dev.jsonl",
        },
        "offset": 1,
        "ctx": lambda x: x["context"] + "\n" + x['question'],
        "choices": lambda x: [
            {"text": x["answerA"], "models": [], "correct": False},
            {"text": x["answerA"], "models": [], "correct": False},
            {"text": x["answerB"], "models": [], "correct": False},
        ],
        "num_choices": 3
    }
}

embeddings = {
    "roberta":{
        "alphanli":{
            "train":{
                "ai2": "data/roberta-large-alphanli-train-ai2.df",
                "st": "data/roberta-large-alphanli-train-st.df"
            },
            "dev": {
                "ai2": "data/roberta-large-alphanli-dev-ai2.df",
                "st": "data/roberta-large-alphanli-dev-st.df"
            }
        }
    }
}

closest_indices = {
    "alphanli": {
        "roberta": {
            "ai2": "data/roberta-alphanli-ai2.rank",
            "st": None,
        },
        "bert": {
            "ai2": "data/roberta-alphanli-ai2.rank",
            "st": None,
        },
        "xlnet": {
            "ai2": "data/roberta-alphanli-ai2.rank",
            "st": None,
        }
    }
}
