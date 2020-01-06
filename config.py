predictions = {
    'roberta': {
        "alphanli": "output/roberta-roberta-large-alphanli-pred",
        "hellaswag": "output/roberta-roberta-large-hellaswag-pred",
        "physicaliqa": "output/roberta-roberta-large-physicaliqa-pred",
        "socialiqa": "output/roberta-roberta-large-socialiqa-pred"
    },
    'bert': {
        "alphanli":         "output/bert-bert-large-cased-alphanli-pred",
        "hellaswag":    "output/bert-bert-large-cased-hellaswag-pred",
        "physicaliqa":         "output/bert-bert-large-cased-physicaliqa-pred",
        "socialiqa":         "output/bert-bert-large-cased-socialiqa-pred"
    },
    'xlnet': {
        "alphanli":         "output/xlnet-xlnet-large-cased-alphanli-pred",
        "hellaswag":    "output/xlnet-xlnet-large-cased-hellaswag-pred",
        "physicaliqa":         "output/xlnet-xlnet-large-cased-physicaliqa-pred",
        "socialiqa":         "output/xlnet-xlnet-large-cased-socialiqa-pred"
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
    "physicaliqa": {
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
    "socialiqa": {
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
            {"text": x["answerB"], "models": [], "correct": False},
            {"text": x["answerC"], "models": [], "correct": False},
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
        },
        "hellaswag":{
            "train":{
                "ai2": "data/roberta-large-hellaswag-train-ai2.df",
                "st": "data/roberta-large-hellaswag-train-st.df"
            },
            "dev": {
                "ai2": "data/roberta-large-hellaswsocialiqai2.df",
                "st": "data/roberta-large-hellaswag-dev-st.df"
            }
        },
        "physicaliqa":{
            "train":{
                "ai2": "data/roberta-large-physicaliqa-train-ai2.df",
                "st": "data/roberta-large-physicaliqa-train-st.df"
            },
            "dev": {
                "ai2": "data/roberta-large-physicaliqa-dev-ai2.df",
                "st": "data/roberta-large-physicaliqa-dev-st.df"
            }
        },
        "socialiqa":{
            "train":{
                "ai2": "data/roberta-large-socialiqa-train-ai2.df",
                "st": "data/roberta-large-socialiqa-train-st.df"
            },
            "dev": {
                "ai2": "data/roberta-large-socialiqa-dev-ai2.df",
                "st": "data/roberta-large-socialiqa-dev-st.df"
            }
        }
    },
    "bert":{
        "alphanli":{
            "train":{
                "ai2": "data/bert-large-cased-alphanli-train-ai2.df",
                "st": "data/bert-large-cased-alphanli-train-st.df"
            },
            "dev": {
                "ai2": "data/bert-large-cased-alphanli-dev-ai2.df",
                "st": "data/bert-large-cased-alphanli-dev-st.df"
            }
        },
        "hellaswag":{
            "train":{
                "ai2": "data/bert-large-cased-hellaswag-train-ai2.df",
                "st": "data/bert-large-cased-hellaswag-train-st.df"
            },
            "dev": {
                "ai2": "data/bert-large-cased-hellaswsocialiqai2.df",
                "st": "data/bert-large-cased-hellaswag-dev-st.df"
            }
        },
        "physicaliqa":{
            "train":{
                "ai2": "data/bert-large-cased-physicaliqa-train-ai2.df",
                "st": "data/bert-large-cased-physicaliqa-train-st.df"
            },
            "dev": {
                "ai2": "data/bert-large-cased-physicaliqa-dev-ai2.df",
                "st": "data/bert-large-cased-physicaliqa-dev-st.df"
            }
        },
        "socialiqa":{
            "train":{
                "ai2": "data/bert-large-cased-socialiqa-train-ai2.df",
                "st": "data/bert-large-cased-socialiqa-train-st.df"
            },
            "dev": {
                "ai2": "data/bert-large-cased-socialiqa-dev-ai2.df",
                "st": "data/bert-large-cased-socialiqa-dev-st.df"
            }
        }
    },
    "xlnet":{
        "alphanli":{
            "train":{
                "ai2": "data/xlnet-large-cased-alphanli-train-ai2.df",
                "st": "data/xlnet-large-cased-alphanli-train-st.df"
            },
            "dev": {
                "ai2": "data/xlnet-large-cased-alphanli-dev-ai2.df",
                "st": "data/xlnet-large-cased-alphanli-dev-st.df"
            }
        },
        "hellaswag":{
            "train":{
                "ai2": "data/xlnet-large-cased-hellaswag-train-ai2.df",
                "st": "data/xlnet-large-cased-hellaswag-train-st.df"
            },
            "dev": {
                "ai2": "data/xlnet-large-cased-hellaswsocialiqai2.df",
                "st": "data/xlnet-large-cased-hellaswag-dev-st.df"
            }
        },
        "physicaliqa":{
            "train":{
                "ai2": "data/xlnet-large-cased-physicaliqa-train-ai2.df",
                "st": "data/xlnet-large-cased-physicaliqa-train-st.df"
            },
            "dev": {
                "ai2": "data/xlnet-large-cased-physicaliqa-dev-ai2.df",
                "st": "data/xlnet-large-cased-physicaliqa-dev-st.df"
            }
        },
        "socialiqa":{
            "train":{
                "ai2": "data/xlnet-large-cased-socialiqa-train-ai2.df",
                "st": "data/xlnet-large-cased-socialiqa-train-st.df"
            },
            "dev": {
                "ai2": "data/xlnet-large-cased-socialiqa-dev-ai2.df",
                "st": "data/xlnet-large-cased-socialiqa-dev-st.df"
            }
        }
    }
}

closest_indices = {
    "alphanli": {
        "roberta": {
            "ai2": "data/roberta-alphanli-ai2.rank",
            "st": "data/roberta-alphanli-st.rank",
        },
        "bert": {
            "ai2": "data/bert-alphanli-ai2.rank",
            "st": "data/bert-alphanli-st.rank",
        },
        "xlnet": {
            "ai2": "data/xlnet-alphanli-ai2.rank",
            "st": "data/xlnet-alphanli-st.rank",
        }
    },
    "hellaswag": {
        "roberta": {
            "ai2": "data/roberta-hellaswag-ai2.rank",
            "st": "data/roberta-hellaswag-st.rank",
        },
        "bert": {
            "ai2": "data/bert-hellaswag-ai2.rank",
            "st": "data/bert-hellaswag-st.rank",
        },
        "xlnet": {
            "ai2": "data/xlnet-hellaswag-ai2.rank",
            "st": "data/xlnet-hellaswag-st.rank",
        }
    },
    "physicaliqa": {
        "roberta": {
            "ai2": "data/roberta-physicaliqa-ai2.rank",
            "st": "data/roberta-physicaliqa-st.rank",
        },
        "bert": {
            "ai2": "data/bert-physicaliqa-ai2.rank",
            "st": "data/bert-physicaliqa-st.rank",
        },
        "xlnet": {
            "ai2": "data/xlnet-physicaliqa-ai2.rank",
            "st": "data/xlnet-physicaliqa-st.rank",
        }
    },
    "socialiqa": {
        "roberta": {
            "ai2": "data/roberta-socialiqa-ai2.rank",
            "st": "data/roberta-socialiqa-st.rank",
        },
        "bert": {
            "ai2": "data/bert-socialiqa-ai2.rank",
            "st": "data/bert-socialiqa-st.rank",
        },
        "xlnet": {
            "ai2": "data/xlnet-socialiqa-ai2.rank",
            "st": "data/xlnet-socialiqa-st.rank",
        }
    }
}
