from bottle import route, run, template, static_file, request
import bottle
import os
import json
import math
from collections import defaultdict
from nn import *
from config import *
from loguru import logger

bottle.TEMPLATE_PATH.insert(0, 'views')

# pylint: disable=no-member

app = application = bottle.Bottle()

@app.route('/<filename:path>')
def send_static(filename):
    # print(filename, static_file(filename, root='static/'))
    return static_file(filename, root='static/')


@app.route('/', method='GET')
def index():
    return template(
        'index.html',
        tasks=['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa'],
        models=['roberta', 'bert', 'xlnet'],
        embedders=['ai2', 'st'],
        filters={},
        task="alphanli",
        embedder="ai2",
        result={},
        total=0,
        margins=[],
        all="all"
    )


@app.route('/', method='POST')
def retrieve():
    logger.info(f"Request: {request.forms.__dict__}")
    task = request.forms.get('task')
    embedder = request.forms.get('embedder')

    filters = {}

    for model in ['roberta', 'bert', 'xlnet']:
        if request.forms.get(model, None) is not None:
            filters[model] = request.forms.get(model, None)
    logger.info(f"Filters: {filters}")

    order, flattened = get_order(filters or {m: None for m in ['roberta', 'bert', 'xlnet']}, task)

    train_dataset, dev_dataset = load_dataset(task, order=order)

    if filters == {}:
        filters = {m: None for m in ['roberta', 'bert', 'xlnet']}
        valid_indices = list(range(len(dev_dataset)))
    else:
        valid_indices = filtering(filters, task, order=order)

    margins = heatmap(filters, task, order)

    result = {}
    closest = get_closest(filters, task, embedder, order=flattened)

    for i, model in enumerate(filters):
        preds, probs, labels = load_predictions(predictions[model][task], order=order)
        for j, (pred, prob, label) in enumerate(zip(preds, probs, labels)):
            # print(j)
            if j not in valid_indices: continue
            if j not in result:
                result[j] = dev_dataset[j]
            # print(closest[i][j * datasets[task]["num_choices"] + pred - datasets[task]["offset"]])
            # print(j * datasets[task]["num_choices"] + pred - datasets[task]["offset"])
            result[j]["choices"][pred - datasets[task]["offset"]]["models"].append({
                "model": model,
                "margin": "-" if pred == label else prob[pred - datasets[task]["offset"]] - prob[label - datasets[task]["offset"]],
                "closest": [
                    {
                        "ctx": train_dataset[math.floor(int(x) / datasets[task]["num_choices"])]["ctx"],
                        "choice": train_dataset[math.floor(int(x) / datasets[task]["num_choices"])]["choices"][int(x) % datasets[task]["num_choices"]]
                    } for x in closest[i][j * datasets[task]["num_choices"] + pred - datasets[task]["offset"]]
                ]
            })


    return template(
        'index.html',
        tasks=['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa'],
        models=['roberta', 'bert', 'xlnet'],
        embedders=['ai2', 'st'],
        filters=filters,
        task=task,
        embedder=embedder,
        result=result,
        total=len(dev_dataset),
        margins=margins,
        all=request.forms.get("all")
    )

def filtering(filters, task, order=None):

    valid_indices = set()

    for j, model in enumerate(filters):
        model_indices = set()
        preds, probs, labels = load_predictions(predictions[model][task], order=order)

        for i, (pred, prob, label) in enumerate(zip(preds, probs, labels)):

            if pred == label and filters[model] == "correct":
                model_indices.add(i)
            elif pred != label and filters[model] == "wrong":
                model_indices.add(i)
        valid_indices = model_indices if j == 0 else valid_indices.intersection(model_indices)
    return valid_indices

def heatmap(filters, task, order=None, flatten=True):

    margins = []
    for i, model in enumerate(filters):
        preds, probablities, labels = load_predictions(predictions[model][task], order=order)
        margin = []
        for j, (pred, prob, label) in enumerate(zip(preds, probablities, labels)):
            margin.append([j, i, "-"] if pred == label else [j, i, prob[pred - datasets[task]["offset"]] - prob[label - datasets[task]["offset"]]])
        logger.info(f"Model {model} accuracy: {sum( 1 if x[-1] == '-' else 0 for x in margin) / len(margin)}")
        if flatten:
            margins.extend(margin)
        else:
            margins.append(margin)
    return margins

def get_closest(filters, task, embedder, order=None):

    results = []

    for i, model in enumerate(filters):
        if order is None:
            results.append(np.loadtxt(closest_indices[task][model][embedder]))
        else:
            rank = np.loadtxt(closest_indices[task][model][embedder])
            results.append(rank[order, :])

    return results

def get_order(filters, task):

    margins = heatmap(filters, task, None, flatten=False)
    order = [i for i in range(len(margins[0]))]
    order = sorted(order, key=lambda j: sum(1 if c[j][-1] == '-' else 0 for c in margins))

    flatten_order = []

    for x in order:

        flatten_order.extend(list(range(x * datasets[task]["num_choices"], (x+1) * datasets[task]["num_choices"])))

    return order, flatten_order

if __name__ == "__main__":
    run(app=app, host='localhost', port=7778, reloader=True)
