from typing import Any
import matplotlib.pyplot as plt
import json

def run_clustering(args: Any) -> None:
    features: list = []
    labels: list = []
    with open(args.filename, 'r') as f:
        data = json.load(f)
        features = data['features']
        labels = data['labels']

    assert len(features) > 0, "Input feature list is empty"
    assert len(labels) > 0, "Label list is empty"
    assert len(features[0]) in [2, 3], "Please provide 2D or 3D features"

    if len(features[0]) == 2:
        plt.scatter([x[0] for x in features], [x[1] for x in features], c=labels)
        plt.xlabel(args.xlabel)
        plt.ylabel(args.ylabel)
        plt.title(args.title)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter([x[0] for x in features],
                   [x[1] for x in features],
                   [x[2] for x in features],
                   c=labels,)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        ax.set_zlabel(args.zlabel)
        ax.set_title(args.title)

    plt.show()


def run_scatterplot(args: Any) -> None:
    features: list = []
    labels: list = []

    with open(args.filename, 'r') as f:
        data = json.load(f)
        features = data['features']
        if args.labels:
            labels = data['labels']

    assert len(features) > 0, "Input feature list is empty"
    assert len(features[0]) in [2, 3], "You can only visualize 2D/3D features. Set n_components=2"

    if len(features[0]) == 2:
        if args.labels:
            plt.scatter([x[0] for x in features], [x[1] for x in features], c=labels)
        else:
            plt.scatter([x[0] for x in features], [x[1] for x in features])

        plt.xlabel(args.xlabel)
        plt.ylabel(args.ylabel)
        plt.title(args.title)

    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if args.labels:
            ax.scatter([x[0] for x in features],
                    [x[1] for x in features],
                    [x[2] for x in features],
                    c=labels,)
        else:
            ax.scatter([x[0] for x in features],
                    [x[1] for x in features],
                    [x[2] for x in features],)

        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        ax.set_zlabel(args.zlabel)
        ax.set_title(args.title)

    plt.show()

