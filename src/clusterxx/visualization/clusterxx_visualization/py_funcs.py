from typing import Any
import matplotlib.pyplot as plt
import json

def run_clustering(args: Any) -> None:
    features: list = []
    labels: list = []
    with open(args.filename, 'r') as f:
        data = json.load(f)
        print(type(data['features']))
        print(len(data['features']))
        features = data['features']
        labels = data['labels']

    assert len(features) > 0, "Input feature list is empty"
    assert len(labels) > 0, "Label list is empty"
    assert len(features[0]) in [2, 3], "Please provide 2D or 3D features"

    if len(features[0]) == 2:
        plt.scatter([x[0] for x in features], [x[1] for x in features], c=labels)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter([x[0] for x in features],
                   [x[1] for x in features],
                   [x[2] for x in features],
                   c=labels,)

    plt.show()


