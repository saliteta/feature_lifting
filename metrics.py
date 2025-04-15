import argparse
from evaluation.metric_class import Metrics

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', help="labels contain json and jpg", type=str, required=True)
    parser.add_argument('--feature_location', help="lifted feature location, feature.pt", type=str, required=True)
    parser.add_argument('--output', help="segmented result", default='./evaluation/outputs', type=str)
    args = parser.parse_args()
    return args






if __name__ == "__main__":
    args = parser()

    labels = args.labels
    feature_location = args.feature_location
    output = args.output

    metrics = Metrics(labels, feature_location)

    metrics.metrics(mode='Light', output_location=output, filtering_enable=True)