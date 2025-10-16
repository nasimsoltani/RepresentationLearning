import os
import json
import argparse
import re
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np


def extract_distance_from_filename(filename):
    match = re.search(r'_(\d+)ft_', filename)
    if match:
        return int(match.group(1))
    return None


def normalized_mse(y_true, y_pred):
    var = np.var(y_true)
    if var == 0:
        return 0.0
    return mean_squared_error(y_true, y_pred) / var


def normalized_mae(y_true, y_pred):
    mean_abs = np.mean(np.abs(y_true))
    if mean_abs == 0:
        return 0.0
    return mean_absolute_error(y_true, y_pred) / mean_abs


def calculate_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    if not data:
        return {}

    if not isinstance(data, list):
        data = [data]

    distances = {}

    # CFO regression case
    if 'cfo_predictions' in file_path:
        for entry in data:
            distance = extract_distance_from_filename(entry['filename'])
            if distance is not None:
                if distance not in distances:
                    distances[distance] = {'preds': [], 'gts': []}
                distances[distance]['preds'].append(entry['y_pred_real'])
                distances[distance]['gts'].append(entry['y_true_real'])

        results = {}
        for dist, values in distances.items():
            if len(values['gts']) > 1:
                y_true = np.array(values['gts'])
                y_pred = np.array(values['preds'])
                results[dist] = {
                    'mse': mean_squared_error(y_true, y_pred),
                    'nmse': normalized_mse(y_true, y_pred),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'nmae': normalized_mae(y_true, y_pred)
                }
        return results

    # Channel regression case
    elif 'channel_predictions' in file_path:
        for entry in data:
            distance = extract_distance_from_filename(entry['filename'])
            if distance is not None:
                if distance not in distances:
                    distances[distance] = {'preds': [], 'gts': []}

                gts = np.array(entry['y_true']).flatten().tolist()
                preds = np.array(entry['y_pred']).flatten().tolist()
                distances[distance]['gts'].extend(gts)
                distances[distance]['preds'].extend(preds)

        results = {}
        for dist, values in distances.items():
            if len(values['gts']) > 1:
                y_true = np.array(values['gts'])
                y_pred = np.array(values['preds'])
                results[dist] = {
                    'mse': mean_squared_error(y_true, y_pred),
                    'nmse': normalized_mse(y_true, y_pred),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'nmae': normalized_mae(y_true, y_pred)
                }
        return results

    # RF fingerprinting classification case
    elif 'predictions.json' in file_path:
        for entry in data:
            distance = extract_distance_from_filename(entry['filename'])
            if distance is not None:
                if distance not in distances:
                    distances[distance] = {'preds': [], 'gts': []}
                distances[distance]['preds'].append(entry['pred_class'])
                distances[distance]['gts'].append(entry['gt_class'])

        results = {}
        for dist, values in distances.items():
            results[dist] = {'accuracy': accuracy_score(values['gts'], values['preds'])}
        return results

    return {}


def main(root_folder):
    output_data = {}

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename in ['cfo_predictions.json', 'channel_predictions.json', 'predictions.json']:
                file_path = os.path.join(dirpath, filename)
                exp_name = os.path.basename(os.path.dirname(file_path))

                metrics = calculate_metrics(file_path)

                if not metrics:
                    continue

                if exp_name not in output_data:
                    output_data[exp_name] = {}

                if 'cfo_predictions.json' in filename:
                    key = 'cfo'
                elif 'channel_predictions.json' in filename:
                    key = 'channel'
                elif 'predictions.json' in filename:
                    key = 'rf'
                else:
                    continue

                sorted_distances = sorted(metrics.keys())

                # Initialize if not exists
                if key not in output_data[exp_name]:
                    output_data[exp_name][key] = {'distance': sorted_distances}

                # Append all metrics
                metric_names = metrics[sorted_distances[0]].keys()
                for mname in metric_names:
                    output_data[exp_name][key][mname] = [metrics[d][mname] for d in sorted_distances]

    output_filepath = os.path.join(os.path.dirname(root_folder), 'distance_results_mse.json')

    with open(output_filepath, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Results saved to {output_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate experiment results by distance.')
    parser.add_argument('root_folder', type=str, help='The high-level folder to process.')
    args = parser.parse_args()
    main(args.root_folder)
