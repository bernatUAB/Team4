import argparse
import os
from typing import Any, Callable, Dict, List
import cv2
from descriptor import ImageDescriptor
import distances as dist
from query_by_sample import load_database, load_queries, add_descriptors_to_dataset
import numpy as np
import pickle

def find_closest(query: Dict[str, Any], dataset: List[Dict[str, Any]], distance_func: Callable, k: int = 10):
    for entry in dataset:
        distance = distance_func(entry['descriptor'], query['descriptor'])
        entry['distance'] = distance

    similarity_functions = ['hist_intersection', 'hellinger_similarity']

    reverse_sort = distance_func.__name__ in similarity_functions
    
    return list(sorted(dataset, key=lambda e: e['distance'], reverse=reverse_sort))[:k]

def calculate_average_precision(ground_truth_indices: List[int], predicted_indices: List[int]) -> float:
    precision_at_k = []
    num_correct = 0
    for i, p in enumerate(predicted_indices):
        if p in ground_truth_indices:
            num_correct += 1
            precision_at_k.append(num_correct / (i + 1))

    if not precision_at_k:
        return 0.0

    return np.mean(precision_at_k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, default="./dataset/BBDD")
    parser.add_argument("queries_path", type=str, default="./dataset/qsd1_w1")
    parser.add_argument("-k", "--top_k", type=int, default=1)
    args = parser.parse_args()

    distance_functions_to_test = {
        "Euclidean Distance": dist.euclidean_distance,
        "L1 Distance": dist.l1_distance,
        "X^2 Distance": dist.x2_distance,
        "Histogram Intersection": dist.hist_intersection,
        "Hellinger Similarity": dist.hellinger_similarity,
        "KL Divergence": dist.kl_divergence,
        "Jensen Shanon Divergence": dist.jensen_shannon_divergence,
        "Quadratic Form Distance": dist.quadratic_form_distance,
    }

    descriptor_maker = ImageDescriptor(color_space='GRAY', bins_per_channel=128, normalize_histograms=True)

    print("Loading dataset...")
    database = load_database(args.dataset_path)
    queries = load_queries(args.queries_path)
    
    gt_path = os.path.join(args.queries_path, "gt_corresps.pkl")
    try:
        with open(gt_path, 'rb') as f:
            ground_truth = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_path}")
        return

    print("Computing descriptors for database...")
    add_descriptors_to_dataset(database)
    print("Computing descriptors for queries...")
    add_descriptors_to_dataset(queries)

    print("Evaluation")
    for dist_name, dist_func in distance_functions_to_test.items():
        print(f"Evaluating with: {dist_name}")
        
        average_precisions = []
        for i, query in enumerate(queries):
            closest_k = find_closest(query, database, dist_func, k=args.top_k)

            predicted_names = [entry['name'] for entry in closest_k]
            
            predicted_indices = [int(os.path.splitext(entry['name'])[0].split('_')[-1]) for entry in closest_k]
            
            gt_indices = ground_truth[i]
            if not isinstance(gt_indices, list):
                gt_indices = [gt_indices]
            is_correct = not set(predicted_indices).isdisjoint(set(gt_indices))
            status = "-> CORRECT" if is_correct else "-> INCORRECT"

            print(f"  Query: {query['name']} | GT: {gt_indices} | Predictions: {predicted_names} { status }")



            ap = calculate_average_precision(gt_indices, predicted_indices)
            average_precisions.append(ap)

        mean_ap = np.mean(average_precisions)
        print(f"Mean Average Precision (mAP) @{args.top_k}: {mean_ap:.4f}")



if __name__ == "__main__":
    main()