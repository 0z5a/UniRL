import numpy as np
import re
from typing import Literal

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    This handles both English and Chinese characters correctly.
    """
    # Create a matrix of size (len(s1)+1) x (len(s2)+1)
    dp = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)

    # Initialize the first row and column
    for i in range(len(s1) + 1):
        dp[i, 0] = i
    for j in range(len(s2) + 1):
        dp[0, j] = j

    # Fill the matrix
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = min(
                    dp[i-1, j] + 1,    # deletion
                    dp[i, j-1] + 1,    # insertion
                    dp[i-1, j-1] + 1   # substitution
                )

    return dp[len(s1), len(s2)]

def calculate_correspondence(list_a, list_b):
    """
    Calculate the correspondence between two lists of strings based on edit distance.

    Args:
        list_a: List of N strings
        list_b: List of M strings

    Returns:
        A list of tuples (i, j, distance) where i is the index in list_a,
        j is the index in list_b, and distance is the edit distance.
        The list is sorted by distance (ascending).
    """
    # Calculate edit distance for all pairs
    distances = []
    for i, str_a in enumerate(list_a):
        for j, str_b in enumerate(list_b):
            dist = levenshtein_distance(str_a, str_b)
            distances.append((i, j, dist))

    # Sort by distance
    distances.sort(key=lambda x: x[2])

    return distances

def find_best_matches(list_a, list_b):
    """
    Find the best matching pairs between two lists of strings.
    Each string in list_a is matched with the closest string in list_b,
    and vice versa.

    Args:
        list_a: List of N strings
        list_b: List of M strings

    Returns:
        A list of tuples (i, j, distance) representing matched pairs.
    """
    all_distances = calculate_correspondence(list_a, list_b)

    # Track which indices have been matched
    matched_a = set()
    matched_b = set()
    matches = []

    # Assign matches based on minimal distance
    for i, j, dist in all_distances:
        if i not in matched_a and j not in matched_b:
            matches.append((i, j, dist))
            matched_a.add(i)
            matched_b.add(j)

            # Stop if all elements from either list are matched
            if len(matched_a) == len(list_a) or len(matched_b) == len(list_b):
                break

    return matches

def print_matches(list_a, list_b, matches):
    """Print the matching pairs in a readable format."""
    print(f"Found {len(matches)} matches:")
    for i, j, dist in matches:
        print(f"A[{i}]: '{list_a[i]}' ↔ B[{j}]: '{list_b[j]}' (distance: {dist})")

    # Print unmatched items from list_a
    unmatched_a = [i for i in range(len(list_a)) if i not in {m[0] for m in matches}]
    if unmatched_a:
        print("\nUnmatched items from list A:")
        for i in unmatched_a:
            print(f"A[{i}]: '{list_a[i]}'")

    # Print unmatched items from list_b
    unmatched_b = [j for j in range(len(list_b)) if j not in {m[1] for m in matches}]
    if unmatched_b:
        print("\nUnmatched items from list B:")
        for j in unmatched_b:
            print(f"B[{j}]: '{list_b[j]}'")


def get_matches_info(list_a, list_b):
    matches = find_best_matches(list_a, list_b)
    # for i, j, dist in matches:
        # print(f"A[{i}]: '{list_a[i]}' ↔ B[{j}]: '{list_b[j]}' (distance: {dist})")

    matches_dict ={}
    for i, j, dist in matches:
        matches_dict[j] = (i, list_a[i], dist)

    unmatched_a = [i for i in range(len(list_a)) if i not in {m[0] for m in matches}]
    unmatched_b = [j for j in range(len(list_b)) if j not in {m[1] for m in matches}]

    resorted_list_a = [matches_dict[j][1] for j in range(len(list_b)) if j in matches_dict] + [list_a[i] for i in unmatched_a]

    matches_rlt = {
        'matches': matches,
        'resorted_list_a': resorted_list_a,
        'unmatched_a': unmatched_a,
        'unmatched_b': unmatched_b,
    }

    return matches_rlt

def calculate_metrics(list_a, list_b, matches):
    """
    Calculate precision, recall, and F1 score.
    list_a: ground truth
    list_b: predictions
    matches: matching pairs between list_a and list_b
    """
    # Count of matched items
    matched_count = len(matches)
    # Adjust matched_count based on edit distance
    adjusted_matched_count = 0
    for i, j, dist in matches:
        if dist == 0:
            # Perfect match
            adjusted_matched_count += 1
        else:
            # Normalize the contribution based on edit distance
            # The closer to 0 the distance is, the closer to 1 the contribution
            max_len = max(len(list_a[i]), len(list_b[j]))
            if max_len > 0:
                # Normalize distance relative to the length of the longer string
                normalized_match = 1 - (dist / max_len)
                adjusted_matched_count += normalized_match

    # Use the adjusted count instead of the raw count
    matched_count = adjusted_matched_count

    # Total items in ground truth and predictions
    gt_count = len(list_a)
    pred_count = len(list_b)

    # Calculate metrics
    if gt_count == 0 and pred_count == 0:
        precision = 1.0
        recall = 1.0
        f1_score = 1.0
    else:
        precision = matched_count / pred_count if pred_count > 0 else 0
        recall = matched_count / gt_count if gt_count > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'matched_count': matched_count,
        'gt_count': gt_count,
        'pred_count': pred_count
    }


# def test_ocr_rw_rule():
#     # Example usage
#     list_a = ["苹果", "香蕉", "apple pie", "橙子", "火龙果", "apple", "beer"]
#     list_b = ["apple", "橙", "香蕉树", "火龙果", "pear"]

#     matches = find_best_matches(list_a, list_b)
#     # print_matches(list_a, list_b, matches)

#     matches_rlt = get_matches_info(list_a, list_b)
#     # print_matches(list_a, list_b, matches_rlt['matches'])

#     metrics = calculate_metrics(list_a, list_b, matches_rlt['matches'])
#     print(matches_rlt)
#     print(f"Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1 Score: {metrics['f1_score']}")

#     # Calculate reward
#     reward = metrics_to_reward(metrics)
#     print(f"RL Reward: {reward}")

#     # Example with custom weights
#     custom_reward = metrics_to_reward(metrics, alpha=0.8, beta=0.1, gamma=0.1)
#     print(f"Custom weighted RL Reward: {custom_reward}")


# if __name__ == "__main__":
#     test_ocr_rw_rule()
