import json
from collections import Counter, defaultdict
import pandas as pd
import string
import re
import sys
import os


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    if ground_truth is None or prediction is None:
        return exact_match_score(prediction, ground_truth)
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    # print (normalize_answer(prediction))
    # print (normalize_answer(ground_truth))
    # print ('-'*40)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def precision_score(prediction, ground_truth):
    if ground_truth is None or prediction is None:
        return exact_match_score(prediction, ground_truth)
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    return precision


def recall_score(prediction, ground_truth):
    if ground_truth is None or prediction is None:
        return exact_match_score(prediction, ground_truth)
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return recall


def exact_match_score(prediction, ground_truth):
    if ground_truth is None or prediction is None:
        return prediction == ground_truth
    # print (normalize_answer(prediction),normalize_answer(ground_truth))
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(gold_answers, predictions, out_res_file):
    f1 = exact_match = precision = recall = total = 0.

    for ground_truths, prediction in zip(gold_answers, predictions):
        # print (ground_truths,prediction)
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        recall += metric_max_over_ground_truths(
            recall_score, prediction, ground_truths)
        precision += metric_max_over_ground_truths(
            precision_score, prediction, ground_truths)
        # print (exact_match,f1,recall,precision)
        # break

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    precision = 100.0 * precision / total
    recall = 100.0 * recall / total

    with open(out_res_file, "w") as f:
        f.write("EM: %.4f" % exact_match + "\n")
        f.write("F1: %.4f" % f1 + "\n")
        f.write("Precision: %.4f" % precision + "\n")
        f.write("Recall: %.4f" % recall + "\n")