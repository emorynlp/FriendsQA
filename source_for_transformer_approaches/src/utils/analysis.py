# coding=utf-8
# Copyright 2020 Changmao Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import string
import re
import sys
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def remove_underline(text):
        return text.replace('_', ' ')

    return remove_underline(white_space_fix(remove_articles(remove_punc(lower(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# evaluate f1 under 1 question word. For example, calculate F1 for all what questions
def evaluate(dataset):
    f1 = exact_match = total = 0
    # for article in dataset:
    #     for paragraph in article['paragraphs']:
    #         for qa in paragraph['qas']:
    #             total += 1
    #             if qa['id'] not in predictions:
    #                 message = 'Unanswered question ' + qa['id'] + \
    #                           ' will receive score 0.'
    #                 print(message, file=sys.stderr)
    #                 continue
    #             ground_truths = list(map(lambda x: x['text'], qa['answers']))
    #             prediction = predictions[qa['id']]
    #             exact_match += metric_max_over_ground_truths(
    #                 exact_match_score, prediction, ground_truths)
    #             f1 += metric_max_over_ground_truths(
    #                 f1_score, prediction, ground_truths)

    for id, qa in dataset.items():
        total += 1
        ground_truths = list(map(lambda x: x['text'], qa['answers']))
        prediction = qa['prediction']
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def print_categorize(categorized_file):
    with open(categorized_file) as json_in:
        data = json.load(json_in)
        # print(data)
        for key, value in data.items():
            print(key, evaluate(value))



