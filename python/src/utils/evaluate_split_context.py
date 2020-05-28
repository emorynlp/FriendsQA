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

from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import os
from tqdm import tqdm


def eval_best(data_file, result_file):
    with open(os.path.join(data_file), "r", encoding="utf-8") as reader:
        input_data = json.load(reader)
    with open(os.path.join(result_file), "r", encoding="utf-8") as reader:
        result_data = json.load(reader)
    data = input_data["data"]
    f1 = exact_match = total = 0
    for i in tqdm(range(len(data))):
        content_list = []
        utterances = data[i]["paragraphs"][0]["utterances:"]
        qas = data[i]["paragraphs"][0]["qas"]
        for ui, utterance in enumerate(utterances):
            speaker = utterance["speakers"][0].split(" ")
            if len(speaker) >= 2:
                speaker = speaker[0] + "_" + speaker[1]
            else:
                speaker = speaker[0]
            u_text = "u" + str(ui) + " " + speaker + " " + utterance["utterance"]
            content_list.append(u_text)
        for qa in qas:
            total += 1
            q_id = qa["id"]
            result = result_data[q_id]
            pred_uid = result["uid"]
            pred_left = result["inner_left"]
            pred_right = result["inner_right"]
            pred_utterance = None
            pred_answer_text = None
            answers = qa["answers"]
            answer_texts = []
            answer_uids = []
            for answer in answers:
                answer_texts.append(answer["answer_text"])
                answer_uids.append(answer["utterance_id"])
            if 0 <= pred_uid < len(content_list) and pred_uid in answer_uids:
                pred_utterance = content_list[pred_uid]
            if pred_utterance:
                pred_u_tokens = pred_utterance.split(" ")
                if pred_left <= pred_right and 0 <= pred_left < len(pred_u_tokens) and 0 <= pred_right < len(pred_u_tokens):
                    pred_answer_text = " ".join(pred_u_tokens[pred_left:pred_right + 1])
                else:
                    pred_answer_text = pred_utterance

            if pred_answer_text:
                f1 += metric_max_over_ground_truths(f1_score, pred_answer_text, answer_texts)
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, pred_answer_text, answer_texts)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def eval_n_best(data_file, result_file):
    with open(os.path.join(data_file), "r", encoding="utf-8") as reader:
        input_data = json.load(reader)
    with open(os.path.join(result_file), "r", encoding="utf-8") as reader:
        result_data = json.load(reader)
    data = input_data["data"]
    f1 = exact_match = total = 0
    for i in tqdm(range(len(data))):
        content_list = []
        utterances = data[i]["paragraphs"][0]["utterances:"]
        qas = data[i]["paragraphs"][0]["qas"]
        for ui, utterance in enumerate(utterances):
            speaker = utterance["speakers"][0].split(" ")
            if len(speaker) >= 2:
                speaker = speaker[0] + "_" + speaker[1]
            else:
                speaker = speaker[0]
            u_text = "u" + str(ui) + " " + speaker + " " + utterance["utterance"]
            content_list.append(u_text)
        for qa in qas:
            total += 1
            q_id = qa["id"]
            results = result_data[q_id]
            f1s = []
            exact_matches = []
            for result in results:
                pred_uid = result["uid"]
                pred_left = result["inner_left"]
                pred_right = result["inner_right"]
                pred_utterance = None
                pred_answer_text = None
                answers = qa["answers"]
                answer_texts = []
                answer_uids = []
                for answer in answers:
                    answer_texts.append(answer["answer_text"])
                    answer_uids.append(answer["utterance_id"])
                if 0 <= pred_uid < len(content_list) and pred_uid in answer_uids:
                    pred_utterance = content_list[pred_uid]
                if pred_utterance:
                    pred_u_tokens = pred_utterance.split(" ")
                    if pred_left <= pred_right and 0 <= pred_left < len(pred_u_tokens) and 0 <= pred_right < len(
                            pred_u_tokens):
                        pred_answer_text = " ".join(pred_u_tokens[pred_left:pred_right + 1])
                    else:
                        pred_answer_text = pred_utterance
                if pred_answer_text:
                    f1s.append(metric_max_over_ground_truths(f1_score, pred_answer_text, answer_texts))
                    exact_matches.append(metric_max_over_ground_truths(
                        exact_match_score, pred_answer_text, answer_texts))
            if len(f1s) > 0:
                f1 += max(f1s)
                exact_match += max(exact_matches)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude or ch == '_')

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
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for FriendsQA split context ')
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    parser.add_argument("--do_best_eval", action="store_true", help="do n best evaluate")
    args = parser.parse_args()
    if args.do_best_eval:
        print(json.dumps(eval_best(args.dataset_file, args.prediction_file)))
    else:
        print(json.dumps(eval_n_best(args.dataset_file, args.prediction_file)))