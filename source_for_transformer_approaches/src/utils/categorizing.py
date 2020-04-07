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
import copy
import re

categorized = {}
utter_id_regex = re.compile('u[0-9][0-9][0-9]')
q_words = ['What', 'When', 'Who', 'Where', 'Why', 'How']


def find_utter(context, text):
    index = context.find(text)
    if index == -1:
        return ""
    index -= 4  # u001 format of utter id
    while index >= 0:
        if utter_id_regex.match(context[index:index + 4]):
            return context[index:index + 4]
        index -= 1
    return ""


def categorizing( data_file,prediction_file, categorized_file):
    with open(prediction_file) as pred, open(data_file) as dev:
        pred_json = json.load(pred)
        dev_json = json.load(dev)
        total = same_utter = 0
        # what_count = when_count = who_count = where_count = why_count = how_count = 0
        # what_count_same = when_count_same = who_count_same = where_count_same = why_count_same = how_count_same = 0
        count_dict = {}
        for q in q_words:
            categorized[q] = {}
            count_dict[q] = 0
            count_dict[q + "_same"] = 0

        for para in dev_json['data']:
            # print(len(para['paragraphs']))
            qas = para['paragraphs'][0]['qas']
            context = para['paragraphs'][0]['context']
            # print(context)
            for qa in qas:
                # #of total questions
                total += 1

                # actual prediction
                if qa['id'] not in pred_json:
                    continue
                prediction = pred_json[qa['id']]
                qa['prediction'] = prediction

                # predicted utter id
                pred_utter = find_utter(context, prediction)
                qa['predicted_utterance'] = pred_utter
                # print("prediction: ", pred_utter)

                # question word
                q_word = qa['id'].replace("_Paraphrased", "").split('_')[-1]

                count_dict[q_word] += 1
                same = False
                for a in qa['answers']:
                    one_utter = find_utter(context, a['text'])
                    # print("correct: ", one_utter)

                    if one_utter == pred_utter:
                        same = True
                        same_utter += 1
                        count_dict[q_word + "_same"] += 1
                        break
                # if not same:
                #     print("context: ", context)
                #     print("pred utter: ", pred_utter)
                #     print("pred: ", prediction)
                #     print("answer: ", qa['answers'])
                #     print("q: ", qa['question'])
                #     print(qa['id'])
                #     print('=' * 20)
                # print(qa['id'].replace("_Paraphrased", "").split('_')[-1])

                categorized[q_word][qa['id']] = qa
                # categorized[qa['id'].replace("_Paraphrased", "").split('_')[-1]].append(qa)
                # print(qa['id'])
                # print(qa)
        print(same_utter)
        print(total)
        print(count_dict)
        total_q = 0.0
        total_q_same = 0.0
        for key, value in count_dict.items():
            if key + "_same" in count_dict:
                print(key + ":", count_dict[key + "_same"] * 100.0 / count_dict[key])
                total_q += count_dict[key]
                total_q_same += count_dict[key + "_same"]
        print("overall: ", total_q_same * 100.0 / total_q)
        print(total_q)
        print(categorized)
        print(dev_json)
    with open(categorized_file, 'w') as out:
        json.dump(categorized, out, indent=2)

