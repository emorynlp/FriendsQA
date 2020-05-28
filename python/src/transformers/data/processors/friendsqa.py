import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from collections import OrderedDict
from collections import Counter
import operator

import numpy as np
from tqdm import tqdm

from ...file_utils import is_tf_available, is_torch_available
from ...tokenization_utils import PreTrainedTokenizer
from .utils import DataProcessor
import copy

logger = logging.getLogger(__name__)


class FriendsQAProcessor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        with open(os.path.join(data_dir, "friendsqa_trn.json"), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir):
        with open(os.path.join(data_dir, "friendsqa_dev.json"), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "dev")

    def get_test_examples(self, data_dir):
        with open(os.path.join(data_dir, "friendsqa_tst.json"), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "test")

    def _create_examples(self, input_data, set_type):
        examples = []
        data = input_data["data"]
        for i in tqdm(range(len(data))):
            content_list = []
            utterances = data[i]["paragraphs"][0]["utterances:"]
            qas = data[i]["paragraphs"][0]["qas"]
            n_length = len(utterances)
            for ui, utterance in enumerate(utterances):
                speaker = utterance["speakers"][0].split(" ")
                if len(speaker) >= 2:
                    speaker = speaker[0] + "_" + speaker[1]
                else:
                    speaker = speaker[0]
                u_text = "u" + str(ui) + " " + speaker + " " + utterance["utterance"]
                content_list.append(u_text)
            for qa in qas:
                q_id = qa["id"]
                question = qa["question"]
                answers = qa["answers"]
                if set_type == "train":
                    for a_id, answer in enumerate(answers):
                        guid = "%s-%s" % (set_type, str(q_id))
                        answer_text = answer["answer_text"]
                        utterance_id = answer["utterance_id"]
                        is_speaker = answer["is_speaker"]
                        if is_speaker:
                            inner_start = 1
                            inner_end = 1
                        else:
                            inner_start = answer["inner_start"] + 2
                            inner_end = answer["inner_end"] + 2
                        left_labels = n_length * [-1]
                        right_labels = n_length * [-1]
                        left_labels[utterance_id] = inner_start
                        right_labels[utterance_id] = inner_end
                        examples.append(FriendsQAExample(guid=guid, contents=content_list,
                                                         question=question, utterance_label=utterance_id,
                                                         left_labels=left_labels, right_labels=right_labels,
                                                         answer_text=answer_text))
                else:
                    guid = "%s-%s" % (set_type, str(q_id))
                    examples.append(FriendsQAExample(guid=guid, contents=content_list, question=question))
        return examples


def friendsqa_convert_example_to_features(examples, tokenizer, max_line_length=64,
                                                  max_line_number=107, max_question_length=128):
    pad_token_ids = []
    for i in range(max_line_length):
        pad_token_ids.append(tokenizer.pad_token_id)
    assert len(pad_token_ids) == max_line_length

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 1000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))
        guid = example.guid
        contents = example.contents
        utterance_label = example.utterance_label
        left_labels = example.left_label
        right_labels = example.right_label
        question = example.question
        question_inputs = tokenizer.encode_plus(question, None, add_special_tokens=True,
                                                max_length=max_question_length, )
        question_input_ids = question_inputs["input_ids"]
        attention_mask = [1] * len(question_input_ids)
        padding_length = max_question_length - len(question_input_ids)
        question_input_ids = question_input_ids + ([tokenizer.pad_token_id] * padding_length)
        question_attention_masks = attention_mask + ([0] * padding_length)
        lines_input_ids = []
        attention_masks = []
        for content in contents:
            tokens = tokenizer.tokenize(content)
            tokens = tokens[:max_line_length - 2]
            inputs = tokenizer.encode_plus(" ".join(tokens), None, add_special_tokens=True,
                                           max_length=max_line_length, )
            input_ids = inputs["input_ids"]
            attention_mask = [1] * len(input_ids)
            padding_length = max_line_length - len(input_ids)
            input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            attention_masks.append(attention_mask)
            lines_input_ids.append(input_ids)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("utterances_input_ids: %s" % " ".join([str(x) for x in lines_input_ids]))
            logger.info("attention_masks: %s" % " ".join([str(x) for x in attention_masks]))
            logger.info("number of utterances: %s" % str(len(lines_input_ids)))
            logger.info("question_input_ids: %s" % " ".join([str(x) for x in question_input_ids]))
            logger.info("question_attention_masks: %s" % " ".join([str(x) for x in question_attention_masks]))
        if len(lines_input_ids) > max_line_number:
            lines_input_ids = lines_input_ids[:max_line_number]
            attention_masks = attention_masks[:max_line_number]
            right_labels = right_labels[:max_line_number]
            left_labels = left_labels[:max_line_number]
        features.append(FriendsQAFeatures(guid=guid, lines_input_ids=lines_input_ids,
                                          attention_masks=attention_masks,
                                          question_input_ids=question_input_ids,
                                          question_attention_masks=question_attention_masks,
                                          utterance_label=utterance_label, right_labels=right_labels,
                                          left_labels=left_labels))
    return features


class FriendsQAExample(object):
    def __init__(self, guid, contents, question, utterance_label=None,
                 left_labels=None, right_labels=None, answer_text=None):
        self.guid = guid
        self.contents = contents
        self.question = question
        self.utterance_label = utterance_label
        self.left_label = left_labels
        self.right_label = right_labels
        self.answer_text = answer_text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class FriendsQAFeatures(object):
    def __init__(self, guid, lines_input_ids, question_input_ids, attention_masks=None,
                 question_attention_masks=None, utterance_label=None, left_labels=None, right_labels=None):
        self.guid = guid
        self.lines_input_ids = lines_input_ids
        self.attention_masks = attention_masks
        self.question_input_ids = question_input_ids
        self.question_attention_masks = question_attention_masks
        self.utterance_label = utterance_label
        self.left_label = left_labels
        self.right_label = right_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


