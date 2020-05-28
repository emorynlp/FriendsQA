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


class UOPProcessor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError()

    def get_labels(self):
        return ["Yes", "No"]

    def get_train_examples(self, data_dir):
        with open(os.path.join(data_dir, "uop_train.json"), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir):
        with open(os.path.join(data_dir, "uop_dev.json"), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        examples = []
        for i in tqdm(range(len(input_data))):
            guid = "%s-%s" % (set_type, str(i))
            utterances = input_data[i]["utterances"]
            label = input_data[i]["is_correct_order"]
            examples.append(UOPExample(guid=guid, contents=utterances, label=label))
        return examples


def uop_convert_example_to_features(examples, tokenizer, max_line_length=128, max_line_number=107):
    processor = UOPProcessor()
    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 100 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))
        contents = example.contents
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
        label = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("lines_input_ids: %s" % " ".join([str(x) for x in lines_input_ids]))
            logger.info("attention_masks: %s" % " ".join([str(x) for x in attention_masks]))
            logger.info("number of lines: %s" % str(len(lines_input_ids)))
            logger.info("label: %s (id = %d)" % (example.label, label))
        if len(lines_input_ids) > max_line_number:
            lines_input_ids = lines_input_ids[:max_line_number]
            attention_masks = attention_masks[:max_line_number]
        features.append(UOPFeatures(lines_input_ids=lines_input_ids, attention_masks=attention_masks, label=label))
    return features


class UOPExample(object):
    def __init__(self, guid, contents, label=None):
        self.guid = guid
        self.contents = contents
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class UOPFeatures(object):
    def __init__(self, lines_input_ids, attention_masks=None, label=None):
        self.lines_input_ids = lines_input_ids
        self.attention_masks = attention_masks
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


