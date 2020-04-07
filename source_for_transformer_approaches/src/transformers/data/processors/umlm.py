import json
import logging
import os
from .utils import DataProcessor, InputFeatures, InputExample

logger = logging.getLogger(__name__)


class UMLMProcessor(DataProcessor):

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError()

    def get_labels(self):
        return self.vocab_list

    def get_train_examples(self, data_dir):
        with open(os.path.join(data_dir, "umlm_train.json"), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir):
        with open(os.path.join(data_dir, "umlm_dev.json"), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "dev")

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def umlm_convert_example_to_features(examples, tokenizer, max_length=512, pad_token=0, pad_token_segment_id=0):
    processor = UMLMProcessor(tokenizer.vocab)
    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))
        inputs = tokenizer.encode_plus(
            example.text_a, None, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        label = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )
    return features