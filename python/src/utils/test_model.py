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

import torch
import numpy as np
from src.transformers import BertConfig, BertForUtteranceLanguageModeling, BertForUtteranceOrderPrediction, \
    BertForDialogueSpanQuestionAnswering, RobertaForUtteranceOrderPrediction, RobertaForUtteranceLanguageModeling, \
    RobertaForDialogueSpanQuestionAnswering, RobertaConfig
from torchsummaryX import summary


def test_BertForUtteranceLanguageModeling():
    num_samples = 5
    seq_len = 64
    config = BertConfig(max_position_embeddings=512)
    model = BertForUtteranceLanguageModeling(config)
    input_ids = np.ones(shape=(num_samples, seq_len), dtype=np.int32)
    attention_masks = np.ones(shape=(num_samples, seq_len), dtype=np.int32)
    label_ids = np.full(shape=(num_samples,), dtype=np.int32, fill_value=0)
    model.cpu()
    model.float()
    summary(model, torch.tensor(input_ids.astype(np.int64)),
            torch.tensor(attention_masks.astype(np.int64)),
            torch.tensor(label_ids.astype(np.int64))
            )

def test_BertForUtteranceOrderPrediction():
    num_samples = 5
    num_utterances = 6
    seq_len = 64
    max_utterances = 10
    config = BertConfig(max_position_embeddings=512)
    utterance_config = BertConfig(max_position_embeddings=max_utterances+1, num_hidden_layers=2)
    model = BertForUtteranceOrderPrediction(config, utterance_config, max_utterances)
    utterances_input_ids = np.ones(shape=(num_samples, num_utterances, seq_len), dtype=np.int32)
    attention_masks = np.ones(shape=(num_samples, num_utterances, seq_len), dtype=np.int32)
    label_ids = np.full(shape=(num_samples, ), dtype=np.int32, fill_value=0)
    model.cpu()
    model.float()
    summary(model, torch.tensor(utterances_input_ids.astype(np.int64)),
            torch.tensor(attention_masks.astype(np.int64)),
            torch.tensor(label_ids.astype(np.int64))
            )

def test_BertForDialogueSpanQuestionAnswering():
    num_samples = 5
    num_utterances = 6
    seq_len = 64
    max_utterances = 10
    config = BertConfig(max_position_embeddings=512)
    utterance_config = BertConfig(max_position_embeddings=max_utterances+1, num_hidden_layers=2)
    model = BertForDialogueSpanQuestionAnswering(config, utterance_config, max_utterances, seq_len)
    utterances_input_ids = np.ones(shape=(num_samples, num_utterances, seq_len), dtype=np.int32)
    attention_masks = np.ones(shape=(num_samples, num_utterances, seq_len), dtype=np.int32)
    question_input_ids = np.ones(shape=(num_samples, 512), dtype=np.int32)
    question_attention_masks = np.ones(shape=(num_samples, 512), dtype=np.int32)
    left_ids = np.full(shape=(num_samples, num_utterances), dtype=np.int32, fill_value=1)
    right_ids = np.full(shape=(num_samples, num_utterances), dtype=np.int32, fill_value=1)
    label_ids = np.full(shape=(num_samples,), dtype=np.int32, fill_value=1)
    model.cpu()
    model.float()
    summary(model,
            torch.tensor(question_input_ids.astype(np.int64)),
            torch.tensor(utterances_input_ids.astype(np.int64)),
            torch.tensor(question_attention_masks.astype(np.int64)),
            torch.tensor(attention_masks.astype(np.int64)),
            torch.tensor(label_ids.astype(np.int64)),
            torch.tensor(left_ids.astype(np.int64)),
            torch.tensor(right_ids.astype(np.int64)),
            )

def test_RobertaForUtteranceLanguageModeling():
    num_samples = 5
    seq_len = 64
    config = RobertaConfig(max_position_embeddings=512)
    model = RobertaForUtteranceLanguageModeling(config)
    input_ids = np.ones(shape=(num_samples, seq_len), dtype=np.int32)
    attention_masks = np.ones(shape=(num_samples, seq_len), dtype=np.int32)
    label_ids = np.full(shape=(num_samples,), dtype=np.int32, fill_value=0)
    model.cpu()
    model.float()
    summary(model, torch.tensor(input_ids.astype(np.int64)),
            torch.tensor(attention_masks.astype(np.int64)),
            torch.tensor(label_ids.astype(np.int64))
            )


def test_RobertaForUtteranceOrderPrediction():
    num_samples = 5
    num_utterances = 6
    seq_len = 64
    max_utterances = 10
    config = RobertaConfig(max_position_embeddings=512)
    utterance_config = BertConfig(max_position_embeddings=max_utterances+1, num_hidden_layers=2)
    model = RobertaForUtteranceOrderPrediction(config, utterance_config, max_utterances)
    utterances_input_ids = np.ones(shape=(num_samples, num_utterances, seq_len), dtype=np.int32)
    attention_masks = np.ones(shape=(num_samples, num_utterances, seq_len), dtype=np.int32)
    label_ids = np.full(shape=(num_samples, ), dtype=np.int32, fill_value=0)
    model.cpu()
    model.float()
    summary(model, torch.tensor(utterances_input_ids.astype(np.int64)),
            torch.tensor(attention_masks.astype(np.int64)),
            torch.tensor(label_ids.astype(np.int64))
            )

def test_RobertaForDialogueSpanQuestionAnswering():
    num_samples = 5
    num_utterances = 6
    seq_len = 64
    max_utterances = 10
    config = RobertaConfig(max_position_embeddings=512)
    utterance_config = BertConfig(max_position_embeddings=max_utterances+1, num_hidden_layers=2)
    model = RobertaForDialogueSpanQuestionAnswering(config, utterance_config, max_utterances, seq_len)
    utterances_input_ids = np.ones(shape=(num_samples, num_utterances, seq_len), dtype=np.int32)
    attention_masks = np.ones(shape=(num_samples, num_utterances, seq_len), dtype=np.int32)
    question_input_ids = np.ones(shape=(num_samples, 512), dtype=np.int32)
    question_attention_masks = np.ones(shape=(num_samples, 512), dtype=np.int32)
    left_ids = np.full(shape=(num_samples, num_utterances), dtype=np.int32, fill_value=1)
    right_ids = np.full(shape=(num_samples, num_utterances), dtype=np.int32, fill_value=1)
    label_ids = np.full(shape=(num_samples,), dtype=np.int32, fill_value=1)
    model.cpu()
    model.float()
    summary(model,
            torch.tensor(question_input_ids.astype(np.int64)),
            torch.tensor(utterances_input_ids.astype(np.int64)),
            torch.tensor(question_attention_masks.astype(np.int64)),
            torch.tensor(attention_masks.astype(np.int64)),
            torch.tensor(label_ids.astype(np.int64)),
            torch.tensor(left_ids.astype(np.int64)),
            torch.tensor(right_ids.astype(np.int64)),
            )


if __name__ == "__main__":
    test_BertForUtteranceLanguageModeling()
    test_BertForUtteranceOrderPrediction()
    test_BertForDialogueSpanQuestionAnswering()
    test_RobertaForUtteranceOrderPrediction()
    test_RobertaForUtteranceLanguageModeling()
    test_RobertaForDialogueSpanQuestionAnswering()