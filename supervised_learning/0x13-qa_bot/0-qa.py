#!/usr/bin/env python3
"""Use pretrained model to answer a question given a reference document"""


import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Use pretrained model to answer a question given a reference document"""
    tok = BertTokenizer.from_pretrained
    tokenizer = tok('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    qs = tokenizer.tokenize(question)
    ref = tokenizer.tokenize(reference)
    tokens = ['[CLS]'] + qs + ['[SEP]'] + ref + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(qs) + 1) + [1] * (len(ref) + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
                             (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer
