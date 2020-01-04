from  dataprocess  import coqa_lib
from dataprocess import coqa_data_augmentation
from dataprocess.tokenization import FullTokenizer,BasicTokenizer
import tensorflow as tf
import collections
import requests
import json
import os
from absl import flags
from absl import logging
import six
import math

with tf.io.gfile.GFile(os.path.join(os.path.dirname(__file__),'coqa_settings.json'), 'rb') as reader:
    coqa_settings = json.loads(reader.read().decode('utf-8'))


def make_coqa_example(question,current_turn,story ,history):
    is_training=False

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def create_marker_string(maker, len):
        m = []
        for i in range(len):
            m.append(maker)
        return ''.join(m)

    examples = []

    paragraph_text =story['text']
    story_id = story['id']
    # print(story_id)
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True

    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    history_q = []
    history_a = []
    history_q_marker = []
    history_a_marker = []
    turn_id=1
    current_question_text = question
    qa_history_text=""
    qa_history_marker_text=""

    for i in range(len(history)):
        turn_id = history[i]['turn']
        question_text = question
        # print('turn id:',turn_id)
        answer =  history[i]['answer']

        start_position = None
        end_position = None
        is_yes = 0
        is_no = 0
        is_unknown = 0

        rationale_text = ""

        qa_history_text = ""
        qa_history_marker_text = ""


        history_q.append(question_text.strip())
        history_q_marker.append(create_marker_string('Q', len(question_text.strip().split())))

        history_a.append((answer + ".").strip())
        history_a_marker.append(create_marker_string('A', len((answer + ".").strip().split())))

        for j in range(i):  # not include the current question and answers in the training example
            qa_history_text = qa_history_text + ' ' + history_q[j] + ' ' + history_a[j]
            qa_history_marker_text = qa_history_marker_text + ' ' + history_q_marker[j] + history_a_marker[j]

    if not is_training:
        start_position = -1
        end_position = -1
        rationale_text = ""
        gold_answer_text = ""
        is_yes = None
        is_no = None
        is_unknown = None

    example = coqa_lib.CoqaExample(
        story_id=story_id,
        turn_id=current_turn,
        question_text=current_question_text,
        doc_tokens=doc_tokens,
        gold_answer_text=gold_answer_text,
        orig_answer_text=rationale_text,
        start_position=start_position,
        end_position=end_position,
        is_yes=is_yes,
        is_no=is_no,
        is_unknown=is_unknown,
        qa_history_text=qa_history_text,
        qa_history_marker_text=qa_history_marker_text
    )
    examples.append(example)

    return examples
def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
        feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))
        return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["decode_ids"] = create_int_feature(feature.decode_ids)
    features["decode_mask"] = create_int_feature(feature.decode_mask)

    if self.is_training:
        features["start_positions"] = create_int_feature([feature.start_position])
        features["end_positions"] = create_int_feature([feature.end_position])
        features["is_yes"] = create_int_feature([feature.is_yes])
        features["is_no"] = create_int_feature([feature.is_no])
        features["is_unknown"] = create_int_feature([feature.is_unknown])
        features["answer_mask"] = create_int_feature(feature.answer_mask)
        features["answer_ids"] = create_int_feature(feature.answer_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example.SerializeToString()

def reverse_qas(qa_str, marker_str):
    ques = []
    ans = []

    qas = []
    k = 0
    marks = marker_str.split()
    qas_str = qa_str.split()
    for i in range(len(marks)):
        l = len(marks[i])
        qas.append(qas_str[k:k + l])
        k += l

    for ms, qas in zip(marks, qas):
        q = []
        a = []
        for m, qa in zip(ms, qas):
            if m == 'Q':
                q.append(qa)
            if m == 'A':
                a.append(qa)
        ques.append(' '.join(q))
        ans.append(' '.join(a))
    return ques, ans

def tokenize_qa_history(tokenizer,ques,ans):
    qas_tokens=[]
    q_type_ids=[]
    ans_tokens=[]
    qa_history_tokens=[]
    qa_history_type_ids=[]

    for q,a in zip(ques,ans):
        qas_tokens.append(tokenizer.tokenize(''.join(q)))
        ans_tokens.append(tokenizer.tokenize(''.join(a)))

    for q, a in zip(qas_tokens,ans_tokens):
        q_type_ids=[]
        a_type_ids=[]
        for i in range(len(q)):
          q_type_ids.append(2)
        qa_history_tokens.extend(q)
        qa_history_type_ids.extend(q_type_ids)

        for i in range(len(a)):
          a_type_ids.append(3)
        qa_history_tokens.extend(a)
        qa_history_type_ids.extend(a_type_ids)

    return  qa_history_tokens,qa_history_type_ids


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The CoQA annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in CoQA, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def convert_examples_to_features(examples, tokenizer):
    max_query_length = coqa_settings['max_query_length']
    max_answer_length = coqa_settings['max_answer_length']
    max_seq_length = coqa_settings['max_seq_length']
    doc_stride = coqa_settings['doc_stride']

    is_training=False

    base_id = 1000000000
    unique_id = base_id
    #token types: 1 - original context words, 0 - current question  2 previous questions 3 - answeers to the previous questions
    features=[]

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        max_qa_history = max_query_length - len(query_tokens)

        qas,ans = reverse_qas(example.qa_history_text,example.qa_history_marker_text)

        qa_history_tokens, qa_history_type_ids =tokenize_qa_history(tokenizer,qas,ans)

        qa_history_tokens = qa_history_tokens[-1 * (len(qa_history_tokens) % max_qa_history):]
        qa_history_type_ids=qa_history_type_ids[-1 * (len(qa_history_tokens) % max_qa_history):]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        answer_tokens = []
        decode_tokens =[]
        decode_tokens.append("[START]")

        if is_training:
            answer_tokens =answer_tokens+ tokenizer.tokenize(example.gold_answer_text)
            if len(answer_tokens) > max_answer_length - 1:
                answer_tokens = answer_tokens[0:max_answer_length - 1]

            decode_tokens = decode_tokens + answer_tokens
            answer_tokens.append("[STOP]")


            tok_start_position = orig_to_tok_index[example.start_position]

            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP], [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - len(qa_history_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)



            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)

            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)

            if len(qa_history_tokens) > 0:
                for token, t in zip(qa_history_tokens, qa_history_type_ids):
                    tokens.append(token)
                    segment_ids.append(1)

            tokens.append("[SEP]")
            segment_ids.append(1)


            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            answer_ids = tokenizer.convert_tokens_to_ids(answer_tokens)
            decode_ids = tokenizer.convert_tokens_to_ids(decode_tokens)


            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            answer_mask = [1] * len(answer_ids)
            decode_mask= [1] * len(decode_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            while len(answer_ids) < max_answer_length:
                answer_ids.append(0)
                answer_mask.append(0)

            while len(decode_ids) < max_answer_length:
                decode_ids.append(0)
                decode_mask.append(0)
            if len(answer_ids)!=max_answer_length:
                print("===>", len(answer_ids), max_answer_length)
                print(answer_tokens)
            if len(input_ids)!=max_seq_length:
                print("===>", len(input_ids), max_seq_length)
                print(input_ids)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(answer_ids) == max_answer_length
            assert len(decode_ids) == max_answer_length
            assert len(decode_mask) == max_answer_length

            assert len(answer_mask) == max_answer_length

            start_position = None
            end_position = None
            is_yes = example.is_yes
            is_no = example.is_no
            is_unknown = example.is_unknown

            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if is_yes==1 or is_no==1 or is_unknown==1:
                    start_position = -1
                    end_position = -1
                else:
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        #doc_offset = 1  # we added [CLS] infront of context doc (note we moved query to the end #len(query_tokens) + 2
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                    if end_position <0 or start_position<0:
                        end_position=0
                        start_position=0


            feature = coqa_lib.InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                decode_ids=decode_ids,
                decode_mask=decode_mask,
                answer_ids=answer_ids,
                answer_mask = answer_mask,
                start_position=start_position,
                end_position=end_position,
                is_yes = is_yes,
                is_no = is_no,
                is_unknown = is_unknown
               )

            # Run callback
            features.append(feature)

            unique_id += 1
    return features

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):  # pylint: disable=consider-using-enumerate
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def get_raw_results(predictions):
  """Converts multi-replica predictions to RawResult."""
  for unique_ids, start_logits, end_logits  in zip(predictions['unique_ids'],
                                                  predictions['start_logits'],
                                                  predictions['end_logits']
                                                  ):

          yield coqa_lib.RawResult(
              unique_id=unique_ids.numpy(),
              start_logits=start_logits.numpy(),
              end_logits=end_logits.numpy()
          )
def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the CoQA eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def askCoqaAI( question,turn,story ,history ):
    n_best_size=coqa_settings['n_best_size']
    max_answer_length=coqa_settings['max_answer_length']

    examples=make_coqa_example(question,turn ,story ,history)

    vocabfile =  os.path.join(os.path.dirname(__file__), coqa_settings['vocab_file'])

    tokenizer = FullTokenizer(vocab_file=vocabfile)

    features = convert_examples_to_features(examples,tokenizer)

    # defining the api-endpoint
    API_ENDPOINT =coqa_settings['model_api']

    # your API key here
    #API_KEY = "XXXXXXXXXXXXXXXXX"

    # data to be sent to api
    data ={ "inputs":{
        "input_ids":[features[0].input_ids],
        "unique_ids":[features[0].unique_id],
        "input_mask":   [features[0].input_mask],
        "segment_ids":[features[0].segment_ids]
        }
    }

    # sending post request and saving response as response object
    r = requests.post(url=API_ENDPOINT, data=json.dumps(data))
    outputs = json.loads(r.text)['outputs']

    unique_ids = tf.convert_to_tensor(outputs['unique_ids'])
    start_logits = tf.convert_to_tensor(outputs['bert_span_logits'])
    end_logits = tf.convert_to_tensor(outputs['bert_span_logits_1'])
    ynu_logits = tf.convert_to_tensor(outputs['yesnounknown_logits'] )

    s_logits = tf.concat([start_logits, ynu_logits], axis=-1)

    e_logits = tf.concat([end_logits, ynu_logits], axis=-1)

    prediction = dict(
        unique_ids=unique_ids,
        start_logits=s_logits,
        end_logits=e_logits
    )
    results=[]

    for rlt in get_raw_results(prediction):
         results.append(rlt)


    example_index_to_features = collections.defaultdict(list)
    for feature in features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    _Prediction = collections.namedtuple('Prediction', ['id', 'turn_id', 'answer'])

    all_predictions = []  # collections.OrderedDict()

    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            if feature.unique_id not in unique_id_to_result:
                logging.info('%s not found.' % (feature.unique_id))
            if feature.unique_id in unique_id_to_result:
                result = unique_id_to_result[feature.unique_id]
                start_indexes = _get_best_indexes(result.start_logits, n_best_size )
                end_indexes = _get_best_indexes(result.end_logits, n_best_size )
                # if we could have irrelevant answers, get the min score of irrelevant

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        # here we have to add 3 for yes,no,unknown for available tokens

                        if start_index >= len(feature.tokens) + 3:
                            continue
                        if end_index >= len(feature.tokens) + 3:
                            continue
                        if start_index < len(feature.tokens):  # only for text span start and end
                            if start_index not in feature.token_to_orig_map:
                                continue
                            if end_index not in feature.token_to_orig_map:
                                continue
                            if not feature.token_is_max_context.get(start_index, False):
                                continue
                            if end_index < start_index:
                                continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:

            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index == 384:
                final_text = "yes"
            elif pred.start_index == 385:
                final_text = "no"
            elif pred.start_index == 386:
                final_text = "unknown"
            elif pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                do_lower_case=True

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

    return nbest_json[0]["text"]

