from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

import os
import re
curr_dir = os.getcwd()

#Data Processing

def folder_data(folder_path):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(folder_path):
        with tf.gfile.GFile(os.path.join(folder_path, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

def load_dataset(folder_path):
    
    pos_df = folder_data(os.path.join(folder_path, "pos"))
    neg_df = folder_data(os.path.join(folder_path, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
	
train_df = load_dataset(os.path.join(curr_dir, "Dataset", "aclImdb", "train"))
test_df = load_dataset(os.path.join(curr_dir,"Dataset", "aclImdb", "test"))

train_df_sample = train_df.sample(2200)
test_df_sample = test_df.sample(2200)

#BERT

URL = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def bert_tokenizer():
    with tf.Graph().as_default():
        URL = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        bert_module = hub.Module(URL)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as ses:
            vocab, lower_case = ses.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
        
    return bert.tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=lower_case)
	
tokenizer = bert_tokenizer()
tokenizer.tokenize("End of Earth wil not be the end of Humanity")

train_ex = train_df_sample.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                     text_a = x['sentence'], 
                                                                     text_b = None, 
                                                                     label = x['polarity']), 
                                                                     axis = 1)

test_ex = test_df_sample.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                    text_a = x['sentence'], 
                                                                    text_b = None, 
                                                                    label = x['polarity']), axis = 1)
																	
train_features = bert.run_classifier.convert_examples_to_features(train_ex, [0,1], 128, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_ex, [0,1], 128, tokenizer)

def model_initialization(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
    bert_module = hub.Module(URL, trainable=True)
    bert_inputs = dict(input_ids = input_ids, 
                       input_mask = input_mask, 
                       segment_ids = segment_ids)
    bert_outputs = bert_module(inputs = bert_inputs, 
                               signature = "tokens", 
                               as_dict = True)
    output_layer = bert_outputs["pooled_output"]
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable("output_weights", 
                                     [num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logit = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logit, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        if is_predicting:
            return (predicted_labels, log_probs)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)
		
		
def model_func(num_labels, learning_rate, num_train_steps, num_warmup_steps):   
    def model_fn(features, labels, mode, params):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
        if not is_predicting:
            (loss, predicted_labels, log_probs) = model_initialization(is_predicting, input_ids, 
                                                               input_mask, segment_ids, 
                                                               label_ids, num_labels)
        
            train_op = bert.optimization.create_optimizer(loss, 
                                                          learning_rate, 
                                                          num_train_steps, 
                                                          num_warmup_steps, 
                                                          use_tpu=False)
        
            def metric_fn(label_ids, predicted_labels):               
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
                auc = tf.metrics.auc(label_ids, predicted_labels)
                recall = tf.metrics.recall(label_ids, predicted_labels)
        
                return {"eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "auc": auc,
                        "precision": precision
                       }
    
            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
													eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = model_initialization(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
        
            predictions = {'probabilities': log_probs,
                           'labels': predicted_labels
                        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    return model_fn
	
	
test_config = tf.estimator.RunConfig(
    model_dir=curr_dir,
    save_summary_steps=100,
    save_checkpoints_steps=500)	

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1

num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

model_fn = model_func(num_labels=len([0, 1]),
                            learning_rate=2e-05,
                            num_train_steps=num_train_steps,
                            num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   config=test_config,
                                   params={"batch_size": BATCH_SIZE})
	
	
	
train_input_fn = bert.run_classifier.input_fn_builder(features=train_features,
                                                      seq_length=128,
                                                      is_training=True,
                                                      drop_remainder=False)

estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

test_input_fn = run_classifier.input_fn_builder(features=test_features,
                                                seq_length=128,
                                                is_training=False,
                                                drop_remainder=False)
												
estimator.evaluate(input_fn=test_input_fn, steps=None)

													  
#ELMo	
	
	
url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)	
	
train_x1 = list(train_df['sentence'])
train_y1 = list(train_df['polarity'])	
	
test_x1 = list(test_df['sentence'])
test_y1 = list(test_df['polarity'])

x_train = np.asarray(train_x1[:2500])
y_train = np.asarray(train_y1[:2500])

x_test = np.asarray(test_x1[2500:3500])
y_test = np.asarray(test_y1[2500:3500])

from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K


def ELMoEmbd(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
	
input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbd, output_shape=(1024, ))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(2, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    history = model.fit(x_train, y_train, epochs=1, batch_size=16)
    model.save_weights('./elmo-model.h5')

	
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./elmo-model.h5')  
    predicts = model.predict(x_test, batch_size=16)	
	
accu  = metrics.accuracy(y_test, y_preds)	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	