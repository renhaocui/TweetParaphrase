import tensorflow_datasets as tfds
import tensorflow as tf
import time
import transformer_model as tm
import pickle


'''
# load MRPC data
example, metadata = tfds.load('glue/mrpc', with_info=True)
train_example = example['train'].concatenate(example['test'])
val_example = example['validation']

train_example = train_example.filter(lambda x: x['label'] == 1)
val_example = val_example.filter(lambda x: x['label'] == 1)

def gen1():
  for data in train_example:
    yield (data['sentence1'], data['sentence2'])

def gen2():
  for data in val_example:
    yield (data['sentence1'], data['sentence2'])
'''

dataFile = 'data/para-nmt-5m-processed.txt'
modelLocation = 'model/transformer'

trainList = []
valList = []
with open(dataFile, 'r') as inputFile:
    for index, line in enumerate(inputFile):
        temp = line.strip().split('\t')
        if index % 20 == 0:
            valList.append((temp[0], temp[1]))
        trainList.append((temp[0], temp[1]))

def gen_train():
    for item in trainList:
        yield item

def gen_val():
    for item in valList:
        yield item


train_examples = tf.python.data.ops.dataset_ops.DatasetV1Adapter.from_generator(gen_train, (tf.string, tf.string))
val_examples = tf.python.data.ops.dataset_ops.DatasetV1Adapter.from_generator(gen_val, (tf.string, tf.string))

tokenizer_src = tfds.features.text.SubwordTextEncoder.build_from_corpus((src.numpy() for src, trg in train_examples), target_vocab_size=2**13)
tokenizer_trg = tfds.features.text.SubwordTextEncoder.build_from_corpus((trg.numpy() for src, trg in train_examples), target_vocab_size=2**13)

sample_string = 'Transformer is awesome.'
tokenized_string = tokenizer_src.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))
original_string = tokenizer_src.decode(tokenized_string)
print('The original string: {}'.format(original_string))
assert original_string == sample_string

input_vocab_size = tokenizer_src.vocab_size + 2
target_vocab_size = tokenizer_trg.vocab_size + 2

def encode(lang1, lang2):
    lang1 = [tokenizer_src.vocab_size] + tokenizer_src.encode(lang1.numpy()) + [tokenizer_src.vocab_size + 1]
    lang2 = [tokenizer_trg.vocab_size] + tokenizer_trg.encode(lang2.numpy()) + [tokenizer_trg.vocab_size + 1]
    return lang1, lang2


def tf_encode(src, trg):
    result_src, result_trg = tf.py_function(encode, [src, trg], [tf.int64, tf.int64])
    result_src.set_shape([None])
    result_trg.set_shape([None])
    return result_src, result_trg


def filter_max_length(x, y, max_length=tm.MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


train_preprocessed = (train_examples.map(tf_encode).filter(filter_max_length).cache().shuffle(tm.BUFFER_SIZE))
val_preprocessed = (val_examples.map(tf_encode).filter(filter_max_length))

train_dataset = (train_preprocessed.padded_batch(tm.BATCH_SIZE, padded_shapes=([None], [None])).prefetch(tf.data.experimental.AUTOTUNE))
val_dataset = (val_preprocessed.padded_batch(tm.BATCH_SIZE, padded_shapes=([None], [None])))
train_dataset = (train_preprocessed.padded_batch(tm.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE))
val_dataset = (val_preprocessed.padded_batch(tm.BATCH_SIZE))

src_batch, trg_batch = next(iter(val_dataset))
learning_rate = tm.CustomSchedule(tm.d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
transformer = tm.Transformer(tm.num_layers, tm.d_model, tm.num_heads, tm.dff, input_vocab_size, target_vocab_size, pe_input=input_vocab_size, pe_target=target_vocab_size, rate=tm.dropout_rate)

checkpoint_path = modelLocation + '/checkpoints'
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(src, trg):
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = tm.create_masks(src, trg_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(src, trg_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = tm.loss_function(trg_real, predictions, loss_object)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(trg_real, predictions)


for epoch in range(tm.EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (src, trg)) in enumerate(train_dataset):
        train_step(src, trg)
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

with open(modelLocation + '/source.tk', 'wb') as handle:
    pickle.dump(tokenizer_src, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(modelLocation + '/target.tk', 'wb') as handle:
    pickle.dump(tokenizer_trg, handle, protocol=pickle.HIGHEST_PROTOCOL)
transformer.save_weights(modelLocation + '/model_weights', save_format='tf')
print('Model Saved')
