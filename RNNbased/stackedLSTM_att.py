from keras_attention import AttentionLayer
import numpy as np
import re
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
import warnings
from contextlib import redirect_stdout
warnings.filterwarnings("ignore")

max_len_source = 40
max_len_target = 40
batch_size = 128
n_epochs = 50
latent_dim = 300
vocab_size = 10000

######### helper functions ###########

def text_cleaner(text):
    newString = text.lower()
    newString = newString.replace('\r', ' ').replace('\n', ' ')
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"', '', newString)
    newString = re.sub(r"'s\b", "", newString)
    #newString = re.sub("[^a-zA-Z]", " ", newString)
    return newString


def loadData(dataFilename):
    print('Loading Data...')
    data = {'source': [], 'target': []}
    with open(dataFilename, 'r') as dataFile:
        for line in dataFile:
            item = line.strip().split('\t')
            data['source'].append(text_cleaner(item[0]))
            data['target'].append(text_cleaner(item[1]))
    return data

'''
def seq2target(input_seq):
    newString=''
    for i in input_seq:
        if(i != 0 and i != word2index_y['ssoss']) and i != word2index_y['eeoss']:
            newString = newString + index2word_y[i] + ' '
    return newString.strip()


def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if i != 0:
            newString = newString + index2word_x[i] + ' '
    return newString
'''

######### load & proprocess ##########
data = loadData('data/Tweet_URL/2016_Oct_10--2017_Jan_08_paraphrase.txt')
data['target'] = list(map(lambda x: 'ssoss ' + x + ' eeoss', data['target']))
x_tr, x_val, y_tr, y_val = train_test_split(data['source'], data['target'], test_size=0.1, random_state=2020, shuffle=True)
x_tokenizer = Tokenizer(num_words=vocab_size)
x_tokenizer.fit_on_texts(x_tr+x_val)
x_tr = x_tokenizer.texts_to_sequences(x_tr)
x_val = x_tokenizer.texts_to_sequences(x_val)
x_tr = pad_sequences(x_tr, maxlen=max_len_source, padding='post', truncating='post')
x_val = pad_sequences(x_val, maxlen=max_len_source, padding='post', truncating='post')
x_voc_size = len(x_tokenizer.word_index) + 1
y_tokenizer = Tokenizer(num_words=vocab_size)
y_tokenizer.fit_on_texts(y_tr+y_val)
y_tr = y_tokenizer.texts_to_sequences(y_tr)
y_val = y_tokenizer.texts_to_sequences(y_val)
y_tr = pad_sequences(y_tr, maxlen=max_len_target, padding='post', truncating='post')
y_val = pad_sequences(y_val, maxlen=max_len_target, padding='post', truncating='post')
y_voc_size = len(y_tokenizer.word_index) + 1

######### models #########
K.clear_session()
# Encoder
encoder_inputs = Input(shape=(max_len_source,))
enc_emb = Embedding(x_voc_size, latent_dim, trainable=True, name='encoder_embedding', mask_zero=True)(encoder_inputs)
#LSTM 1
encoder_LSTM1 = LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_LSTM1', recurrent_dropout=0.2)
encoder_output1, state_h1, state_c1 = encoder_LSTM1(enc_emb)
#LSTM 2
encoder_LSTM2 = LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_LSTM2')
encoder_output2, state_h2, state_c2 = encoder_LSTM2(encoder_output1)
#LSTM 3
encoder_LSTM3 = LSTM(latent_dim, return_state=True, return_sequences=True, name='encoder_LSTM3', recurrent_dropout=0.2)
encoder_outputs, state_h, state_c = encoder_LSTM3(encoder_output2)

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(y_voc_size, latent_dim, trainable=True, name='decoder_embedding', mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)
#LSTM using encoder_states as initial state
decoder_LSTM = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_LSTM', recurrent_dropout=0.2)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_LSTM(dec_emb, initial_state=[state_h, state_c])
# Attention Layer
attn_layer = AttentionLayer(name='decoder_attention')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
# Concatenate attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='att_concat_layer')([decoder_outputs, attn_out])
#Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'), name='output_layer')
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
with open('model/model.summary', 'w') as f:
    with redirect_stdout(f):
        model.summary()
#model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=n_epochs, callbacks=[es], batch_size=batch_size, validation_data=([x_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]), verbose=2)
# Save the trained model
with open('model/stackedLSTM/Tweet_URL/source.tk', 'wb') as handle:
    pickle.dump(x_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('model/stackedLSTM/Tweet_URL/target.tk', 'wb') as handle:
    pickle.dump(y_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model.save('model/stackedLSTM/Tweet_URL/model.h5')
#model.save_weights('model/stackedLSTM_modelWeights.h5')
print('Model Saved')

'''
########## inference ##########
# Load preprocess
#x_tokenizer = pickle.load(open('model/text.tk', 'rb'))
#y_tokenizer = pickle.load(open('model/summary.tk', 'rb'))
index2word_y = y_tokenizer.index_word
index2word_x = x_tokenizer.index_word
word2index_y = y_tokenizer.word_index

# Load models
#model = load_model('model/stackedLSTM_model.h5')
# Load encoder
#encoder_inputs = model.input[0]
#encoder_outputs, state_h, state_c = model.layers[6].output
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
# Load decoder
#decoder_inputs = model.input[1]
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_len_source, latent_dim))
# Get the embeddings of the decoder sequence
#dec_emb_layer = model.layers[5]
dec_emb2 = dec_emb_layer(decoder_inputs)
#decoder_LSTM = model.layers[7]
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_LSTM(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
#attention inference
#attn_layer = model.layers[8]
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
# A dense softmax layer to generate prob dist. over the target vocabulary
#decoder_dense = model.layers[10]
decoder_outputs2 = decoder_dense(decoder_inf_concat)
# Final decoder model
decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + [state_h2, state_c2])

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Chose the 'ssoss' word as the first word of the target sequence
    target_seq[0, 0] = word2index_y['ssoss']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #if sampled_token_index == 0:
        #    continue
        sampled_token = index2word_y[sampled_token_index]
        if sampled_token != 'eeoss':
            decoded_sentence += ' ' + sampled_token
            # Exit condition: either hit max length or find EOS.
            if sampled_token == 'eeoss' or len(decoded_sentence.split()) >= (max_len_target - 1):
                stop_condition = True
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        # Update internal states
        e_h, e_c = h, c
    return decoded_sentence


def decode_beam(input_seq, topK):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2index_y['ssoss']

    output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
    tempNLLProbs = -np.log(output_tokens.flatten())
    all_candidates = []
    rankedCandidates = np.argsort(tempNLLProbs)
    for i in range(topK):
        candidate = (tempNLLProbs[rankedCandidates[i]], [rankedCandidates[i]], h, c)
        all_candidates.append(candidate)

    for i in range(max_len_target):
        tempCandidates = []
        for candidate in all_candidates:
            if candidate[1][-1] != word2index_y['eeoss'] and len(candidate[1]) <= max_len_target and candidate[1][-1] != 0:
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = candidate[1][-1]
                currentScore = candidate[0]
                e_h = candidate[2]
                e_c = candidate[3]
                output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
                tempNLLProbs = -np.log(output_tokens.flatten())
                rankedCandidates = np.argsort(tempNLLProbs)
                for j in range(topK):
                    currentTokens = candidate[1][:]
                    currentTokens.append(rankedCandidates[j])
                    temp = (currentScore + tempNLLProbs[rankedCandidates[j]], currentTokens, h, c)
                    tempCandidates.append(temp)
        if len(tempCandidates) == 0:
            break
        tempCandidates.sort()
        if len(tempCandidates) > topK:
            all_candidates = tempCandidates[:topK]
        else:
            all_candidates = tempCandidates[:]

    #print(all_candidates[0][0], all_candidates[0][1])
    #print(all_candidates[1][0], all_candidates[1][1])
    outputSentence = ''
    for item in all_candidates[0][1]:
        if index2word_y[item] != 'eeoss':
            outputSentence += index2word_y[item] + ' '

    return outputSentence.strip()


with open('output/sample.output', 'w') as sampleFile:
    for i in range(len(x_val)):
        try:
            predictedSeq = decode_beam(x_val[i].reshape(1, max_len_source), 3)
            #print("Source:", seq2text(x_val[i]))
            sampleFile.write('Source: ' + seq2text(x_val[i]) + '\n')
            #print("Original target:", seq2target(y_val[i]))
            sampleFile.write("Original target: " + seq2target(y_val[i]) + '\n')
            #print("Predicted target: ", predictedSeq)
            sampleFile.write("Predicted target: " + predictedSeq + '\n\n')
        except:
            continue
'''