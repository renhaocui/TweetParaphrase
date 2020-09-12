import numpy as np
import pickle
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model, load_model
from keras_attention import AttentionLayer
from keras.preprocessing.sequence import pad_sequences

max_len_source = 40
max_len_target = 40
latent_dim = 300

def preProcess(inputText, source_tokenizer):
    textVector = source_tokenizer.texts_to_sequences([inputText])
    textVector = pad_sequences(textVector, maxlen=max_len_source, padding='post', truncating='post')
    return textVector


def loadModel(modelName):
    model = load_model('model/stackedLSTM/' + modelName + '/model.h5', custom_objects={'AttentionLayer': AttentionLayer})
    #model.summary()
    encoder_inputs = model.input[0]
    encoder_outputs, state_h, state_c = model.layers[6].output
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_hidden_input_h')
    decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_hidden_input_c')
    decoder_hidden_state_input = Input(shape=(max_len_source, latent_dim))
    dec_emb_layer = model.layers[5]
    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_lstm = model.layers[7]
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
    attn_layer = model.layers[8]
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
    decoder_dense = model.layers[10]
    decoder_outputs2 = decoder_dense(decoder_inf_concat)
    decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + [state_h2, state_c2])

    source_tokenizer = pickle.load(open('model/stackedLSTM/' + modelName + '/source.tk', 'rb'))
    target_tokenizer = pickle.load(open('model/stackedLSTM/' + modelName + '/target.tk', 'rb'))
    index2word_target = target_tokenizer.index_word
    word2index_target = target_tokenizer.word_index

    return encoder_model, decoder_model, source_tokenizer, target_tokenizer, index2word_target, word2index_target


def decode(input_seq, encoder_model, decoder_model):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    print('Input text encoded')
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2index_target['ssoss']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = index2word_target[sampled_token_index]
        if sampled_token != 'eeoss':
            decoded_sentence += ' ' + sampled_token
            if sampled_token == 'eeoss' or len(decoded_sentence.split()) >= (max_len_target - 1):
                stop_condition = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        e_h, e_c = h, c
    return decoded_sentence


def decode_beam(input_seq, encoder_model, decoder_model, index2word_target, word2index_target, topK):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2index_target['ssoss']

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
            if candidate[1][-1] != word2index_target['eeoss'] and len(candidate[1]) <= max_len_target and candidate[1][-1] != 0:
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
        if item == 0:
            print('aaa')
        if index2word_target[item] != 'eeoss':
            outputSentence += index2word_target[item] + ' '

    return outputSentence.strip()


if __name__ == '__main__':
    '''
    inputText = "she does n't hear it probably .	i guess he ca n't hear ."
    if len(inputText) == 0:
        with open('data/test.txt', 'r') as testFile:
            for line in testFile:
                testData.append(line.strip())
    else:
        testData.append(inputText)
    '''

    encoder, decoder, source_tokenizer, target_tokenizer, index2word_target, word2index_target = loadModel('Tweet_URL')
    print('Model Loaded')

    testData = ['the transistors are almost never put into the hands of human beings .', 'Medical experts said the condition was mildly worrying but easily-manageable.',
                'Hopefully going to see the purge tonight', 'officer killed after being hit by car during chase is identified', 'Now this is how to spend a Saturday.']
    for content in testData:
        text = preProcess(content, source_tokenizer)
        #output = decode(text, encoder, decoder)
        output = decode_beam(text, encoder, decoder, index2word_target, word2index_target, 3)
        print('Input: ' + content)
        print('Output: ' + output)
