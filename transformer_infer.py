import tensorflow as tf
import transformer_model as tm
import pickle
import numpy as np


def generate(inp_sentence, tokenizer_src, tokenizer_trg, MAX_LENGTH):
    start_token = [tokenizer_src.vocab_size]
    end_token = [tokenizer_src.vocab_size + 1]
    inp_sentence = start_token + tokenizer_src.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_trg.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = tm.create_masks(encoder_input, output)
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_trg.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def generate_with_beam(inp_sentence, tokenizer_src, tokenizer_trg, MAX_LENGTH, beam_size):
    start_token = [tokenizer_src.vocab_size]
    end_token = [tokenizer_src.vocab_size + 1]
    inp_sentence = start_token + tokenizer_src.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_trg.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    eosIndex = tokenizer_trg.vocab_size + 1

    for step in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = tm.create_masks(encoder_input, output)
        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]
        if step == 0:
            probs, indices = tf.math.top_k(predictions, k=beam_size, sorted=True)
            probs = tf.reshape(probs, [beam_size])
            probs = tf.map_fn(np.log, tf.keras.utils.normalize(probs, axis=-1, order=1)).numpy()
            indices = tf.reshape(indices, [beam_size])
            topSeqCandidates = tf.split(indices, beam_size, -1)
            topSeqNLLProbs = probs[0]
        else:
            tempSeqCandidates = []
            tempSeqNLLProbs = np.array([])
            for index, currentToken in enumerate(topSeqCandidates):
                currentSeqLength = tf.shape(currentToken).numpy()[0]
                if currentToken[currentSeqLength-1].numpy() != eosIndex:
                    currentToken = tf.expand_dims(currentToken, 0)
                    enc_padding_mask, combined_mask, dec_padding_mask = tm.create_masks(encoder_input, currentToken)
                    predictions, attention_weights = transformer(encoder_input, currentToken, False, enc_padding_mask, combined_mask, dec_padding_mask)
                    predictions = predictions[:, -1:, :]
                    probs, indices = tf.math.top_k(predictions, k=beam_size, sorted=True)

                    probs = tf.reshape(probs, [beam_size])
                    probs = tf.map_fn(np.log, tf.keras.utils.normalize(probs, axis=-1, order=1)).numpy()
                    currentSeqNLLProbs = probs[0] + topSeqNLLProbs[index]
                    currentSeqNLLProbs = currentSeqNLLProbs / (currentSeqLength + 1)
                    tempSeqNLLProbs = np.concatenate((tempSeqNLLProbs, currentSeqNLLProbs), axis=None)

                    indices = tf.reshape(indices, [beam_size])
                    currentCandidates = tf.split(indices, beam_size, -1)
                    for token in currentCandidates:
                        token = tf.expand_dims(token, 0)
                        tempSeqCandidates.append(tf.concat([currentToken, token], axis=-1))
                else:
                    tempSeqNLLProbs = np.concatenate((tempSeqNLLProbs, topSeqNLLProbs[index]), axis=None)
                    tempSeqCandidates.append(tf.expand_dims(currentToken, 0))

            topIndices = np.flip(np.argsort(tempSeqNLLProbs))[:beam_size]
            topSeqNLLProbs = tempSeqNLLProbs[topIndices]
            topSeqCandidates = []
            for index in topIndices:
                topSeqCandidates.append(tf.squeeze(tempSeqCandidates[index], axis=0))

            #print(topSeqNLLProbs)
            #print(topSeqCandidates)

    return topSeqCandidates, attention_weights


def paraphrase(sentence, tokenizer_src, tokenizer_trg):
    result, attention_weights = generate(sentence, tokenizer_src, tokenizer_trg, tm.MAX_LENGTH)
    print(result)
    predicted_sentence = tokenizer_trg.decode([i for i in result if i < tokenizer_trg.vocab_size])
    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))


def paraphrase_beam(sentence, tokenizer_src, tokenizer_trg, beam_size):
    resultList, attention_weights = generate_with_beam(sentence, tokenizer_src, tokenizer_trg, tm.MAX_LENGTH, beam_size)
    print('Input: {}'.format(sentence))
    for index, result in enumerate(resultList):
        predicted_sentence = tokenizer_trg.decode([i for i in result if i < tokenizer_trg.vocab_size])
        print('Output ' + str(index) + ': ' + predicted_sentence)


if __name__ == '__main__':
    modelLocation = 'model/Transformer/PIT2015'
    tokenizer_src = pickle.load(open(modelLocation + '/source.tk', 'rb'))
    tokenizer_trg = pickle.load(open(modelLocation + '/target.tk', 'rb'))
    input_vocab_size = tokenizer_src.vocab_size + 2
    target_vocab_size = tokenizer_trg.vocab_size + 2

    transformer = tm.Transformer(tm.num_layers, tm.d_model, tm.num_heads, tm.dff, input_vocab_size, target_vocab_size, pe_input=input_vocab_size, pe_target=target_vocab_size, rate=tm.dropout_rate)
    transformer.load_weights(modelLocation + '/model_weights')
    sentenceList = ['the transistors are almost never put into the hands of human beings .']
    #sentenceList = ['the transistors are almost never put into the hands of human beings .', 'Medical experts said the condition was mildly worrying but easily-manageable.', 'Hopefully going to see the purge tonight', 'officer killed after being hit by car during chase is identified', 'Now this is how to spend a Saturday.']

    for sentence in sentenceList:
        #paraphrase(sentence, tokenizer_src, tokenizer_trg)
        paraphrase_beam(sentence, tokenizer_src, tokenizer_trg, 3)

