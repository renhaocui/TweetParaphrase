import tensorflow as tf
from transformers import *
import re

modelName = 'GPT2+PMT_epoch3'
model_path = 'drive/My Drive/Cui_workspace/Model/TweetParaphrase_contained/'+modelName
testData = 'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_multiple/commTweets.tokenized_1'
resultFile = open('drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_multiple/results/commTweets.tokenized.contained_1.gpt2', 'w')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def extractParaphrases(input_ids):
  outText = ''
  outputs = model.generate(input_ids, do_sample=True, max_length=len(input_sentence), top_k=100, top_p=0.8, no_repeat_ngram_size=3, num_return_sequences=5)
  for output in outputs:
    generation = tokenizer.decode(output, skip_special_tokens=True)
    if ' >>><<< ' in generation:
      out = generation.split(' >>><<<')[1]
      out = re.split('>+<+', out)[0]
      outText += out + '\t'
  return outText


if __name__ == '__main__':
    with open(testData, 'r') as fr:
      for lineIndex, line in enumerate(fr):
        input_sentence = line.strip() + ' >>><<< '
        input_ids = tokenizer.encode(input_sentence.strip(), add_special_tokens=True, return_tensors='pt')
        outText = extractParaphrases(input_ids)
        if len(outText) > 10:
            resultFile.write(outText.strip() + '\n')
        else:
            resultFile.write('<ERROR>\n')
        if lineIndex % 500 == 0:
          print(lineIndex)
          resultFile.flush()
    resultFile.close()