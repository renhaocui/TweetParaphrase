from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math


def computeSimilarity(originalTextList, paraphraseTextList):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    #model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

    outputScores = []
    originalSentEmbeddings = model.encode(originalTextList)
    paraphraseSentEmbeddings = model.encode(paraphraseTextList)
    for originalSent, paraphraseSent in zip(originalSentEmbeddings, paraphraseSentEmbeddings):
        outputScores.append(cosine_similarity([originalSent], [paraphraseSent]).tolist()[0][0])

    return outputScores


def computeBLEU(originalTextList, paraphraseTextList):
    outputScores = []
    for originalSent, paraphraseSent in zip(originalTextList, paraphraseTextList):
        originalTokens = originalSent.split(' ')
        paraphraseTokens = paraphraseSent.split(' ')
        outputScores.append(sentence_bleu([originalTokens], paraphraseTokens, weights=(1, 0, 0, 0)))

    return outputScores


def computeROUGE(originalTextList, paraphraseTextList):
    outputScores = []
    for originalSent, paraphraseSent in zip(originalTextList, paraphraseTextList):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(originalSent, paraphraseSent)
        outputScores.append(scores['rougeL'][0])

    return outputScores


def completeEvaluate(originalTextFilename, generatedTextFilename, reportFilename):
    originalList = []
    paraphraseList = []
    originalFile = open(originalTextFilename, 'r')
    generatedFile = open(generatedTextFilename, 'r')
    for originalLine, generatedLine in zip(originalFile, generatedFile):
        if '<ERROR>' not in generatedLine:
            paraphraseList.append(generatedLine.strip())
            originalList.append(originalLine.strip())
    originalFile.close()
    generatedFile.close()

    print(len(originalList), len(paraphraseList))
    BLEUScores = computeBLEU(originalList, paraphraseList)
    ROUGEScores = computeROUGE(originalList, paraphraseList)
    SimilarityScores = computeSimilarity(originalList, paraphraseList)
    print('Average BLEU: ' + str(sum(BLEUScores) / len(BLEUScores)))
    print('Average ROUGE: ' + str(sum(ROUGEScores) / len(ROUGEScores)))
    print('Average Similarity: ' + str(sum(SimilarityScores) / len(SimilarityScores)))

    with open(reportFilename, 'w') as scoreReportFile:
        for bScore, rScore, sScore in zip(BLEUScores, ROUGEScores, SimilarityScores):
            scoreReportFile.write(str(bScore) + '\t' + str(rScore) + '\t' + str(sScore) + '\n')

    print('DONE')


def verifyKeyComponents(generatedFilename, itemFilename, reportFilename):
    generatedFile = open(generatedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    containedCount = 0
    totalCount = 0
    with open(reportFilename, 'w') as reportFile:
        for generatedLine, itemLine in zip(generatedFile, itemFile):
            if '<ERROR>' not in generatedLine:
                itemData = json.loads(itemLine.strip())
                itemList = itemData['<RST>'] + itemData['<EXN>']
                lineItemCount = 0
                lineTotalCount = 0
                for item in itemList:
                    lineTotalCount += 1
                    if item.lower() in generatedLine.lower():
                        lineItemCount += 1
                containedCount += (lineItemCount/lineTotalCount)
                totalCount += 1
                reportFile.write(str(lineItemCount/lineTotalCount)+'\n')
            else:
                reportFile.write(generatedLine)

    itemFile.close()
    generatedFile.close()

    print('Coverage: ' + str(containedCount/totalCount))


def generatePerplexity(generatedFilename, reportFilename):
    device = 'cuda'
    model_id = 'gpt2-medium'
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    scoreSum = 0
    totalCount = 0
    reportFile = open(reportFilename, 'w')
    #totalData = ["ivent to the sound of a couple of fuses : 29 per cent of the fine is for the use of this."]
    with open(generatedFilename, 'r') as fi:
        for index, line in enumerate(fi):
            if index%5000 == 0:
                print(index)
            if '<ERROR>' in line:
                reportFile.write(line)
            else:
                input_sentence = line.strip()
                input_ids = torch.tensor(tokenizer.encode(input_sentence)).unsqueeze(0)
                input_ids = input_ids.to(device)
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                loss, logits = outputs[:2]
                ppl = math.exp(loss)
                scoreSum += ppl
                totalCount += 1
                reportFile.write(str(ppl)+'\n')

    reportFile.close()
    print(scoreSum/totalCount)
    print(index)
    print('DONE')
    return None



if __name__ == '__main__':
    completeEvaluate('drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_single/commTweets.content',
                     'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_single/results/commTweets.full.copynet',
                     'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_single/reports/commTweets.full.copynet.performance',
                     repeatTimes=3, split=True)

    verifyKeyComponents('drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_single/results/commTweets.full.copynet',
                        'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_single/commTweets.item',
                        'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_single/reports/commTweets.full.copynet.verify',
                        repeatTimes=3)

    generatePerplexity('drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_single/results/commTweets.full.copynet',
                       'drive/My Drive/Cui_workspace/Data/TweetParaphrase/commTweets/test_single/reports/commTweets.full.copynet.perplexity')
