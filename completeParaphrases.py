import json
#from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations
import random

#sentenceTransformerModel = SentenceTransformer('bert-base-nli-mean-tokens')
sentenceTransformerModel = None

def computeSimilarity(originalText, candidateTextList):
    outputScores = []
    originalSentEmbeddings = sentenceTransformerModel.encode([originalText])
    paraphraseSentEmbeddings = sentenceTransformerModel.encode(candidateTextList)
    for paraphrase in paraphraseSentEmbeddings:
        outputScores.append(cosine_similarity(originalSentEmbeddings, [paraphrase]).tolist()[0][0])

    return outputScores


def applyTokens_copynet(content, RSTList, EXNList):
    outList = []
    RSTIndex = 0
    EXNIndex = 0
    for word in content.split(' '):
        if word == '<RST>':
            if RSTIndex >= len(RSTList):
                outList.append(random.choice(RSTList))
            else:
                outList.append(RSTList[RSTIndex])
            RSTIndex += 1
        elif word == '<EXN>':
            if EXNIndex >= len(EXNList):
                outList.append(random.choice(EXNList))
            else:
                outList.append(EXNList[EXNIndex])
            EXNIndex += 1
        elif '<UNK>' not in word:
            outList.append(word)

    return ' '.join(outList).strip()


def applyTokens_GPT2(content, RSTList, EXNList):
    outList = []
    RSTIndex = 0
    EXNIndex = 0
    for word in content.split(' '):
        if word.lower().startswith('<'):
            if word.lower().startswith('<r') and len(RSTList) > 0:
                if RSTIndex >= len(RSTList):
                    outList.append(random.choice(RSTList))
                else:
                    outList.append(RSTList[RSTIndex])
                RSTIndex += 1
            if word.lower().startswith('<e') and len(EXNList) > 0:
                if EXNIndex >= len(EXNList):
                    outList.append(random.choice(EXNList))
                else:
                    outList.append(EXNList[EXNIndex])
                EXNIndex += 1
        else:
            outList.append(word)

    return ' '.join(outList).strip()


def realizeSingleToken_GPT2_strict(generatedFilename, itemFilename, outputFilename):
    dataFile = open(generatedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    with open(outputFilename, 'w') as outputFile:
        for dataLine, itemLine in zip(dataFile, itemFile):
            if '<ERROR>' in dataLine:
                outputFile.write('<ERROR>\n')
            else:
                itemData = json.loads(itemLine.strip())
                RSTData = itemData['<RST>']
                EXNData = itemData['<EXN>']
                RSTFlag = True if len(RSTData) > 0 else False
                EXNFlag = True if len(EXNData) > 0 else False
                contents = dataLine.strip().split('\t')
                candidateList = []
                for content in contents:
                    RSTValid = False
                    EXNValid = False
                    content = content.strip()
                    outWord = []
                    #content = content.replace('<R', ' <R').replace('<E', ' <E').replace('T>', 'T> ').replace('N>', 'N> ')
                    if RSTFlag:
                        if '<r' in content.lower():
                            RSTValid = True
                    else:
                        RSTValid = True
                    if EXNFlag:
                        if '<e' in content.lower():
                            EXNValid = True
                    else:
                        EXNValid = True
                    for index, word in enumerate(content.split(' ')):
                        if len(word) > 0:
                            if word.lower().startswith('<'):
                                if word.lower().startswith('<r') and RSTFlag:
                                    outWord.append(RSTData[0])
                                if word.lower().startswith('<e') and EXNFlag:
                                    outWord.append(EXNData[0])
                            else:
                                outWord.append(word)
                    content = ' '.join(outWord).replace('RST', '').replace('EXN', '')
                    if RSTValid and EXNValid:
                        if len(content) > 10 and len(content.split(' ')) > 3:
                            if content.count('*') < 5 and content.count('/') < 5 and content.count('#') < 5 and content.count('(') < 5 and content.count('.') < 7 and content.count(',') < 7 \
                                    and content.count('%') < 5 and content.count('@') < 5 and content.count('$') < 5 and content.count('&') < 5 and content.count('\\') < 5 \
                                    and content.count('-') < 5 and content.count('?') < 5 and content.count('+') < 5 and content.count('|') < 5 and content.count(':') < 5 and "''''" not in content:
                                if RSTFlag and EXNFlag:
                                    if content.count(RSTData[0]) < 4 and content.count(EXNData[0]) < 4:
                                        candidateList.append(content[:140])
                                else:
                                    candidateList.append(content[:140])
                if len(candidateList) > 0:
                    outputFile.write('\t'.join(candidateList)+'\n')
                else:
                    outputFile.write('<ERROR>\n')
    dataFile.close()
    itemFile.close()
    print('DONE')


def realizeSingleToken_GPT2(generatedFilename, itemFilename, outputFilename):
    dataFile = open(generatedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    with open(outputFilename, 'w') as outputFile:
        for dataLine, itemLine in zip(dataFile, itemFile):
            if '<ERROR>' in dataLine:
                outputFile.write('<ERROR>\n')
            else:
                itemData = json.loads(itemLine.strip())
                RSTData = itemData['<RST>']
                EXNData = itemData['<EXN>']
                RSTFlag = True if len(RSTData) > 0 else False
                EXNFlag = True if len(EXNData) > 0 else False
                contents = dataLine.strip().split('\t')
                candidateList = []
                for content in contents:
                    content = content.strip()
                    outWord = []
                    #content = content.replace('<R', ' <R').replace('<E', ' <E').replace('T>', 'T> ').replace('N>', 'N> ')
                    for index, word in enumerate(content.split(' ')):
                        if len(word) > 0:
                            if word.lower().startswith('<'):
                                if word.lower().startswith('<r') and RSTFlag:
                                    outWord.append(RSTData[0])
                                if word.lower().startswith('<e') and EXNFlag:
                                    outWord.append(EXNData[0])
                            else:
                                outWord.append(word)
                    content = ' '.join(outWord).replace('RST', '').replace('EXN', '')
                    if len(content) > 10 and len(content.split(' ')) > 3:
                        if RSTFlag and EXNFlag:
                            if content.count(RSTData[0]) < 4 and content.count(EXNData[0]) < 4:
                                candidateList.append(content[:140])
                        elif RSTFlag:
                            if content.count(RSTData[0]) < 4:
                                candidateList.append(content[:140])
                        elif EXNFlag:
                            if content.count(EXNData[0]) < 4:
                                candidateList.append(content[:140])
                if len(candidateList) > 0:
                    outputFile.write('\t'.join(candidateList)+'\n')
                else:
                    outputFile.write('<ERROR>\n')
    dataFile.close()
    itemFile.close()
    print('DONE')


def realizeMultipleToken_Copynet(generatedFilename, originalFilename, itemFilename, outputFilename):
    generateFile = open(generatedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    originalFile = open(originalFilename, 'r')
    lineIndex = 0
    with open(outputFilename, 'w') as outputFile:
        for generatedLine, itemLine, originalLine in zip(generateFile, itemFile, originalFile):
            if lineIndex % 1000 == 0:
                print(lineIndex)
            lineIndex += 1
            if '<ERROR>' in generatedLine:
                outputFile.write('<ERROR>\n')
            else:
                content = generatedLine.strip()
                itemData = json.loads(itemLine.strip())
                RSTData = itemData['<RST>']
                EXNData = itemData['<EXN>']
                if content.count('<RST>') == 0 and content.count('<EXN>') == 0:
                    out = content.replace('<UNK> ', '').replace(' <UNK>', '').replace('<UNK>', '')
                    if len(out) > 10 and len(out.split(' ')) > 3:
                        outputFile.write(out[:140]+'\n')
                    else:
                        outputFile.write('<ERROR>\n')
                else:
                    candidateList = []
                    permRSTTokens = permutations(RSTData)
                    permEXNTokens = permutations(EXNData)
                    for tempRSTData in permRSTTokens:
                        for tempEXNData in permEXNTokens:
                            outContent = applyTokens_copynet(content, tempRSTData, tempEXNData)
                            if len(outContent) > 10 and len(outContent.split(' ')) > 3:
                                candidateList.append(outContent[:140])
                    if len(candidateList) > 0:
                        candidateScores = computeSimilarity(originalLine.strip(), candidateList)
                        output = candidateList[candidateScores.index(max(candidateScores))]
                        outputFile.write(output + '\n')
                    else:
                        outputFile.write('<ERROR>\n')

    originalFile.close()
    generateFile.close()
    itemFile.close()
    print('DONE')


def realizeMultipleToken_GPT2(generatedFilename, originalFilename, itemFilename, outputFilename):
    generateFile = open(generatedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    originalFile = open(originalFilename, 'r')
    lineIndex = 0
    with open(outputFilename, 'w') as outputFile:
        for generatedLine, itemLine, originalLine in zip(generateFile, itemFile, originalFile):
            if lineIndex % 1000 == 0:
                print(lineIndex)
            lineIndex += 1
            if '<ERROR>' in generatedLine:
                outputFile.write('<ERROR>\n')
            else:
                itemData = json.loads(itemLine.strip())
                RSTData = itemData['<RST>']
                EXNData = itemData['<EXN>']
                candidateList = []
                for content in generatedLine.strip().split('\t'):
                    content = content.strip()
                    #content = content.replace('<R', ' <R').replace('<E', ' <E').replace('T>', 'T> ').replace('N>', 'N> ')
                    permRSTTokens = permutations(RSTData)
                    permEXNTokens = permutations(EXNData)
                    for tempRSTData in permRSTTokens:
                        for tempEXNData in permEXNTokens:
                            outContent = applyTokens_GPT2(content, tempRSTData, tempEXNData)
                            if len(outContent) > 10 and len(outContent.split(' ')) > 3:
                                candidateList.append(outContent[:140])
                if len(candidateList) > 0:
                    candidateScores = computeSimilarity(originalLine.strip(), candidateList)
                    output = candidateList[candidateScores.index(max(candidateScores))]
                    outputFile.write(output + '\n')
                else:
                    outputFile.write('<ERROR>\n')

    originalFile.close()
    generateFile.close()
    itemFile.close()
    print('DONE')


def selectParaphrase_similarity(realizedFilePath, originalFilePath, outputFilePath):
    generatedFile = open(realizedFilePath, 'r')
    originalFile = open(originalFilePath, 'r')
    with open(outputFilePath, 'w') as outputFile:
        lineIndex = 0
        for generatedLine, originalLine in zip(generatedFile, originalFile):
            if lineIndex % 2000 == 0:
                print(lineIndex)
            if '<ERROR>' in generatedLine:
                outputFile.write('<ERROR>\n')
            else:
                candidateList = generatedLine.strip().split('\t')
                candidateScores = computeSimilarity(originalLine.strip(), candidateList)
                output = candidateList[candidateScores.index(max(candidateScores))]
                outputFile.write(output+'\n')
            lineIndex += 1
    generatedFile.close()
    originalFile.close()
    print('DONE')


if __name__ == '__main__':
    tokenStyle = 'single'
    originalFilename = 'commTweets.content'
    generatedFilename = 'contained/commTweets.tokenized.only.gpt2'
    itemFilename = 'commTweets.item'

    originalFilePath = f'data/commTweet/test_{tokenStyle}/{originalFilename}'
    generatedFilePath = f'data/commTweet/test_{tokenStyle}/results/{generatedFilename}'
    itemFilePath = f'data/commTweet/test_{tokenStyle}/{itemFilename}'
    realizedFilePath = generatedFilePath+'.realized'

    #realizeSingleToken_GPT2(generatedFilePath, itemFilePath, realizedFilePath)
    #selectParaphrase_similarity(realizedFilePath, originalFilePath, realizedFilePath+'.selected')
    realizeMultipleToken_Copynet(generatedFilePath, itemFilePath, originalFilePath, realizedFilePath)
