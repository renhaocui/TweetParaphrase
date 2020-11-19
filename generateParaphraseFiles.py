import json
import re
import random

def checkTokenCount(generatedFilename, itemFilename):
    generatedFile = open(generatedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    RSTCount = 0
    EXNCount = 0
    trueRSTCount = 0
    trueEXNCount = 0
    RSTDiffCount = 0
    EXNDiffCount = 0
    totalCount = 0
    for generatedLine, itemLine in zip(generatedFile, itemFile):
        itemData = json.loads(itemLine.strip())
        contents = generatedLine.strip().split('\t')
        for content in contents:
            totalCount += 1
            RSTLineCount = content.lower().count('<r')
            EXNLineCount = content.lower().count('<e')
            RSTCount += RSTLineCount
            EXNCount += EXNLineCount
            trueRSTCount += len(itemData['<RST>'])
            trueEXNCount += len(itemData['<EXN>'])
            RSTDiffCount += (RSTLineCount - len(itemData['<RST>']))
            EXNDiffCount += (EXNLineCount - len(itemData['<EXN>']))
    generatedFile.close()
    itemFile.close()

    print(RSTCount / totalCount)
    print(EXNCount / totalCount)
    print(RSTDiffCount / totalCount)
    print(EXNDiffCount / totalCount)
    print(trueRSTCount / totalCount)
    print(trueEXNCount / totalCount)

    print('DONE')


def checkTokenCount2(generatedFilename, itemFilename):
    RSTCount = 0
    EXNCount = 0
    totalCount = 0
    generatedFile = open(generatedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    for generatedLine, itemLine in zip(generatedFile, itemFile):
        if '<ERROR>' not in generatedLine:
            totalCount += 1
            content = generatedLine.strip()
            itemData = json.loads(itemLine.strip())
            if len(itemData['<RST>']) > 0:
                RSTCount += content.count(itemData['<RST>'][0])
            if len(itemData['<EXN>']) > 0:
                EXNCount += content.count(itemData['<EXN>'][0])
    itemFile.close()
    generatedFile.close()
    print(RSTCount / totalCount)
    print(EXNCount / totalCount)

    print('DONE')




def completeOriginalCommTweets(dataFilename, tokenFilename, outputFilename, repeatTimes=1):
    dataFile = open(dataFilename, 'r')
    tokenFile = open(tokenFilename, 'r')
    with open(outputFilename, 'w') as outputFile:
        for dataLine, tokenLine in zip(dataFile, tokenFile):
            tokenData = json.loads(tokenLine.strip())
            for token, value in tokenData.items():
                text = dataLine.strip().replace(token, value)
            for j in range(repeatTimes):
                outputFile.write(text+'\n')

    dataFile.close()
    tokenFile.close()
    print('DONE')


def completeCopynetCommTweets_singleToken(dataFilename, itemFilename, outputFilename):
    dataFile = open(dataFilename, 'r')
    itemFile = open(itemFilename, 'r')
    with open(outputFilename, 'w') as outputFile:
        for dataLine, itemLine in zip(dataFile, itemFile):
            itemData = json.loads(itemLine.strip())
            lineData = dataLine.strip().split('\t')
            for content in lineData:
                outWord = []
                for word in content.split(' '):
                    if word in itemData and len(itemData[word]) > 0:
                        outWord.append(itemData[word][0])
                    else:
                        outWord.append(word)
                outputFile.write(' '.join(outWord)+'\n')

    dataFile.close()
    itemFile.close()
    print('DONE')


def completeCopynetCommTweets_multipleToken(dataFilename, itemFilename, outputFilename, shuffle=False):
    dataFile = open(dataFilename, 'r')
    itemFile = open(itemFilename, 'r')
    with open(outputFilename, 'w') as outputFile:
        for dataLine, itemLine in zip(dataFile, itemFile):
            itemData = json.loads(itemLine.strip())
            RSTData = itemData['<RST>']
            EXNData = itemData['<EXN>']
            if shuffle:
                random.shuffle(RSTData)
                random.shuffle(EXNData)
            content = dataLine.strip()
            if content.count('<RST>') <= len(RSTData) and content.count('<EXN>') <= len(EXNData):
                RSTIndex = 0
                EXNIndex = 0
                outWord = []
                for word in content.split(' '):
                    if word == '<RST>':
                        index = min([RSTIndex, len(RSTData)-1])
                        outWord.append(RSTData[index])
                        RSTIndex += 1
                    elif word == '<EXN>':
                        index = min([EXNIndex, len(EXNData)-1])
                        outWord.append(EXNData[index])
                        EXNIndex += 1
                    elif '<UNK>' not in word:
                        outWord.append(word)
                outputFile.write(' '.join(outWord)+'\n')
            else:
                outputFile.write('<ERROR>\n')

    dataFile.close()
    itemFile.close()
    print('DONE')


def completeGPT2CommTweets_singleToken(generatedFilename, itemFilename, outputFilename):
    dataFile = open(generatedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    with open(outputFilename, 'w') as outputFile:
        lineIndex = 0
        for dataLine, itemLine in zip(dataFile, itemFile):
            if '<ERROR>' in dataLine:
                outputFile.write('<ERROR>\n')
            else:
                itemData = json.loads(itemLine.strip())
                RSTFlag = True if len(itemData['<RST>']) > 0 else False
                EXNFlag = True if len(itemData['<EXN>']) > 0 else False
                candidates = []
                contents = dataLine.strip().split('\t')
                for content in contents:
                    RSTValid = False
                    EXNValid = False
                    content = content.strip()
                    outWord = []
                    content = content.replace('<R', ' <R').replace('<E', ' <E').replace('T>', 'T> ').replace('N>', 'N> ')
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
                                    outWord.append(itemData['<RST>'][0])
                                if word.lower().startswith('<e') and EXNFlag:
                                    outWord.append(itemData['<EXN>'][0])
                            else:
                                outWord.append(word)
                    content = ' '.join(outWord).replace('RST', '').replace('EXN', '')
                    if RSTValid and EXNValid:
                        if len(content) > 10 and len(content.split(' ')) > 3:
                            if content.count('*') < 5 and content.count('/') < 5 and content.count('#') < 5 and content.count('(') < 5 and content.count('.') < 7 and content.count(',') < 7 \
                                    and content.count('%') < 5 and content.count('@') < 5 and content.count('$') < 5 and content.count('&') < 5 and content.count('\\') < 5 \
                                    and content.count('-') < 5 and content.count('?') < 5 and content.count('+') < 5 and content.count('|') < 5 and content.count(':') < 5 and "''''" not in content:
                                if RSTFlag and EXNFlag:
                                    if content.count(itemData['<RST>'][0]) < 3 and content.count(itemData['<EXN>'][0]) < 3:
                                        if len(content) < 150:
                                            candidates.append(content[:140])
                                else:
                                    if len(content) < 150:
                                        candidates.append(content[:140])
                if len(candidates) == 1:
                    outputFile.write(candidates[0] + '\n')
                elif len(candidates) == 0:
                    outputFile.write('<ERROR>\n')
                else:
                    outputFile.write(candidates[0] + '\n')
            lineIndex += 1

    dataFile.close()
    itemFile.close()
    print('DONE')


def completeGPT2CommTweets_multipleToken(generatedFilename, itemFilename, outputFilename, shuffle=False):
    dataFile = open(generatedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    with open(outputFilename, 'w') as outputFile:
        lineIndex = 0
        for dataLine, itemLine in zip(dataFile, itemFile):
            if '<ERROR>' in dataLine:
                outputFile.write('<ERROR>\n')
            else:
                itemData = json.loads(itemLine.strip())
                RSTData = itemData['<RST>']
                EXNData = itemData['<EXN>']
                if shuffle:
                    random.shuffle(RSTData)
                    random.shuffle(EXNData)
                RSTFlag = True if len(RSTData) > 0 else False
                EXNFlag = True if len(EXNData) > 0 else False
                candidates = []
                contents = dataLine.strip().split('\t')
                for content in contents:
                    content = content.strip()
                    RSTValid = False
                    EXNValid = False
                    if content.lower().count('<r') <= len(RSTData) + 1 and content.lower().count('<e') <= len(EXNData) + 1:
                        content = content.replace('<R', ' <R').replace('<E', ' <E').replace('T>', 'T> ').replace('N>', 'N> ')
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
                        RSTIndex = 0
                        EXNIndex = 0
                        outWord = []
                        for index, word in enumerate(content.split(' ')):
                            if len(word) > 0:
                                if word.lower().startswith('<'):
                                    if word.lower().startswith('<r') and RSTFlag:
                                        index = min([RSTIndex, len(RSTData) - 1])
                                        outWord.append(RSTData[index])
                                        RSTIndex += 1
                                    if word.lower().startswith('<e') and EXNFlag:
                                        index = min([EXNIndex, len(EXNData) - 1])
                                        outWord.append(EXNData[index])
                                        EXNIndex += 1
                                else:
                                    outWord.append(word)
                        content = ' '.join(outWord).replace('RST', '').replace('EXN', '')
                        if RSTValid and EXNValid:
                            if len(content) > 10 and len(content.split(' ')) > 3:
                                if content.count('*') < 5 and content.count('/') < 5 and content.count('#') < 5 and content.count('(') < 5 and content.count('.') < 7 and content.count(',') < 7 \
                                        and content.count('%') < 5 and content.count('@') < 5 and content.count('$') < 5 and content.count('&') < 5 and content.count('\\') < 5 \
                                        and content.count('-') < 5 and content.count('?') < 5 and content.count('+') < 5 and content.count('|') < 5 and content.count(':') < 5 and "''''" not in content:
                                    if len(content) < 150:
                                        candidates.append(content[:140])

                if len(candidates) == 1:
                    outputFile.write(candidates[0] + '\n')
                elif len(candidates) == 0:
                    outputFile.write('<ERROR>\n')
                else:
                    outputFile.write(candidates[0] + '\n')
            lineIndex += 1

    dataFile.close()
    itemFile.close()
    print('DONE')



if __name__ == '__main__':
    tokenStyle = 'single'
    originalFilename = 'commTweets.content'
    generatedFilename = 'contained/commTweets.contained.gpt2'
    itemFilename = 'commTweets.item'

    originalFilePath = f'data/commTweet/test_{tokenStyle}/{originalFilename}'
    generatedFilePath = f'data/commTweet/test_{tokenStyle}/results/{generatedFilename}'
    itemFilePath = f'data/commTweet/test_{tokenStyle}/{itemFilename}'

    #checkTokenCount(generatedFilePath, itemFilePath)
    #checkTokenCount2(generatedFilePath, itemFilePath)

    #completeOriginalComm('data/commTweet/commTweets.NNP.tokenized.sampled',
    #                       'data/commTweet/commTweets.NNP.tokenized.sampled.items',
    #                       'data/commTweet/commTweets.NNP.tokenized.sampled.original', repeatTimes=1)

    #completeCopynetCommTweets_singleToken('data/commTweet/test_single/results/commTweets.1.tokenized.only.copynet',
    #                               'data/commTweet/test_single/commTweets.item_1.item',
    #                               'data/commTweet/test_single/reports/commTweets.only.copynet')

    #completeCopynetCommTweets_multipleToken('data/commTweet/test_multiple/results/copynet/commTweets.tokenized.only.copynet',
    #                               'data/commTweet/test_multiple/commTweets.item',
    #                               'data/commTweet/test_multiple/results/copynet/commTweets.only.copynet.shuffled', shuffle=True)

    type = 'only'
    completeGPT2CommTweets_singleToken(f'data/commTweet/test_single/results/{type}/commTweets.tokenized.{type}.gpt2',
                                          f'data/commTweet/test_single/commTweets.item',
                                          f'data/commTweet/test_single/results/{type}/commTweets.{type}.gpt2')

    #completeGPT2CommTweets_multipleToken(f'data/commTweet/test_multiple/results/{type}/commTweets.tokenized.{type}.gpt2',
    #                                      f'data/commTweet/test_multiple/commTweets.item',
    #                                      f'data/commTweet/test_multiple/results/{type}/commTweets.{type}.gpt2.shuffled', shuffle=True)
