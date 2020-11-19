import json
import random
NNPToken = ['<RST1>', '<RST2>', '<RST3>', '<RST4>', '<RST5>']
NUMToken = []


def tokenMapper(POSType, relation):
    if POSType == 'NNP':
        if relation in ['ROOT']:
            return 'NNPR'
        elif relation in ['pobj', 'obj', 'dobj', 'oprd']:
            return 'NNPO'
        elif relation in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']:
            return 'NNPS'
        else:
            return None
    elif POSType == 'NUM':
        return 'NUM'


def validateRelation(POSType, relation):
    if POSType == 'NNP':
        if relation in ['ROOT', 'pobj', 'obj', 'dobj', 'oprd', 'nsubj', 'nsubjpass', 'csubj', 'csubjpass']:
            return True
    elif POSType == 'NUM':
        if relation == 'nummod':
            return True
    return False


def parseDependencyFile_train_new(inputFilename, outputFilename, contentFilename, contained=False, NNPPOSList=['NNP','NNPS'], NUMPOSList=['CD', '$']):
    outputFile = open(outputFilename, 'w')
    contentFile = open(contentFilename, 'w')
    with open(inputFilename, 'r') as inputFile:
        for lineIndex, line in enumerate(inputFile):
            data = json.loads(line.strip())
            if lineIndex % 2 == 0:
                # source sentence
                caseEnd = False
                srcWordList = []
                srcTokenList = []
                srcContent = ''
                previousToken = None
                for item in data:
                    word = item[0]
                    pos = item[1]
                    relation = item[3]
                    srcContent += word + ' '
                    if pos in NNPPOSList:
                        if relation == 'compound':
                            if previousToken:
                                previousToken += ' ' + word
                            else:
                                previousToken = word
                        else:
                            if previousToken:
                                wordToken = previousToken + ' ' + word
                            else:
                                wordToken = word
                            if validateRelation('NNP', relation):
                                srcWordList.append(wordToken)
                                srcTokenList.append('<RST>')
                            previousToken = None
                    elif pos in NUMPOSList:
                        if relation == 'compound':
                            if previousToken:
                                previousToken += ' ' + word
                            else:
                                previousToken = word
                        else:
                            if previousToken:
                                wordToken = previousToken + ' ' + word
                            else:
                                wordToken = word
                            if validateRelation('NUM', relation):
                                srcWordList.append(wordToken)
                                srcTokenList.append('<EXN>')
                            previousToken = None
            else:
                # target sentence
                caseEnd = True
                trgWordList = []
                trgTokenList = []
                trgContent = ''
                previousToken = None
                for item in data:
                    word = item[0]
                    pos = item[1]
                    relation = item[3]
                    trgContent += word + ' '
                    if pos in NNPPOSList:
                        if relation == 'compound':
                            if previousToken:
                                previousToken += ' ' + word
                            else:
                                previousToken = word
                        else:
                            if previousToken:
                                wordToken = previousToken + ' ' + word
                            else:
                                wordToken = word
                            if validateRelation('NNP', relation):
                                trgWordList.append(wordToken)
                                trgTokenList.append('<RST>')
                            previousToken = None
                    elif pos in NUMPOSList:
                        if relation == 'compound':
                            if previousToken:
                                previousToken += ' ' + word
                            else:
                                previousToken = word
                        else:
                            if previousToken:
                                wordToken = previousToken + ' ' + word
                            else:
                                wordToken = word
                            if validateRelation('NUM', relation):
                                trgWordList.append(wordToken)
                                trgTokenList.append('<EXN>')
                            previousToken = None

            if caseEnd:
                contentFile.write(srcContent.strip() + ' >>><<< ' + trgContent.strip() + '\n')
                if len(srcWordList) > 0:
                    processedWordList = []
                    for index, word in enumerate(srcWordList):
                        if word in trgWordList:
                            if word not in processedWordList:
                                output = srcContent.strip() + ' >>><<< ' + trgContent.strip()
                                token = srcTokenList[index]
                                output = output.replace(word, token)
                                outputFile.write(output + '\n')
                                processedWordList.append(word)
                else:
                    if contained:
                        outputFile.write(srcContent.strip() + ' >>><<< ' + trgContent.strip() + '\n')

    outputFile.close()
    contentFile.close()


def parseDependencyFile_infer_new(inputFilename, singleOutputFilename, multipleOutputFilename, singleContentFilename, multipleContentFilename, singleItemFilename, multipleItemFilename, NNPPOSList=['NNP','NNPS'], NUMPOSList=['CD', '$']):
    singleOutputFile = open(singleOutputFilename, 'w')
    multipleOutputFile = open(multipleOutputFilename, 'w')
    singleContentFile = open(singleContentFilename, 'w')
    multipleContentFile = open(multipleContentFilename, 'w')
    singleItemFile = open(singleItemFilename, 'w')
    multipleItemFile = open(multipleItemFilename, 'w')

    with open(inputFilename, 'r') as inputFile:
        for lineIndex, line in enumerate(inputFile):
            data = json.loads(line.strip())
            NNPWordSet = set()
            NUMWordSet = set()
            wordList = []
            tokenList = []
            content = ''
            previousToken = None
            for item in data:
                word = item[0]
                pos = item[1]
                relation = item[3]
                content += word + ' '
                if pos in NNPPOSList:
                    if relation == 'compound':
                        if previousToken:
                            previousToken += ' ' + word
                        else:
                            previousToken = word
                    else:
                        if previousToken:
                            wordToken = previousToken + ' ' + word
                        else:
                            wordToken = word
                        if validateRelation('NNP', relation):
                            if wordToken not in NNPWordSet:
                                NNPWordSet.add(wordToken)
                            wordList.append(wordToken)
                            tokenList.append('<RST>')
                        previousToken = None
                elif pos in NUMPOSList:
                    if relation == 'compound':
                        if previousToken:
                            previousToken += ' ' + word
                        else:
                            previousToken = word
                    else:
                        if previousToken:
                            wordToken = previousToken + ' ' + word
                        else:
                            wordToken = word
                        if validateRelation('NUM', relation):
                            if wordToken not in NUMWordSet:
                                NUMWordSet.add(wordToken)
                            wordList.append(wordToken)
                            tokenList.append('<EXN>')
                        previousToken = None

            if len(wordList) > 0:
                itemDict = {'<RST>': list(NNPWordSet), '<EXN>': list(NUMWordSet)}
                processedWordList = []
                output = content.strip()
                for index, word in enumerate(wordList):
                    if word not in processedWordList:
                        token = tokenList[index]
                        output = output.replace(word, token)
                        processedWordList.append(word)
                if len(NNPWordSet) > 1 or len(NUMWordSet) > 1:
                    multipleOutputFile.write(output + '\n')
                    multipleContentFile.write(content.strip() + '\n')
                    multipleItemFile.write(json.dumps(itemDict) + '\n')
                else:
                    singleOutputFile.write(output + '\n')
                    singleContentFile.write(content.strip() + '\n')
                    singleItemFile.write(json.dumps(itemDict) + '\n')

    singleOutputFile.close()
    multipleOutputFile.close()
    singleContentFile.close()
    multipleContentFile.close()
    singleItemFile.close()
    multipleItemFile.close()



def parseDependencyFile_train(inputFilename, outputFilename, contained=False, targetTokenList=['NNP', 'NNPS']):
    outputFile = open(outputFilename, 'w')
    tokenCount = {}
    with open(inputFilename, 'r') as inputFile:
        for lineIndex, line in enumerate(inputFile):
            data = json.loads(line.strip())
            if lineIndex % 2 == 0:
                # source sentence
                content = ''
                srcCandidateTokens = []
                trgCandidateTokens = []
                previousToken = None
                for item in data:
                    content += item[0] + ' '
                    if item[1] in targetTokenList:
                        if previousToken:
                            if previousToken[2] == item[0]:
                                srcCandidateTokens[-1] += ' ' + item[0]
                            else:
                                srcCandidateTokens.append(item[0])
                            previousToken = item
                        else:
                            srcCandidateTokens.append(item[0])
                            previousToken = item
                    else:
                        previousToken = None
                content = content.strip() + ' >>><<< '
            else:
                # target sentence
                previousToken = None
                for item in data:
                    content += item[0] + ' '
                    if item[1] in targetTokenList:
                        if previousToken:
                            # if previousToken[2] == item[0]:
                            trgCandidateTokens[-1] += ' ' + item[0]
                            # else:
                            #    trgCandidateTokens.append(item[0])
                            previousToken = item
                        else:
                            trgCandidateTokens.append(item[0])
                            previousToken = item
                    else:
                        previousToken = None
                content = content.strip()

            if set(srcCandidateTokens) == set(trgCandidateTokens):
                size = len(set(srcCandidateTokens))
                if size not in tokenCount:
                    tokenCount[size] = 1
                else:
                    tokenCount[size] += 1
                if size > 0:
                    for index, token in enumerate(set(srcCandidateTokens)):
                        content = content.replace(token, NNPToken[index])
                    outputFile.write(content + '\n')
                else:
                    if contained and not content.endswith(' >>><<< '):
                        outputFile.write(content + '\n')

    outputFile.close()
    print(tokenCount)


def parseDependencyFile_infer(inputFilename, outputFilename, tokenFile, tokenLimit=1, sampleSize=None):
    outputFile = open(outputFilename, 'w')
    tokenFile = open(tokenFile, 'w')
    totalData = []
    totalToken = []
    tokenCount = {}
    with open(inputFilename, 'r') as inputFile:
        for line in inputFile:
            tokenData = {}
            data = json.loads(line.strip())
            content = ''
            tokens = []
            previousToken = None
            for item in data:
                content += item[0] + ' '
                if item[1] in ['NNP', 'NNPS']:
                    if previousToken:
                        # if previousToken[2] == item[0]:
                        tokens[-1] += ' ' + item[0]
                        # else:
                        #    tokens.append(item[0])
                        previousToken = item
                    else:
                        tokens.append(item[0])
                        previousToken = item
                else:
                    previousToken = None
            content = content.strip()

            if tokens:
                size = len(set(tokens))
                if size not in tokenCount:
                    tokenCount[size] = 1
                else:
                    tokenCount[size] += 1
                if size <= len(NNPToken) and size <= tokenLimit:
                    for index, token in enumerate(set(tokens)):
                        tokenData[NNPToken[index]] = token
                        totalToken.append(tokenData)
                        content = content.replace(token, NNPToken[index])
                        totalData.append(content)

    if sampleSize:
        sampledIndices = random.sample(list(range(len(totalData))), sampleSize)
    else:
        sampledIndices = list(range(len(totalData)))

    for index, content in enumerate(totalData):
        if index in sampledIndices:
            outputFile.write(content.replace('\n', ' ').replace('\r', ' ') + '\n')
            tokenFile.write(json.dumps(totalToken[index]) + '\n')

    print(tokenCount)
    print('DONE')
    outputFile.close()
    tokenFile.close()


def cleanFiles(contentFilename, tokenizedFilename, itemFilename):
    contentFile = open(contentFilename, 'r')
    tokenizedFile = open(tokenizedFilename, 'r')
    itemFile = open(itemFilename, 'r')
    contentFile_new = open(contentFilename+'.new', 'w')
    tokenizedFile_new = open(tokenizedFilename+'.new', 'w')
    itemFile_new = open(itemFilename+'.new', 'w')
    for contentLine, tokenizedLine, itemLine in zip(contentFile, tokenizedFile, itemFile):
        if '<RST> …' not in tokenizedLine and '\\u' not in itemLine and '\\' not in itemLine and '-' not in itemLine and not contentLine.strip().endswith(' …'):
            contentFile_new.write(contentLine)
            tokenizedFile_new.write(tokenizedLine)
            itemFile_new.write(itemLine)
    contentFile.close()
    tokenizedFile.close()
    itemFile.close()
    contentFile_new.close()
    tokenizedFile_new.close()
    itemFile_new.close()



if __name__ == '__main__':
    #parseDependencyFile_train(['drive/My Drive/Cui_WorkSpace/Data/TweetParaphrase/PMT/pmt.full.line.dependency'],
    #                   'drive/My Drive/Cui_WorkSpace/Data/TweetParaphrase/PMT/pmt.full.NNP.tokenized')

    #parseDependencyFile_single('data/commTweet/commTweets.full.dependency',
    #                   'data/commTweet/test_30k/commTweets.NNP.tokenized',
    #                   'data/commTweet/test_30k/commTweets.NNP.items', tokenLimit=1, sampleSize=30000)

    parseDependencyFile_train_new('data/pmt/pmt.full.line.dependency',
                                  'data/pmt/newToken/pmt.full.NNP.tokenized.contained',
                                  'data/pmt/newToken/pmt.full.NNP.content',
                                  contained=True, NNPPOSList=['NNP', 'NNPS'], NUMPOSList=['CD', '$'])

    #parseDependencyFile_infer_new('data/commTweet/commTweets.full.dependency',
    #                              'data/commTweet/test/commTweets.tokenized.single',
    #                              'data/commTweet/test/commTweets.tokenized.multiple',
    #                              'data/commTweet/test/commTweets.content.single',
    #                              'data/commTweet/test/commTweets.content.multiple',
    #                              'data/commTweet/test/commTweets.tokenized.item.single',
    #                              'data/commTweet/test/commTweets.tokenized.item.multiple',
    #                              NNPPOSList=['NNP', 'NNPS'], NUMPOSList=['CD', '$'])

    #cleanFiles('data/commTweet/test/commTweets.content.multiple',
    #           'data/commTweet/test/commTweets.tokenized.multiple',
    #           'data/commTweet/test/commTweets.tokenized.item.multiple')
