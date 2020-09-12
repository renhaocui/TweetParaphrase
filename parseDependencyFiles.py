import json
import random
NNPToken = ['NNPTK', 'NNPTK1', 'NNPTK2', 'NNPTK3', 'NNPTK4', 'NNPTK5', 'NNPTK6', 'NNPTK7', 'NNPTK8', 'NNPTK9']


def parseDependencyFile_dual(inputFilenameList, outputFilename, targetTokenList=['NNP', 'NNPS']):
    outputFile = open(outputFilename, 'w')
    tokenCount = {}
    for inputFilename in inputFilenameList:
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

                if srcCandidateTokens:
                    if set(srcCandidateTokens) == set(trgCandidateTokens):
                        size = len(set(srcCandidateTokens))
                        if size not in tokenCount:
                            tokenCount[size] = 1
                        else:
                            tokenCount[size] += 1
                        if size <= len(NNPToken):
                            for index, token in enumerate(set(srcCandidateTokens)):
                                content = content.replace(token, NNPToken[index])
                            outputFile.write(content + '\n')
    outputFile.close()
    print(tokenCount)


def parseDependencyFile_single(inputFilename, outputFilename, tokenFile, tokenLimit=1, sampleSize=None, primary=False):
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


if __name__ == '__main__':
    # parseDependencyFile_dual(['drive/My Drive/Cui_WorkSpace/Data/TweetParaphrase/PMT/pmt.line.dependency'],
    #                   'drive/My Drive/Cui_WorkSpace/Data/TweetParaphrase/PMT/pmt.NNP.tokenized.full')

    parseDependencyFile_single('drive/My Drive/Cui_WorkSpace/Data/TweetParaphrase/commTweets/commTweets.full.dependency',
                       'drive/My Drive/Cui_WorkSpace/Data/TweetParaphrase/commTweets/commTweets.NNP.tokenized.sampled',
                       'drive/My Drive/Cui_WorkSpace/Data/TweetParaphrase/commTweets/commTweets.NNp.tokenized.sampled.items', tokenLimit=1, sampleSize=1000)

