import json


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


def completeParaphraseCommTweets_singleToken(dataFilename, tokenFilename, outputFilename, paraTokenList=['NPT', 'NN']):
    dataFile = open(dataFilename, 'r')
    tokenFile = open(tokenFilename, 'r')
    with open(outputFilename, 'w') as outputFile:
        lineIndex = 0
        for dataLine, tokenLine in zip(dataFile, tokenFile):
            lineIndex += 1
            tokenData = json.loads(tokenLine.strip())
            for token, value in tokenData.items():
                lineData = dataLine.strip().split('\t')
                if len(lineData) != 5:
                    print(lineIndex)
                for content in lineData:
                    outWord = []
                    for word in content.split(' '):
                        tokenFound = False
                        for paraToken in paraTokenList:
                            if paraToken in word:
                                tokenFound = True
                                break
                        if tokenFound:
                            outWord.append(value)
                        else:
                            outWord.append(word)
                    outputFile.write(' '.join(outWord)+'\n')

    dataFile.close()
    tokenFile.close()
    print('DONE')


if __name__ == '__main__':
    #completeOriginalComm('data/commTweet/commTweets.NNP.tokenized.sampled',
    #                       'data/commTweet/commTweets.NNP.tokenized.sampled.items',
    #                       'data/commTweet/commTweets.NNP.tokenized.sampled.original', repeatTimes=1)
    completeParaphraseCommTweets_singleToken('data/commTweet/commTweets.NNP.tokenized.sampled.copynet',
                                   'data/commTweet/commTweets.NNP.tokenized.sampled.items',
                                   'data/commTweet/commTweets.NNP.tokenized.sampled.copynet.paraphrase')
