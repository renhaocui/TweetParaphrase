import re
import json

def refineCopyNet(inputFilename, outputFilename):
    outputFile = open(outputFilename, 'w')
    with open(inputFilename, 'r') as inputFile:
        for line in inputFile:
            outTokens = []
            previousToken = None
            repeatCount = 0
            for token in line.strip().split(' '):
                if token.lower() != previousToken:
                    outTokens.append(token)
                    previousToken = token.lower()
                    repeatCount = 0
                else:
                    repeatCount += 1
                    if repeatCount < 1:
                        outTokens.append(token)
            outputFile.write(' '.join(outTokens)+'\n')
    outputFile.close()

    return True


def selectCopyNet(generatedFilname, outputFilename):
    outputFile = open(outputFilename, 'w')
    count = 0
    with open(generatedFilname, 'r') as generatedFile:
        for line in generatedFile:
            line = line.replace('<UNK> ', '').replace('<UNK>', '').strip()
            if len(line) > 10 and len(line.split(' ')) > 3:
                outputFile.write(line[:140]+'\n')
            else:
                count += 1
                outputFile.write('<ERROR>\n')
    print(count)
    outputFile.close()


def selectGPT2(generatedFilename, outputFilename):
    outputFile = open(outputFilename, 'w')
    generatedFile = open(generatedFilename, 'r')
    for lineIndex, generatedLine in enumerate(generatedFile):
        candidates = []
        if '<ERROR>' in generatedLine:
            outputFile.write('<ERROR>\n')
        else:
            for content in generatedLine.strip().split('\t'):
                content = content.strip()
                if len(content) > 10 and len(content.split(' ')) > 3:
                    #if content.count('*') < 5 and content.count('/') < 5 and content.count('#') < 5 and content.count('(') < 5 and content.count('.') < 7 and content.count(',') < 7 \
                    #        and content.count('%') < 5 and content.count('@') < 5 and content.count('$') < 5 and content.count('&') < 5 and content.count('\\') < 5 \
                    #        and content.count('-') < 5 and content.count('?') < 5 and content.count('+') < 5 and content.count('|') < 5 and content.count(':') < 5 and "''''" not in content:
                    candidates.append(content[:140])
            if len(candidates) > 0:
                outputFile.write('\t'.join(candidates)+'\n')
            else:
                outputFile.write('<ERROR>\n')

    generatedFile.close()
    outputFile.close()
    print('DONE')


def dataSampler(dataFilename, inspectToken):
    with open(dataFilename, 'r') as dataFile:
        for index, line in enumerate(dataFile):
            if line.count(inspectToken) > 1:
                print(index, line.strip())



if __name__ == '__main__':
    #refineCopyNet('data/commTweet/test_30k/commTweets.general.copynet', 'data/commTweet/test_30k/commTweets.general.copynet.refined')
    #selectGPT2('data/commTweet/test_single/results/full/commTweets.content.gpt2', 'data/commTweet/test_single/results/full/commTweets.full.gpt2')
    #selectGPT2('data/commTweet/test_multiple/results/full/commTweets.content.gpt2', 'data/commTweet/test_multiple/results/full/commTweets.full.gpt2')
    #selectCopyNet('data/commTweet/test_multiple/results/copynet/commTweets.contained.copynet.shuffled', 'data/commTweet/test_multiple/results/copynet/commTweets.contained.copynet.shuffled.selected')
    dataSampler('data/commTweet/test_multiple/commTweets.tokenized', '<RST>')
