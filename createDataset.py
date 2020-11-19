import json
import os
import random
from nltk.corpus import stopwords
from nltk.corpus import wordnet
all_stopwords = stopwords.words()

def createFromFolder(rootDir, outFilename):
    outputSet = set()
    for subdir, dirs, files in os.walk(rootDir):
        for file in files:
            filename = os.path.join(subdir, file)
            if (filename.endswith('.json')) and ('_' not in filename):
                with open(filename, 'r') as fr:
                    for line in fr:
                        data = json.loads(line.strip())
                        content = data['text'].replace('\n', ' ')
                        if content not in outputSet:
                            outputSet.add(content)

    with open(outFilename, 'w') as fo:
        for content in outputSet:
            fo.write(content+'\n')
    print('DONE')


def createParaphraseLM(filename, outputFilename):
    outputFile = open(outputFilename, 'w')
    with open(filename, 'r') as fr:
        for line in fr:
            outputFile.write(line.replace('\t', ' >>><<< '))
            #.replace('\n', ' <|endoftext|> '))
    outputFile.close()


def splitLargeFile(filename, split_num):
    data = []
    with open(filename, 'r') as fr:
        for line in fr:
            data.append(line)

    outputFileList = []
    for i in range(split_num):
        outputFileList.append(open(filename+'_'+str(i+1), 'w'))

    for index, line in enumerate(data):
        fileIndex = index % split_num
        outputFileList[fileIndex].write(line)

    for outputFile in outputFileList:
        outputFile.close()


def generateContentFiles(inputFilename, outputFilename, lenLimit, sampleRatio):
    data = []
    output = []
    outputFile = open(outputFilename, 'w')
    with open(inputFilename, 'r') as inputFile:
        for line in inputFile:
            temp = line.strip().split(' >>><<< ')
            if len(temp[0].split(' ')) > lenLimit and len(temp[1].split(' ')) > lenLimit:
                data.append(line.strip())

    output = random.sample(data, int(len(data)*sampleRatio))
    for item in output:
        temp = item.split(' >>><<< ')
        outputFile.write(temp[0] + '\n')
        outputFile.write(temp[1] + '\n')
    outputFile.close()


def replaceSyn(inputWord, maxNum=5):
    syns = wordnet.synsets(inputWord)
    maxNum = maxNum if len(syns) > maxNum else len(syns)
    for i, syn in enumerate(syns):
        if i > maxNum:
            break
        output = syn.lemmas()[0].name()
        if output != inputWord:
            return output
    return None


def createReconstructedParaData(inputFilename, outFilename):
    outputList = []
    with open(inputFilename, 'r') as fr:
        for lineNum, line in enumerate(fr):
            if lineNum % 1000 == 0:
                print(lineNum)
            outTokenList = []
            tokenList = line.strip().split(' ')
            for token in tokenList:
                if 'http' not in token:
                    if token not in all_stopwords:
                        outTokenList.append(token)
            randomIndexList = random.sample(range(0, len(outTokenList)), round(0.2*len(outTokenList)))
            tempTokenLists = []
            for i in randomIndexList:
                tempTokenLists.append(outTokenList[i])
            shuffledRandomIndexList = random.sample(randomIndexList, len(randomIndexList))
            for index, shuffledListIndex in enumerate(shuffledRandomIndexList):
                outTokenList[shuffledListIndex] = tempTokenLists[index]
            synReplaceCount = 0
            while synReplaceCount < 0.2*len(outTokenList):
                index = random.randint(0, len(outTokenList)-1)
                token = outTokenList[index]
                synToken = replaceSyn(token, maxNum=5)
                if synToken:
                    outTokenList[index] = synToken
                    synReplaceCount += 1
            newContent = (' ').join(outTokenList)
            outputList.append((line.strip(), newContent.strip()))

    with open(outFilename, 'w') as fo:
        for pair in outputList:
            fo.write(pair[0] + ' >>><<< ' + pair[1] + '\n')

    print('DONE')



def combineCommTweets(inputFolder, outputFilename):
    idSet = set()
    outputFile = open(outputFilename, 'w')
    for folder in os.listdir(inputFolder):
        print(folder)
        for filename in os.listdir(inputFolder+'/'+folder):
            if filename.endswith('.json') and '_' not in filename:
                with open(inputFolder+'/'+folder+'/'+filename, 'r') as fr:
                    for line in fr:
                        data = json.loads(line.strip())
                        tweetID = data['id']
                        if tweetID not in idSet and data['lang'] == 'en':
                            idSet.add(tweetID)
                            hashtags = []
                            urls = []
                            mentions = []
                            if 'entities' in data:
                                if 'hashtags' in data['entities']:
                                    for hashtag in data['entities']['hashtags']:
                                        hashtags.append(hashtag['text'])
                                if 'urls' in data['entities']:
                                    for url in data['entities']['urls']:
                                        urls.append(url['url'])
                                if 'user_mentions' in data['entities']:
                                    for mention in data['entities']['user_mentions']:
                                        mentions.append(mention['screen_name'])
                            text = data['text']
                            retweet = data['retweet_count']
                            favorite = data['favorite_count']
                            followers = data['user']['followers_count']
                            userCreateTime = data['user']['created_at']
                            source = data['source']
                            createTime = data['created_at']
                            user_listed_count = data['user']['listed_count']
                            user_favorite_count = data['user']['favourites_count']
                            user_statuses_count = data['user']['statuses_count']
                            user_friends_count = data['user']['friends_count']
                            dataStructure = {'id': tweetID, 'text': text, 'brand': folder, 'hashtags': hashtags, 'urls': urls, 'mentions': mentions, 'source': source, 'retweet_count': retweet, 'favorite_count': favorite,
                                             'create_at': createTime, 'user_create_at': userCreateTime,
                                             'user_followers_count': followers, 'user_listed_count': user_listed_count, 'user_friends_count': user_friends_count, 'user_statuses_count': user_statuses_count, 'user_favorite_count': user_favorite_count}
                            outputFile.write(json.dumps(dataStructure)+'\n')
    outputFile.close()


if __name__ == '__main__':
    # createFromFolder('adData', 'commercialTweets.list')
    createParaphraseLM('data/pmt/pmt.full', 'data/pmt/pmt.full.line.lm')
    #splitLargeFile('data/pmt.full.line.lm', 2)
    # generateContentFiles('data/pmt/pmt.full.line.lm', 'data/pmt/pmt.filtered.content', 10, 1)
    # combineCommTweets('/Volumes/DATA/Data/adData', 'data/commTweets.json')
    # createReconstructedParaData('data/commercialTweets.list', 'data/commercialTweets.paraphrase')
