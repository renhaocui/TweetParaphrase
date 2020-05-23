import json
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def showReplies(brandList, outputFile):
    outList = []
    for brand in brandList:
        for filename in os.listdir('replyData/'+brand):
            if filename.endswith('json'):
                inputFile = open('replyData/' + brand + '/' + filename, 'r')
                for line in inputFile:
                    try:
                        data = json.loads(line.strip())
                    except:
                        continue
                    replyList = []
                    oriTweet = data['text']
                    if len(data['replies']) > 1:
                        for reply in data['replies']:
                            replyList.append(reply['text'].replace('@'+brand, ''))
                        outList.append({'tweet': oriTweet, 'author': brand, 'replies': replyList})
                inputFile.close()

    outputFile = open(outputFile, 'w')
    outputFile.write(json.dumps(outList)+'\n')
    outputFile.close()


def showReplies2(inputFolder, outputFile):
    outList = []
    for filename in os.listdir(inputFolder):
        if filename.endswith('txt'):
            inputFile = open(inputFolder + '/' + filename, 'r')
            for line in inputFile:
                try:
                    data = json.loads(line.strip())
                except:
                    continue
                for item in data:
                    oriTweet = item['tweet']['content']
                    if item['tweet']['author_name'] == 'dominos':
                        replyList = []
                        if len(item['replies']) > 1:
                            for reply in item['replies']:
                                if len(reply['content']) > 10:
                                    replyList.append(reply['content'].replace('@dominos', ''))
                            outList.append({'tweet': oriTweet, 'replies': replyList})
            inputFile.close()

    outputFile = open(outputFile, 'w')
    outputFile.write(json.dumps(outList) + '\n')
    outputFile.close()


if __name__ == '__main__':
    brandList = ['espn', 'PlayStation', 'Sony', 'SpaceX', 'YouTube']
    showReplies(brandList, 'result/api.reply')
    #showReplies2('replies_astute', 'result/reply.result2')