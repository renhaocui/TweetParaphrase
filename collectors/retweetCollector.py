import time
import json
import twitter
import utilities

c_k = 'wHqszw01omK3W7sNwZC2XgY2e'
c_s = '2Kmt5CFVG8UikLLKNgTUbertPfxOBHSHaqZDdMZ5T6vgP11iD8'
a_t = '141612471-rtZFDyJrcaLN96FYpTSRyoCyhMcFySLZCTA2VXXF'
a_t_s = 'zYUlpJTApBhtgnAP1PpypO8TCofZdqIGb9CZO6o5Z8vUA'


idSet = set()

def oauth_login():
    # credentials for OAuth
    CONSUMER_KEY = c_k
    CONSUMER_SECRET = c_s
    OAUTH_TOKEN = a_t
    OAUTH_TOKEN_SECRET = a_t_s
    # Creating the authentification
    auth = twitter.oauth.OAuth(OAUTH_TOKEN,
                               OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY,
                               CONSUMER_SECRET)
    # Twitter instance
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


def collector(brandList):
    twitter_api = oauth_login()
    requestLimit = 75
    requestNum = 0

    for brand in brandList:
        print(brand)
        outputFile = open('retweetData/' + brand + '.json', 'a')
        for index in range(1000):
            index = index + 100
            print(index)
            tweetFile = open('brandData/' + brand + '/'+str(index)+'.json', 'r')
            for line in tweetFile:
                data = json.loads(line.strip())
                tweetID = data['id']
                content = data['text']
                if tweetID not in idSet:
                    idSet.add(tweetID)
                    try:
                        requestNum += 1
                        if requestNum > requestLimit:
                            print 'Wait for 15 mins...'
                            time.sleep(900)
                            requestNum = 1
                        response = twitter_api.statuses.retweets(id=tweetID, count=100)
                        temp = []
                        for res in response:
                            if res['text'].replace('RT @'+brand+': ', '') != content:
                                temp.append(res)
                    except Exception as e:
                        print 'Error ' + str(e)
                        continue
                    output = {'text': content, 'id': tweetID, 'retweets': temp}
                    outputFile.write(json.dumps(output)+'\n')
            tweetFile.close()
        outputFile.close()


def analysis(brandList):
    for brand in brandList:
        outputFile = open('result/'+brand+'.retweet', 'w')
        inputFile = open('retweetData/' + brand + '.json', 'r')
        for line in inputFile:
            data = json.loads(line.strip())
            outputList = []
            if len(data['retweets']) > 1:
                #print data['retweets']
                oriTweet = data['text']
                total = float(len(oriTweet))
                for index in range(len(data['retweets'])):
                    currentStr = data['retweets'][index]['text']
                    llcs = utilities.lcsLen(oriTweet, currentStr)
                    if llcs/total < 0.9:
                        outputList.append(currentStr)
                if len(outputList) > 0:
                    outputFile.write(json.dumps({'original': oriTweet, 'retweets': outputList})+'\n')

        inputFile.close()
        outputFile.close()



if __name__ == '__main__':
    brandList = ['netflix']
    #collector(brandList)
    analysis(brandList)
