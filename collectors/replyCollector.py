import json
import twitter
import utilities
import time

c_k = 'wHqszw01omK3W7sNwZC2XgY2e'
c_s = '2Kmt5CFVG8UikLLKNgTUbertPfxOBHSHaqZDdMZ5T6vgP11iD8'
a_t = '141612471-rtZFDyJrcaLN96FYpTSRyoCyhMcFySLZCTA2VXXF'
a_t_s = 'zYUlpJTApBhtgnAP1PpypO8TCofZdqIGb9CZO6o5Z8vUA'


idSet = set()
maxSearchRound = 20
userTimelineRequestLimit = 900
searchRequestLimit = 180
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


def collectTweets(brandList, index):
    twitter_api = oauth_login()
    requestNum = 0
    for brand in brandList:
        print('extracting tweets for: '+brand)
        recordFile = open("brandData/" + brand + '/' + str(index)+'.json', 'w')
        for i in range(10):
            requestNum += 1
            if requestNum > userTimelineRequestLimit:
                print 'waiting for 15m...'
                time.sleep(900)
                requestNum = 0
            try:
                statuses = twitter_api.statuses.user_timeline(screen_name=brand, include_rts='false',
                                                          exclude_replies='true', count=200)
            except Exception as e:  # take care of errors
                print 'API ERROR: '+str(e)
                continue
            for tweet in statuses:
                if tweet['id'] not in idSet:
                    idSet.add(tweet['id'])
                    recordFile.write(json.dumps(tweet))
                    recordFile.write('\n')
        recordFile.close()
    return True


def collectReplies(brandList, index):
    twitter_api = oauth_login()
    requestNum = 0
    for brand in brandList:
        print('Extracting replies for: ' + brand)
        outputFile = open('replyData/' + brand + '/' + str(index) + '.json', 'w')
        tweetFile = open("brandData/" + brand + '/' + str(index)+'.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            tweetID = data['id']
            content = data['text']
            username = data['user']['name']

            temp = []
            for trial in range(maxSearchRound):
                try:
                    requestNum += 1
                    if requestNum > searchRequestLimit:
                        print 'Wait for 15 mins...'
                        time.sleep(900)
                        requestNum = 0
                    response = twitter_api.search.tweets(q="to:"+username, since_id=tweetID, count=100, lang="en", include_entities=True)
                    for res in response['statuses']:
                        if res['in_reply_to_status_id'] == tweetID and res['id'] not in idSet:
                            temp.append(res)
                            idSet.add(res['id'])
                    if len(temp) > 100:
                        break
                except Exception as e:
                    print 'Error ' + str(e)
                    continue
            if len(temp) > 0:
                output = {'text': content, 'id': tweetID, 'replies': temp}
                outputFile.write(json.dumps(output)+'\n')
        tweetFile.close()
        outputFile.close()


def collect(brandList):
    for index in range(9999999):
        startTime = time.time()
        status = collectTweets(brandList, index)
        endTime = time.time()
        if status:
            collectReplies(brandList, index)
        currentTime = time.time()
        if currentTime < endTime + 43200:
            time.sleep(startTime + 43200 - currentTime)


def removeDuplicates(brandList):
    for brand in brandList:
        outputFile = open('result/' + brand + '.clean', 'w')
        inputFile = open('replyData/' + brand + '.json', 'r')
        for line in inputFile:
            replyIDSet = set()
            data = json.loads(line.strip())
            outputList = []
            if len(data['replies']) > 1:
                oriTweet = data['text']
                for reply in data['replies']:
                    if reply['id'] not in replyIDSet:
                        replyIDSet.add(reply['id'])
                        outputList.append(reply['text'])
            outputFile.write(json.dumps({'original': oriTweet, 'replies': outputList})+'\n')

        inputFile.close()
        outputFile.close()


if __name__ == '__main__':
    brandList = []
    brandFile = open('brand.list', 'r')
    for line in brandFile:
        brandList.append(line.strip())
    brandFile.close()
    print brandList
    collect(brandList)

    #brandList = ['AmericanAir', 'Disney', 'Jeep', 'LEVIS', 'Macys', 'SamsungMobile', 'google', 'kraftfoods', 'ATT', 'amazon', 'GEICO', 'Target', 'Yahoo']
    #brandList = ['netflix', 'McDonalds', 'Delta', 'travelchannel']
    #collector(brandList)
    #analysis(brandList)
    #removeDuplicates(brandList)
