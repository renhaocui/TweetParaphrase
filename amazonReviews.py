# -*- coding: utf-8 -*-
from lxml import html
import requests
import json
from dateutil import parser as dateparser
from time import sleep
from urllib2 import *
import simplejson
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.parse.stanford import StanfordDependencyParser
reload(sys)
sys.setdefaultencoding('utf8')

StanfordJar = '../'


def ParseReviews(asin):
    for i in range(5):
        try:
            # This script has only been tested with Amazon.com
            amazon_url = 'http://www.amazon.com/dp/' + asin
            # Add some recent user agent to prevent amazon from blocking the request
            # Find some chrome user agent strings  here https://udger.com/resources/ua-list/browser-detail?browser=Chrome
            headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
            page = requests.get(amazon_url, headers=headers, verify=False)
            page_response = page.text

            parser = html.fromstring(page_response)
            XPATH_AGGREGATE = '//span[@id="acrCustomerReviewText"]'
            XPATH_REVIEW_SECTION_1 = '//div[contains(@id,"reviews-summary")]'
            XPATH_REVIEW_SECTION_2 = '//div[@data-hook="review"]'

            XPATH_AGGREGATE_RATING = '//table[@id="histogramTable"]//tr'
            XPATH_PRODUCT_NAME = '//h1//span[@id="productTitle"]//text()'
            XPATH_PRODUCT_PRICE = '//span[@id="priceblock_ourprice"]/text()'

            raw_product_price = parser.xpath(XPATH_PRODUCT_PRICE)
            product_price = ''.join(raw_product_price).replace(',', '')

            raw_product_name = parser.xpath(XPATH_PRODUCT_NAME)
            product_name = ''.join(raw_product_name).strip()
            total_ratings = parser.xpath(XPATH_AGGREGATE_RATING)
            reviews = parser.xpath(XPATH_REVIEW_SECTION_1)
            if not reviews:
                reviews = parser.xpath(XPATH_REVIEW_SECTION_2)
            ratings_dict = {}
            reviews_list = []

            if not reviews:
                raise ValueError('unable to find reviews in page')

            # grabing the rating  section in product page
            for ratings in total_ratings:
                extracted_rating = ratings.xpath('./td//a//text()')
                if extracted_rating:
                    rating_key = extracted_rating[0]
                    raw_raing_value = extracted_rating[1]
                    rating_value = raw_raing_value
                    if rating_key:
                        ratings_dict.update({rating_key: rating_value})
            # Parsing individual reviews
            for review in reviews:
                XPATH_RATING = './/i[@data-hook="review-star-rating"]//text()'
                XPATH_REVIEW_HEADER = './/a[@data-hook="review-title"]//text()'
                XPATH_REVIEW_POSTED_DATE = './/a[contains(@href,"/profile/")]/parent::span/following-sibling::span/text()'
                XPATH_REVIEW_TEXT_1 = './/div[@data-hook="review-collapsed"]//text()'
                XPATH_REVIEW_TEXT_2 = './/div//span[@data-action="columnbalancing-showfullreview"]/@data-columnbalancing-showfullreview'
                XPATH_REVIEW_COMMENTS = './/span[@data-hook="review-comment"]//text()'
                XPATH_AUTHOR = './/a[contains(@href,"/profile/")]/parent::span//text()'
                XPATH_REVIEW_TEXT_3 = './/div[contains(@id,"dpReviews")]/div/text()'
                raw_review_author = review.xpath(XPATH_AUTHOR)
                raw_review_rating = review.xpath(XPATH_RATING)
                raw_review_header = review.xpath(XPATH_REVIEW_HEADER)
                raw_review_posted_date = review.xpath(XPATH_REVIEW_POSTED_DATE)
                raw_review_text1 = review.xpath(XPATH_REVIEW_TEXT_1)
                raw_review_text2 = review.xpath(XPATH_REVIEW_TEXT_2)
                raw_review_text3 = review.xpath(XPATH_REVIEW_TEXT_3)

                author = ' '.join(' '.join(raw_review_author).split()).strip('By')

                # cleaning data
                review_rating = ''.join(raw_review_rating).replace('out of 5 stars', '')
                review_header = ' '.join(' '.join(raw_review_header).split())
                review_posted_date = dateparser.parse(''.join(raw_review_posted_date)).strftime('%d %b %Y')
                review_text = ' '.join(' '.join(raw_review_text1).split())

                # grabbing hidden comments if present
                if raw_review_text2:
                    json_loaded_review_data = json.loads(raw_review_text2[0])
                    json_loaded_review_data_text = json_loaded_review_data['rest']
                    cleaned_json_loaded_review_data_text = re.sub('<.*?>', '', json_loaded_review_data_text)
                    full_review_text = review_text + cleaned_json_loaded_review_data_text
                else:
                    full_review_text = review_text
                if not raw_review_text1:
                    full_review_text = ' '.join(' '.join(raw_review_text3).split())

                raw_review_comments = review.xpath(XPATH_REVIEW_COMMENTS)
                review_comments = ''.join(raw_review_comments)
                review_comments = re.sub('[A-Za-z]', '', review_comments).strip()
                review_dict = {
                    'review_comment_count': review_comments,
                    'review_text': full_review_text,
                    'review_posted_date': review_posted_date,
                    'review_header': review_header,
                    'review_rating': review_rating,
                    'review_author': author

                }
                reviews_list.append(review_dict)

            data = {
                'ratings': ratings_dict,
                'reviews': reviews_list,
                'url': amazon_url,
                'price': product_price,
                'name': product_name
            }
            return data
        except ValueError:
            print "Retrying to get the correct response"

    return {"error": "failed to process the page", "asin": asin}


def ReadAsin():
    # Add your own ASINs here
    AsinList = ['B01ETPUQ6E', 'B017HW9DEW']
    extracted_data = []
    for asin in AsinList:
        print "Downloading and processing page http://www.amazon.com/dp/" + asin
        extracted_data.append(ParseReviews(asin))
        sleep(5)
    f = open('reviews.json', 'w')
    json.dump(extracted_data, f, indent=4)
    f.close()


def extractSolr():
    rawFile = open('data/amazonReview/raw.json', 'w')
    solrURL = 'http://34.238.93.154:8983/solr/webfeeds/select?wt=json&'
    queryURL = 'q=*:*&fq=tstamp:%5b2000-01-01T00%3a00%3a00Z+TO+2018-05-01T00%3a00%3a00Z%5d&fq=(feed_layout_type:%22AMAZON_REVIEWS%22)&start=0&rows=160201&fl=context,content,feed_url'
    connection = urlopen(solrURL + queryURL)
    response = simplejson.load(connection)
    #print json.dumps(response['response']['docs'][0])

    for item in response['response']['docs']:
        rawFile.write(json.dumps(item)+'\n')
    rawFile.close()


def orgnaizeSolr(inputFile, outputFile, sampleFolder):
    rawFile = open(inputFile, 'r')
    reviewData = {}
    for line in rawFile:
        data = json.loads(line.strip())
        feed = data['feed_url']
        if feed not in reviewData:
            data.pop('feed_url')
            reviewData[feed] = [data]
        else:
            data.pop('feed_url')
            reviewData[feed].append(data)
    rawFile.close()

    outData = {}
    largest = ('a', -1)
    for feed, value in reviewData.items():
        outData[feed] = {}
        if len(value) > largest[1]:
            largest = (feed, len(value))
        for review in value:
            rate = json.loads(json.loads(review['context'])['subcontext'])['starsValue']
            if 10 < len(review['content']) < 1000:
                if rate not in outData[feed]:
                    outData[feed][rate] = [review]
                else:
                    outData[feed][rate].append(review)


    outFile = open(outputFile, 'w')
    outFile.write(json.dumps(outData))
    outFile.close()

    for star in range(5):
        sampleFile = open(sampleFolder+'/'+str(star+1)+'.content', 'w')
        for review in outData[largest[0]][star+1]:
            sampleFile.write(review['content']+'\n')
        sampleFile.close()
    print(len(reviewData))


def processSolr(inputFile, outputFile):
    contentFile = open(inputFile, 'r')
    outData = {}
    for line in contentFile:
        content = line.strip()
        sentences = sent_tokenize(content)
        for sentence in sentences:
            words = word_tokenize(sentence)
            posPairs = pos_tag(words)
            for pair in posPairs:
                if pair[1] == 'NN':
                    if pair[0] not in outData:
                        outData[pair[0]] = [sentence]
                    else:
                        outData[pair[0]].append(sentence)
    contentFile.close()

    outFile = open(outputFile, 'w')
    for feed, sentences in outData.items():
        if len(sentences) > 2:
            outFile.write(feed + '\n')
            for sentence in sentences:
                outFile.write(sentence+'\n')
            outFile.write('\n')
    outFile.close()



if __name__ == '__main__':
    #ReadAsin()
    #extractSolr()
    #orgnaizeSolr('data/amazonReview/raw.json', 'data/amazonReview/brands.json', 'data/amazonReview/sample')
    processSolr('data/amazonReview/sample/1.content', 'data/amazonReview/sample/1.list')