import en_core_web_lg
#import spacy
import json
from spacy import displacy
nlp = en_core_web_lg.load()

def dependencyParse(inputFilename, outputFilename, lenLimit=5):
  print('Loading data...')
  data = []
  with open(inputFilename, 'r') as inputFile:
    for line in inputFile:
      temp = line.strip().split(' >>><<< ')
      if len(temp[0].split(' ')) > lenLimit and len(temp[1].split(' ')) > lenLimit:
        data.append(temp[0])
        data.append(temp[1])

  print('Processing...')
  outputFile = open(outputFilename, 'w')
  total = str(len(data))
  for index, item in enumerate(data):
    temp = []
    resultList = nlp(item)
    for result in resultList:
      temp.append([result.text, result.tag_, result.head.text, result.dep_])
    outputFile.write(json.dumps(temp)+'\n')
    if index%50000 == 0:
      print(str(index)+' / '+total)
      outputFile.flush()
  outputFile.close()
  print(index)


if __name__ == '__main__':
    #dependencyParse('data/pmt/pmt.full.line.lm_4-4', 'data/pmt/pmt.full.line.dependency_4-4', 10)
    results = nlp("Ridley Scott says he plans to make 6 more alien films")
    for result in results:
        print(result.text, result.tag_, result.head.text, result.dep_)
    displacy.serve(results, style='dep')
