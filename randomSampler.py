

def sampler(inputFilename, outputFilename, fold):
    outputFile = open(outputFilename, 'w')
    with open(inputFilename, 'r') as inputFile:
        for index, line in enumerate(inputFile):
            if index % fold == 0:
                outputFile.write(line)
    outputFile.close()


if __name__ == '__main__':
    sampler('data/para-nmt-5m-processed.txt', 'data/para-nmt-5m-processed.sample', 5)