import argparse
from itertools import groupby

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

addressVocabulary = ['tel', 'fax', 'e-mail', 'email', 'p.o.', 'attn', 'phone', 'tel:', 'fax:', 'e-mail:', 'email:', 'p.o.:', 'attn:', 'phone:']

class SegmentClassifier:
    def train(self, trainX, trainY):
        # self.clf = DecisionTreeClassifier()
        # self.clf = svm.SVC()
        self.clf = RandomForestClassifier()
        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)

    def extract_features(self, text):
        words = text.split()
        centerPosition = rightPositionOfFirstWord(text) - leftPositionOfLastWord(text) / 2 + leftPositionOfLastWord(text)
        features = [
            len(text),
            len(text.strip()),
            len(words),
            1 if '>' in words else 0,
            text.count(' '),
            text.count('\n'), # length of block in lines
            rightPositionOfFirstWord(text),
            1 if abs(40 - centerPosition) <= 3 else 0, # is the text centered
            text.count("--"),
            sum(1 if isWordFollowedByColon(w) else 0 for w in words),
            sum(1 if w.isupper() else 0 for w in words),
            computeRatioOfNumbersToWords(words),
            totalNumberOfNumbers(words),
            numberOfOneAndTwoLetterWords(words) / len(words),
            text.count('@'),
            text.count(':'),
            text.count('-'),
            text.count('.'),
            text.count('?'),
            text.count('/'),
            sum(1 if w.lower() in addressVocabulary else 0 for w in words),
        
            # 624 / 722  0.8642659279778393
            # 63 / 78  0.8076923076923077

            sum(1 if w == 'From:' else 0 for w in words),
            sum(1 if w == 'Lines:' else 0 for w in words),
            sum (1 if w == 'Path:' else 0 for w in words),

            # position relative to NNHEAD 
            # detection for centering, indentation
            # detector for length of block, number of line in block
            # spacing in between words
            # ratio of 1 to 2 letters to everything else
            # sum(1 if isAreaCode(w) else 0 for w in words),

            # length of block in lines
            # position of the block from top/bottom
            # is-text-centered?
                # centerposition = rightpos-leftpos / 2 + leftpos
                # absoffsetfrom40 = abs(40-centerposition)
            # average / max line length in document
            # indent-of-line (chars of whitespace before 1st char)

            # % of ascii characters / (num lines * 80)
            # %ofAlphaNumeric, %ofAsciiGraphicChars

            sum(1 if char.isalnum() else 0 for char in text),

            #ZIP CODE DETECTOR AND NUMBER DETECTOR
            # Department, Institute, Division, Program, Committee, chair etc. on an indented block
        ]
        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)

def rightPositionOfFirstWord(text):
    i = 0
    while text[i] == " ":
        i += 1
    return i

def leftPositionOfLastWord(text):
    i = rightPositionOfFirstWord(text)
    while text[i] != " " or text[i] != "\n":
        i += 1
        if i == len(text):
            break
    return i

def isWordFollowedByColon(word):
    if ord(word[0]) >= 65 and ord(word[0]) <= 90 and word[len(word) - 1] == ':':
        return 1
    return 0

def totalNumberOfNumbers(words):
    numNumbers = 0
    for w in words:
        if any(char.isdigit for char in w):
            numNumbers += 1
    return numNumbers

def computeRatioOfNumbersToWords(words):
    numWords = 0
    numNumbers = 0
    for w in words:
        if any(char.isdigit for char in w):
            numNumbers += 1
        else:
            numWords += 1
    if numWords == 0:
        return numNumbers - 1
    else:
        return numNumbers / numWords

def numberOfOneAndTwoLetterWords(words):
    total = 0
    for w in words:
        if len(w) <= 2:
            total += 1
    return total

def isAreaCode(word):
    if len(word) == 5 and word[0] == '(' and word[4] == ')' and word[2].isdigit() and word[3].isdigit() and word[4].isdigit():
        return 1


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == '#BLANK#':
                continue
            X.append(arr[1])
            y.append(arr[0])
        return X, y


def lines2segments(trainX, trainY):
    segX = []
    segY = []
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        x = '\n'.join(line[0].rstrip('\n') for line in group)
        segX.append(x)
        segY.append(y)
    return segX, segY


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    if args.format == 'segment':
        trainX, trainY = lines2segments(trainX, trainY)
        testX, testY = lines2segments(testX, testY)

    classifier = SegmentClassifier()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()