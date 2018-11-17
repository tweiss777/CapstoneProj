import re

from docx import Document


def gen_paragraph(document):
    # holds the document paragraphs
    resume = [line.text for line in document.paragraphs]

    # True/False array
    isEmptyArr = []
    isEmpty = False
    # Array that will hold the paragraphs
    paragraphs = []

    section = ""
    # array that will populate the true false array
    for line in resume:
        if len(line) == 0 or bool(re.match('^\s+$', line)) or line in '()~><?'';":\/|{},[]@6&*-_.':
            isEmptyArr.append(True)
        else:
            isEmptyArr.append(False)

    for i in range(len(resume)):
        # print("%s " % resume[i] + "||| % s" % isEmptyArr[i])
        if isEmptyArr[i] is not True:
            section += resume[i] + "\n"
        else:
            paragraphs.append(section)
            section = ""
    paragraphs.append(section)
    return paragraphs


def determine_headers(paragraphs):
    # load up the headers.txt file
    headers = open("Headers.txt", "r", encoding='utf-8')
    headersArr = list(headers)

    # strip the new lines from each indice
    for i in range(len(headersArr)):
        headersArr[i] = headersArr[i].rstrip('\n')

    # iterate through each line in the arr


def main():
    with open("TalWeissResume.docx", "rb") as file:
        doc = Document(file)
        paragraphs = gen_paragraph(doc)

    determine_headers(paragraphs)


main()
