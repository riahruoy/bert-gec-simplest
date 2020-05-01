


'''
S Mobile phone is a marvellous invention but as for cigarettes , we should right on it
A 0 1|||M:DET|||The mobile|||REQUIRED|||-NONE-|||0
A 6 6|||M:PUNCT|||,|||REQUIRED|||-NONE-|||0
A 8 9|||R:PREP|||with|||REQUIRED|||-NONE-|||0
A 13 14|||R:OTHER|||write|||REQUIRED|||-NONE-|||0
A 16 16|||M:PUNCT|||:|||REQUIRED|||-NONE-|||0
'''
import copy


class Sentence:
    def __init__(self, question):
        self.question = question
        self.corrections = []

    def addCorrection(self, start, end, type, correction):
        self.corrections.append([int(start), int(end), type, correction])

    def getAnswer(self):
        if len(self.corrections) == 0:
            return self.question
        fixed = []
        pos = 0
        q_list = self.question.split(' ')
        for correction in self.corrections:
            fixed += q_list[pos:(int(correction[0]))]
            fixed += [correction[3]]
            pos = int(correction[1])
        fixed += q_list[pos:]
        return ' '.join(fixed)
    @staticmethod
    def readfile(filename):
        f = open(filename, "r")
        block_start = False
        sentences = []
        current_sentence = Sentence("")

        for line in f:
            line = line.strip('\n')
            if line == "":
                if block_start:
                    sentences.append(copy.deepcopy(current_sentence))
                    block_start = False
            elif line[0] == "S":
                block_start = True
                question = line[2:]
                current_sentence = Sentence(question)
            elif line[0] == "A":
                if not block_start:
                    print("format error")
                index, type, correct, other1, other2, annotator = line.split('|||')
                if type == "noop":
                    continue
                _, start, end = index.split(' ')
                current_sentence.addCorrection(start, end, type, correct)
        return sentences

def readfile(filename):
    sentences = Sentence.readfile(filename)
    # convert to question to answer
    src_list = []
    trg_list = []
    for sentence in sentences:
        src_list.append(sentence.question)
        trg_list.append(sentence.getAnswer())

    return src_list, trg_list

def create_vocab(list):
    word_id = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    id_word = ["<pad>", "<unk>", "<s>", "</s>"]
    for sentence in list:
        for word in sentence.split(' '):
            if word in word_id:
                continue
            word_id[word] =len(id_word)
            id_word.append(word)
    return word_id, id_word


