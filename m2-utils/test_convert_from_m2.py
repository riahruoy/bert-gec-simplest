import unittest

from load_data import Sentence, readfile, create_vocab

'''
S Nowadays computer are very developmented and it is increasing .
A 1 2|||R:NOUN:NUM|||computers|||REQUIRED|||-NONE-|||0
A 4 5|||R:ADJ|||advanced|||REQUIRED|||-NONE-|||0
A 6 7|||R:PRON|||they|||REQUIRED|||-NONE-|||0
A 7 8|||R:VERB:SVA|||are|||REQUIRED|||-NONE-|||0
A 8 9|||UNK|||increasing|||REQUIRED|||-NONE-|||0
'''
'''
S And nowadays , I have been practicing tennis as school activity .
A 1 2|||R:ADV|||recently|||REQUIRED|||-NONE-|||0
A 6 7|||R:SPELL|||practising|||REQUIRED|||-NONE-|||0
A 9 9|||M:DET|||a|||REQUIRED|||-NONE-|||0
'''

class MyTestCase(unittest.TestCase):
    def test_answer (self):
        s = Sentence("Nowadays computer are very developmented and it is increasing .")
        s.addCorrection(1, 2, "R:NOUN:NUM", "computers")
        s.addCorrection(4, 5, "R:ADJ", "advanced")
        s.addCorrection(6, 7, "R:PRON", "they")
        s.addCorrection(7, 8, "R:VERB:SVA", "are")
        s.addCorrection(8, 9, "UNK", "increasing")
        answer = s.getAnswer()
        self.assertEqual("Nowadays computers are very advanced and they are increasing .", answer)

    def test_answer (self):
        s = Sentence("And nowadays , I have been practicing tennis as school activity .")
        s.addCorrection(1, 2, "R:ADV", "recently")
        s.addCorrection(6, 7, "R:SPELL", "practising")
        s.addCorrection(9, 9, "M:DET", "a")
        answer = s.getAnswer()
        self.assertEqual("And recently , I have been practising tennis as a school activity .", answer)

    def test_readfile(self):
        src_list, trg_list = readfile("fce/m2/fce.dev.gold.bea19.m2")
        self.assertEqual("I have just recieved the letter , which lets me know that I have won the first prize .", src_list[3])
        self.assertEqual("I have just received the letter , which lets me know that I have won the first prize .", trg_list[3])

    def test_create_vocab(self):
        word_to_id, id_to_word = create_vocab(["I have just received the letter , which lets me know that I have won the first prize ."])
        self.assertEqual("just", id_to_word[6])
        self.assertEqual(6, word_to_id["just"])

        self.assertEqual("won", id_to_word[16])
        self.assertEqual(16, word_to_id["won"])

if __name__ == '__main__':
    unittest.main()
