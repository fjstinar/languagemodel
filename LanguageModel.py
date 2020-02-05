from collections import defaultdict
import random
import math
class LanguageModel:
    def __init__(self, n_gram, is_laplace_smoothing, backoff = None):
        #initializes ngram model
        self.is_laplace_smoothing = is_laplace_smoothing
        self.backoff = backoff
        #self.lex_freqs = {}
        self.n_gram = n_gram
        self.gram_model = defaultdict(lambda: defaultdict(lambda: 0))
        self.onegram_model = defaultdict(int)
        self.grams = []
        self.sentence_list = []


        #print("not implemented")

    def train(self, training_file_path):
        #doesn't have a return

        with open(training_file_path, "r") as of:
            for line in of:
                self.sentence_list.append(line[:-1])

        #making grams and counting frequencies

        self.grams = [b for l in self.sentence_list for b in l.split(" ")]
        for i in self.grams:
            self.onegram_model[i] += 1
        self.onegram_model['<unk>'] = 0
        for i in zip(list(self.onegram_model.keys()), list(self.onegram_model.values())):
            if i[1] == 1:
                self.onegram_model['<unk>'] += 1
                del self.onegram_model[i[0]]

        #putting <unk> tokens in

        for i in range(0,len(self.sentence_list)):
            sentence = self.sentence_list[i].split()
            for word in sentence:
                if word not in self.onegram_model.keys():
                    self.sentence_list[i] = self.sentence_list[i].replace(word, '<unk>')


        if self.n_gram == 2:
            self.grams = [b for l in self.sentence_list for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
            for i in self.grams:
                self.gram_model[(i[0])][i[1]] += 1


        #laplace smoothing/ calculate probabilities
        if self.is_laplace_smoothing == True:
            for i in self.onegram_model.keys():
                self.onegram_model[i] += 1

            for i in self.gram_model:
                for j in self.gram_model[i]:
                    self.gram_model[i][j] += 1

            total_count = float(sum(self.onegram_model.values()))
            vocab = len(self.onegram_model.values())
            for i in self.onegram_model.keys():
                self.onegram_model[i] /= (total_count + vocab)

            if self.n_gram == 2:
                #get better way to calculate vocab size
                vocab = 0
                for i in self.gram_model:
                    for j in self.gram_model[i]:
                        vocab += 1

                for i in self.gram_model:
                    total_count = float(sum(self.gram_model[i].values()))
                    for j in self.gram_model[i]:
                        self.gram_model[i][j] /= (total_count + vocab)

        else:

            total_count = float(sum(self.onegram_model.values()))
            for i in self.onegram_model.keys():
                self.onegram_model[i] /= total_count


            if self.n_gram == 2:
                for i in self.gram_model:
                    total_count = float(sum(self.gram_model[i].values()))
                    for j in self.gram_model[i]:
                        self.gram_model[i][j] /= total_count


    def generate(self, num_sentences):
        #returns a list of strings generated using Shannon's method of length num_sentences

        #only bigrams rn
        returned_count = 0
        returned_sentences = []

        if self.n_gram == 1:
            while returned_count < num_sentences:
                sent = ['<s>']
                finished = False
                while not finished:
                    r = random.random()
                    th = .0
                    for i in self.onegram_model:
                        th += self.onegram_model[i]
                        if th >= r and i != '<s>':
                            sent.append(i)
                            break
                    if sent[-1] == '</s>':
                        finished = True
                        break
                returned_sentences.append(' '.join(sent))
                returned_count += 1

        if self.n_gram == 2:
            while returned_count < num_sentences:
                sent = ['<s>']
                finished = False
                while not finished:
                    r = random.random()
                    th = .0
                    for j in self.gram_model[sent[-1]]:
                        th += self.gram_model[sent[-1]][j]
                        if th >= r:
                            sent.append(j)
                            break
                    if sent[-1] == '</s>':
                        finished = True
                        break
                returned_sentences.append(' '.join(sent))
                returned_count += 1

        return returned_sentences

    def score(self, sentence):
        #return a probability for the given sentence
        p = .0
        #replace unseen words with <unk>
        for j in sentence.split():
            if j not in self.onegram_model:
                sentence = sentence.replace(j, '<unk>')


        if self.n_gram == 1:
            sent = sentence.split()
            p = math.log(self.onegram_model[sent[-1]])
            for i in sent[:-1]:
                p += math.log(self.onegram_model[i])

        sent = [sentence, ""]
        if self.n_gram == 2:
            sent = [b for l in sent for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
            p = math.log(self.gram_model[sent[-1][0]][sent[-1][1]])
            for i in sent[:-1]:
                p += math.log(self.gram_model[i[0]][i[1]])
        return math.exp(p)
