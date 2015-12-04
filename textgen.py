# -*- coding: utf-8 -*-
import sys
import os
import copy
import random
import collections
import bisect
import numpy
import re
import pickle


class LemmaTypes:
    """
    Lemma types enumeration
    """
    Word, MiddleSentencePunctuation, EndSentencePunctuation = range(3)


class Lemma(object):
    """
    Class defines structure of lemma - basic element of analysis.
    """
    _end_sentence_punctuation = {".", "!", "?"}
    _middle_sentence_punctuation = {",", ":", ";"}

    def __init__(self, letter_sequence):
        self.lemma_value = letter_sequence
        if letter_sequence in self._end_sentence_punctuation:
            self.lemma_type = LemmaTypes.EndSentencePunctuation
        elif letter_sequence in self._middle_sentence_punctuation:
            self.lemma_type = LemmaTypes.MiddleSentencePunctuation
        else:
            self.lemma_type = LemmaTypes.Word

    def is_end_sentence(self):
        return self.lemma_type == LemmaTypes.EndSentencePunctuation


class TextLemmatizer(object):
    """
    Class defines lemmatizer, which parces text into separate lemmas.
    """
    def __init__(self, text):
        # cleaning the text
        text = re.sub(ur'[@$"#%^&*()“”]<>', ' ', text)
        apostrophes_front = re.compile(u"['’]+.*")
        apostrophes_back = re.compile(u".*['’]+")
        self.lemmas_ = [Lemma(element)
                        for element in re.findall(ur"[\w'’]+|[.!?,:;]", text)
                        if not (apostrophes_front.match(element) or
                                apostrophes_back.match(element))]

        # decapitalizing
        inside_capitals = re.compile(".+[A-Z].*")
        for i, lemma in enumerate(self.lemmas_):
            if (i == 0 or inside_capitals.match(lemma.lemma_value) or
               i > 0 and self.lemmas_[i - 1].is_end_sentence()):
                lemma.lemma_value = lemma.lemma_value.lower()
        self.current_position_ = 0

    def __iter__(self):
        sentence = self.next_sentence_()
        while len(sentence) > 0:
            yield sentence
            sentence = self.next_sentence_()

    def next_sentence_(self):
        sentence_begin = self.current_position_
        while (self.current_position_ < len(self.lemmas_) and
               not self.lemmas_[self.current_position_].is_end_sentence()):
            self.current_position_ += 1
        if (self.current_position_ < len(self.lemmas_) and
           self.lemmas_[self.current_position_].is_end_sentence()):
            self.current_position_ += 1
            return self.lemmas_[sentence_begin:self.current_position_]
        return []


class FrequencyDistribution(object):
    """
    Class defines frequency distribution and efficient sampling from it.
    """
    def add(self, item):
        self.counter_[item] += 1
        self.sum_ += 1
        self.items_, self.cum_weights_ = None, None

    def get_random_item(self):
        if self.sum_ == 0:
            raise LookupError("No items found")

        if self.items_ is None:
            self.prepare_distribution_()

        x = random.random() * self.sum_
        index = bisect.bisect(self.cum_weights_, x)
        return self.items_[index if index < len(self.items_) else index - 1]

    def __init__(self):
        self.counter_ = collections.Counter()
        self.sum_ = 0
        self.items_, self.cum_weights_ = None, None

    def prepare_distribution_(self):
        self.items_ = self.counter_.keys()
        self.cum_weights_ = list(numpy.cumsum(self.counter_.values()))


class MarkovChainModel(object):
    """
    Class defines Markov Chain Model of arbitrty order k.
    The structure of the statistics is the following:
    chains[0] = {() : distribution}
    chains[1] = {(word) : distribution}
    chains[2] = {(word, word) : distribution}
    ...
    """
    def add_sequence(self, sequence):
        if len(sequence) > 1:
            self.sentence_starts_.add(sequence[0])
        for i in range(1, len(sequence)):
            for j in range(max(i - self.order, 0), i + 1):
                self.add_chain_(tuple(sequence[j:i]), sequence[i])

    def generate_next_item(self, sequence):
        if not self.chains_[0]:
            raise LookupError("Model was not trained yet")

        if len(sequence) == 0:
            return self.sentence_starts_.get_random_item()

        if len(sequence) > self.order:
            raise LookupError("Model order is less than sequence length")

        sequence = tuple(sequence)
        if sequence not in self.chains_[len(sequence)]:
            sequence = ()

        return self.chains_[len(sequence)][sequence].get_random_item()

    def __init__(self, order=2):
        self.order = order
        self.sentence_starts_ = FrequencyDistribution()
        self.chains_ = [dict() for i in range(order + 1)]

    def add_chain_(self, history_items, item):
        chain_index = len(history_items)
        if history_items not in self.chains_[chain_index]:
            self.chains_[chain_index][history_items] = FrequencyDistribution()
        self.chains_[chain_index][history_items].add(item)


class TextGenerator(object):
    """
    The main class for generating text. After training Markov model on a
    given corpus of .txt files the random text can be generated.
    """
    def generate_text(self, num_words_lower_bound, num_par_sentences=10):
        self.word_gen_count_ = 0
        text = ""
        while self.word_gen_count_ < num_words_lower_bound:
            paragraph = " ".join([self.generate_sentence_()
                                  for i in range(num_par_sentences)])
            text += paragraph + os.linesep * 2
        return text

    def next_sentence(self):
        while True:
            yield self.generate_sentence_()

    def __init__(self, seed=None):
        self.seed_ = seed
        self.train_corpus_path_ = ""
        self.word_gen_count_ = 0
        self.word_processed_count_ = 0
        self.lemmas_to_ids_ = dict()
        self.ids_to_lemmas_ = dict()
        self.markov_model_ = MarkovChainModel()

    def train(self, train_corpus_path):
        if self.seed_ is not None:
            random.seed(self.seed_)
        self.train_corpus_path_ = train_corpus_path
        for root, directories, filenames in os.walk(train_corpus_path):
            for filename in filenames:
                if not filename.endswith(".txt"):
                    continue
                with open(os.path.join(root, filename), "r") as fin:
                    text = fin.read().decode("utf8")
                    self.update_markov_model_(text)
                    print os.path.join(root, filename), "processed"
                    print "Total words processed =", self.word_processed_count_
                    print "Total unique words =", len(self.lemmas_to_ids_)

    def save_model(self, filename):
        fout = open(filename, "wb")
        p = pickle.dump(self, fout)
        fout.close()

    def load_model(self, filename):
        fin = open(filename, "rb")
        m = pickle.load(fin)
        fin.close()
        self.seed_ = m.seed_
        self.train_corpus_path_ = m.train_corpus_path_
        self.word_gen_count_ = m.word_gen_count_
        self.word_processed_count_ = m.word_processed_count_
        self.lemmas_to_ids_ = m.lemmas_to_ids_
        self.ids_to_lemmas_ = m.ids_to_lemmas_
        self.markov_model_ = m.markov_model_

    def update_markov_model_(self, text):
        for sentence in TextLemmatizer(text):
            id_sequence = self.convert_to_id_sequence_(sentence)
            self.markov_model_.add_sequence(id_sequence)

    def convert_to_id_sequence_(self, sentence):
        id_sequence = []
        for lemma in sentence:
            if lemma.lemma_type == LemmaTypes.Word:
                self.word_processed_count_ += 1
            if lemma.lemma_value not in self.lemmas_to_ids_:
                new_id = len(self.lemmas_to_ids_)
                self.lemmas_to_ids_[lemma.lemma_value] = new_id
                self.ids_to_lemmas_[new_id] = lemma
            id_sequence.append(self.lemmas_to_ids_[lemma.lemma_value])
        return id_sequence

    def convert_to_lemma_sequence_(self, lemma_id_sequence):
        return [copy.deepcopy(self.ids_to_lemmas_[lemma_id])
                for lemma_id in lemma_id_sequence]

    def generate_sentence_(self):
        id_seq = self.generate_id_sequence_()
        lemma_seq = self.convert_to_lemma_sequence_(id_seq)
        lemma_seq[0].lemma_value = lemma_seq[0].lemma_value.capitalize()
        s = [(" " if (lemma.lemma_type == LemmaTypes.Word and i > 0) else "") +
             lemma.lemma_value for (i, lemma) in enumerate(lemma_seq)]
        return "".join(s)

    def generate_id_sequence_(self):
        id_sequence = []
        history = ()
        prev_lemma_type = LemmaTypes.EndSentencePunctuation
        while True:
            curr_id = self.markov_model_.generate_next_item(history)
            lemma_type = self.ids_to_lemmas_[curr_id].lemma_type
            if (prev_lemma_type != LemmaTypes.Word and
               lemma_type != LemmaTypes.Word):
                continue
            id_sequence.append(curr_id)
            prev_lemma_type = lemma_type
            if lemma_type == LemmaTypes.EndSentencePunctuation:
                break
            if lemma_type == LemmaTypes.Word:
                self.word_gen_count_ += 1
            if len(history) < self.markov_model_.order:
                history = history + (curr_id, )
            else:
                history = history[1:] + (curr_id, )
        return id_sequence
