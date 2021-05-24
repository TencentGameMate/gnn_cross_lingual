#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper functions for part-of-speech tagging.
"""
import re
import os
import json
import time
import random
import pickle
import logging
from tqdm import tqdm
from datetime import datetime

import nltk
from nltk.tag import StanfordPOSTagger
import spacy
import MeCab

logger = logging.getLogger(__file__)

class DumbTokenizer:
    def tokenize(self, text):
        return text

class PosTagger:
    """
    Multilingual Pos Tagger
    """
    # Tag table conversions
    # Penn TreeBank: https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
    TAG_CONV_PTB = {
        "$": "SYM", "''": "PUNCT", ",": "PUNCT", "-LRB-": "PUNCT", "-RRB-": "PUNCT",
        ".": "PUNCT", ":": "PUNCT", "AFX": "ADJ", "CC": "CCONJ", "CD": "NUM", "DT": "DET",
        "EX": "PRON", "FW": "X", "HYPH": "PUNCT", "IN": "ADP", "JJ": "ADJ", "JJR": "ADJ",
        "JJS": "ADJ", "LS": "X", "MD": "VERB", "NIL": "X", "NN": "NOUN", "NNP": "PROPN",
        "NNPS": "PROPN", "NNS": "NOUN", "PDT": "DET", "POS": "PART", "PRP": "PRON",
        "PRP$": "DET", "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "RP": "ADP", "SYM": "SYM",
        "TO": "PART", "UH": "INTJ", "VB": "VERB", "VBD": "VERB", "VBG": "VERB",
        "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB", "WDT": "DET", "WP": "PRON",
        "WP$": "DET", "WRB": "ADV", "``": "PUNCT"
    }
    # ja: simplified MeCab tagset
    TAG_CONV_MECAB = {
        "代名詞": "PRON", "副詞": "ADV", "助動詞": "AUX", "助詞": "ADP", "動詞": "VERB", "名詞": "NOUN",
        "形容詞": "ADJ", "形状詞": "ADJ", "感動詞": "INTJ", "接尾辞": "CCONJ", "接続詞": "CCONJ", "接頭辞": "X",
        "空白": "PUNCT", "補助記号": "PUNCT", "記号": "SYM", "連体詞": "ADJ"
    }
    # ru: Russian National Corpus (RNC)
    TAG_CONV_RNC = {'A': 'ADJ', 'A-PRO': 'PRON', 'ADV': 'ADV', 'ADV-PRO': 'PRON', 'ANUM': 'ADJ',
                    'CONJ': 'CONJ', 'INTJ': 'X', 'NONLEX': '.', 'NUM': 'NUM', 'PARENTH': 'PRT',
                    'PART': 'PRT', 'PR': 'ADP', 'PRAEDIC': 'PRT', 'PRAEDIC-PRO': 'PRON', 'S': 'NOUN',
                    'S-PRO': 'PRON', 'V': 'VERB'
    }

    def __init__(self, lang, tagger_root_path, **kwargs):
        self.tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.tagger = getattr(self, f"_pos_tag_{lang}_init")(tagger_root_path, **kwargs)
        self.tag = getattr(self, f"_pos_tag_{lang}")
        
        # load cached tag results
        self.cache_results_path = os.path.join(tagger_root_path, "tag_cache", lang)
        if not os.path.exists(self.cache_results_path):
            os.makedirs(self.cache_results_path)
        self.cache_results = {}
        for file_name in os.listdir(self.cache_results_path):
            if file_name.endswith(".json"):
                with open(os.path.join(self.cache_results_path, file_name), encoding="utf-8") as fin:
                    for item in json.load(fin):
                        self.cache_results[item["text"]] = item["res"]

    def _convert_tagset(self, tag_results, conv_table):
        tag_results_ud = []
        for tag_result_sent in tag_results:
            tag_results_ud.append([(word, conv_table.get(pos, "X")) for word, pos in tag_result_sent])
        return tag_results_ud

    def _pos_tag_en_init(self, model_path, *args, **kwargs):
        path_to_model = os.path.join(model_path, "..", "stanford-postagger-full-2020-11-17/models", "english-left3words-distsim.tagger")
        path_to_jar = os.path.join(model_path, "..", "stanford-postagger-full-2020-11-17", "stanford-postagger-4.2.0.jar")
        return StanfordPOSTagger(path_to_model, path_to_jar)

    def _pos_tag_en(self, texts):
        tag_results = self.tagger.tag_sents(texts)
        return self._convert_tagset(tag_results, PosTagger.TAG_CONV_PTB)

    def _pos_tag_de_init(self, model_path, *args, **kwargs):
        #path_to_model = os.path.join(model_path, "stanford-postagger-full-2020-11-17/models", "german-ud.tagger")
        #path_to_jar = os.path.join(model_path, "stanford-postagger-full-2020-11-17", "stanford-postagger-4.2.0.jar")
        #return StanfordPOSTagger(path_to_model, path_to_jar)
        self.tokenizer = DumbTokenizer()
        return spacy.load('de_core_news_lg')

    def _pos_tag_de(self, texts):
        tag_results = []
        for text in tqdm(texts, desc="tagging"):
            doc = self.tagger(text)
            tag_results.append([(token.text, token.pos_) for token in doc])
        return tag_results
        #return self.tagger.tag_sents(texts)

    def _pos_tag_fr_init(self, model_path, *args, **kwargs):
        #path_to_model = os.path.join(model_path, "stanford-postagger-full-2020-11-17/models", "french-ud.tagger")
        #path_to_jar = os.path.join(model_path, "stanford-postagger-full-2020-11-17", "stanford-postagger-4.2.0.jar")
        #return StanfordPOSTagger(path_to_model, path_to_jar)
        self.tokenizer = DumbTokenizer()
        return spacy.load('fr_core_news_lg')

    def _pos_tag_fr(self, texts):
        tag_results = []
        for text in tqdm(texts, desc="tagging"):
            doc = self.tagger(text)
            tag_results.append([(token.text, token.pos_) for token in doc])
        return tag_results
    
    def _pos_tag_es_init(self, model_path, *args, **kwargs):
        self.tokenizer = DumbTokenizer()
        return spacy.load('es_core_news_lg')

    def _pos_tag_es(self, texts):
        tag_results = []
        for text in tqdm(texts, desc="tagging"):
            doc = self.tagger(text)
            tag_results.append([(token.text, token.pos_) for token in doc])
        return tag_results

    def _pos_tag_ja_init(self, model_path, *args, **kwargs):
        self.tokenizer = DumbTokenizer()
        return MeCab.Tagger("-r /dev/null")

    def _pos_tag_ja(self, texts):
        tag_results = []
        for text in texts:
            res = self.tagger.parse(text)
            tag_result_sent = []
            for line in res.split("\n"):
                if line == "EOS":
                    break
                seg = line.split("\t")
                tag_result_sent.append((seg[0], PosTagger.TAG_CONV_MECAB.get(seg[4].split("-")[0])))
            tag_results.append(tag_result_sent)
        return tag_results

    def _pos_tag_th_init(self, model_path, *args, **kwargs):
        from pythainlp.tokenize import word_tokenize
        self.tokenizer = DumbTokenizer()
        self.tokenizer.tokenize = lambda sent: word_tokenize(sent, keep_whitespace=False)
        return None

    def _pos_tag_th(self, texts):
        from pythainlp.tag import pos_tag_sents as pos_tag_th
        return pos_tag_th(texts, corpus='pud')
    
    def _pos_tag_ru_init(self, model_path, *args, **kwargs):
        self.tokenizer.tokenize = lambda sent: nltk.word_tokenize(sent, language="russian")
    
    def _pos_tag_ru(self, texts):
        tag_results = []
        nltk.pos_tag(texts[0], tagset="universal")
        size = len(texts)
        for i, text in enumerate(texts):
            try:
                res = nltk.pos_tag(text, lang="rus")
            except Exception as e:
                print(f"{i} text: {text}")
                raise
            tag_results.append(res)

            if i % 1000 == 0:
                logger.info(f"pos tag ru done {i}/{size}")
        return self._convert_tagset(tag_results, PosTagger.TAG_CONV_RNC)
    
    def pos_tag_sents(self, texts):
        # first fetch data from cache
        results = [None] * len(texts)
        need_tag = []
        for i, sentence in enumerate(texts):
            if sentence in self.cache_results:
                results[i] = {
                    "text": sentence,
                    "res": self.cache_results[sentence]
                }
            else:
                need_tag.append(self.tokenizer.tokenize(sentence))
        
        # run pos tag
        if need_tag:
            time0 = time.time()
            tag_results = self.tag(need_tag)
            cache_results = []
            i = 0
            for res in tag_results:
                while results[i] is not None:
                    i += 1
                item = {
                    "text": texts[i],
                    "res": res
                }
                cache_results.append(item)
                results[i] = item
            # save to cache
            save_name = os.path.join(self.cache_results_path, datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".json")
            with open(save_name, "w", encoding="utf-8") as fout:
                json.dump(cache_results, fout, ensure_ascii=False)
            time1 = time.time() - time0
            logger.info(f"Tagged {len(need_tag)} sentences in {time1:.2f} seconds")

        return results
