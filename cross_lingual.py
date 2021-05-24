#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple
import os
import json
import time
import random
import logging
from math import log10

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import nltk
from nltk import data as nltk_data
from nltk import word_tokenize
from nltk.corpus import stopwords

import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info, generate_mask_tensor

#from nltk.stem import WordNetLemmatizer
#from laserembeddings import Laser
import bert_finetune as mbert_ft
from pos_tag import PosTagger

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


WORD_NODE_TYPE = "word"
DOC_NODE_TYPE = "doc"
DOC_TRANS_EDGE_TYPE = "translate"
DOC_SIM_EDGE_TYPE = "similar"

POS_TAG_LIST = {"ADJ", "ADV", "CCONJ", "ADP", "VERB", "NOUN", "PROPN", "PRON", "PART", "INTJ"}
PUNCTUATION = list(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·""")

LANG_NAME = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "ja": "japanese",
    "th": "thai",
    "es": "spanish"
}

nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)
nltk.download('stopwords', quiet = True)
nltk.download('universal_tagset', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class MQAGraphDataset(DGLDataset):
    def __init__(self, args):
        self.source_path = args.source_path
        self.target_path = args.target_path
        self.output_path = args.data_cache_path
        self.word_node_cnt = args.word_node_cnt
        self.num_classes = args.num_classes
        self.emb_size = args.emb_size

        self.trans_cache_path = args.trans_cache_path
        self.sim_edges = args.sim_edges
        self.trans_edges = args.trans_edges
        self.merge_word = args.merge_word
        self.trans_labels = args.trans_labels
        self.no_pos_tag = args.no_pos_tag
        self.use_unlabeled = args.use_unlabeled

        self.tagger_root_path = args.tagger_root_path

        self.doc_doc_edge_stored = args.doc_doc_edge_stored
        self.doc_doc_edge_file = args.doc_doc_edge_file

        self.transformer_type = args.transformer_type
        self.encode_maxlen = args.encode_maxlen
        self.mbert_model_path = args.mbert_model_path
        self.mbert_feature_cache = args.mbert_feature_file

        self.src_lan = args.src_lan
        self.tgt_lan = args.tgt_lan

        self.device = args.device

        self.graph_path = os.path.join(self.output_path, 'dgl_hetero_graph.bin')
        self.info_path = os.path.join(self.output_path, 'dgl_hetero_info.pkl')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # self.generate_train_val_test()
        # has_cache()
        # download()
        # process()
        # save()
        # load()
        super().__init__("mqa")

    def _file_read(self, file_path):
        texts, labels = [], []
        with open(file_path, 'r', encoding='UTF-8') as fin:
            for line in fin:
                sentence = line.rstrip()
                seg = sentence.split('\t')
                texts.append(seg[1])
                labels.append(seg[0])
        return texts, labels

    def data_source_read(self):
        data_train = os.path.join(self.source_path, 'train.txt')
        data_valid = os.path.join(self.source_path, 'test.txt')
        data_test = os.path.join(self.source_path, 'eval.txt')
        self.source_train_set = []
        self.source_train_label = []
        self.source_valid_set = []
        self.source_valid_label = []
        self.source_test_set = []
        self.source_test_label = []
        #source_vocab = {}
        #source_vocab_size = 0
        self.source_train_set, self.source_train_label = self._file_read(data_train)
        self.source_valid_set, self.source_valid_label = self._file_read(data_valid)
        source_text = self.source_train_set + self.source_valid_set
        if self.use_unlabeled:
            self.source_test_set, self.source_test_label = self._file_read(data_test)
            source_text += self.source_test_set
        source_dic = {}
        for i in range(len(source_text)):
            source_dic[source_text[i]] = 1
        self.dedup_source_text = set(source_dic.keys())
        logger.info(f"dedup_source_text: {len(self.dedup_source_text)}")

    def data_target_read(self):
        data_train = os.path.join(self.target_path, 'train.txt')
        data_valid = os.path.join(self.target_path, 'test.txt')
        data_test = os.path.join(self.target_path, 'eval.txt')
        self.target_train_set = []
        self.target_train_label = []
        self.target_valid_set = []
        self.target_valid_label = []
        self.target_test_set = []
        self.target_test_label = []
        #source_vocab = {}
        #source_vocab_size = 0
        self.target_train_set, self.target_train_label = self._file_read(data_train)
        self.target_valid_set, self.target_valid_label = self._file_read(data_valid)
        target_text = self.target_train_set + self.target_valid_set
        if self.use_unlabeled:
            self.target_test_set, self.target_test_label = self._file_read(data_test)
            target_text += self.target_test_set
        target_dic = {}
        for i in range(len(target_text)):
            target_dic[target_text[i]] = 1
        self.dedup_target_text = set(target_dic.keys())
        logger.info(f"dedup_target_text: {len(self.dedup_target_text)}")

    def generate_corpus2id(self):
        logger.info("Start generating doc dictionary.")
        self.dedup_source_text = list(self.dedup_source_text)
        self.dedup_target_text = list(self.dedup_target_text)
        bilingual_text = []
        bilingual_text.extend(self.dedup_source_text)
        bilingual_text.extend(self.dedup_target_text)
        self.bilingual_corpus2id = {}
        self.ibilingual_corpus2id = {}
        node_index = 0
        for i, item in enumerate(bilingual_text):
            if item not in self.bilingual_corpus2id:
                self.bilingual_corpus2id[item] = node_index
                self.ibilingual_corpus2id[node_index] = item
                node_index += 1
        logger.info(len(self.bilingual_corpus2id))

    def data_generate_train_val_test(self):
        self.train_labels, self.valid_src_labels, self.valid_labels, self.test_labels = [], [], [], []
        train_ids, valid_src_ids, valid_ids, test_ids = [], [], [], []

        dedup_set = set()
        for text, label in zip(self.source_train_set, self.source_train_label):
            if text not in dedup_set:
                train_ids.append(self.bilingual_corpus2id[text])
                self.train_labels.append(label)
                dedup_set.add(text)
        for text, label in zip(self.source_valid_set, self.source_valid_label):
            valid_src_ids.append(self.bilingual_corpus2id[text])
            self.valid_src_labels.append(label)
        for text, label in zip(self.target_train_set, self.target_train_label):
            valid_ids.append(self.bilingual_corpus2id[text])
            self.valid_labels.append(label)
        for text, label in zip(self.target_valid_set, self.target_valid_label):
            test_ids.append(self.bilingual_corpus2id[text])
            self.test_labels.append(label)

        self.train_ids = {DOC_NODE_TYPE: train_ids}
        self.valid_src_ids = {DOC_NODE_TYPE: valid_src_ids}
        self.valid_ids = {DOC_NODE_TYPE: valid_ids}
        self.test_ids = {DOC_NODE_TYPE: test_ids}

    def _generate_trans_doc(self):
        self.trans_dic = {self.src_lan: {}, self.tgt_lan: {}}
        for mode in ["train", "test", "eval"]:
            for trans_pair in [(self.src_lan, self.tgt_lan), (self.tgt_lan, self.src_lan)]:
#            for trans_pair in [(self.src_lan, self.tgt_lan)]:
                trans_dic = self.trans_dic[trans_pair[1]]
                trans_file = os.path.join(self.trans_cache_path, f"{mode}_{trans_pair[0]}-{trans_pair[1]}.txt")
                if os.path.exists(trans_file):
                    data = pd.read_csv(trans_file, sep='\t', header=None, names=['raw', 'trans'], quoting=3)
                    data = data.dropna(axis=0, how='any')
                    data = data[~data['trans'].isin(['unsuccessful-trans'])]
                    trans_dic_raw = data[["trans", "raw"]].set_index("trans").to_dict()["raw"]
                    for trans_key in tqdm(trans_dic_raw, ncols=80):
                        trans_dic[trans_key] = trans_dic_raw[trans_key]

    def augment_dic(self):
        for tg in self.trans_dic[self.src_lan]:
            self.dedup_source_text.add(tg)
        for tg in self.trans_dic[self.tgt_lan]:
            self.dedup_target_text.add(tg)
        # add translated texts to labels
        if self.trans_labels:
            source_label_dic = {}
            for text, label in zip(self.source_train_set + self.source_valid_set,
                                self.source_train_label + self.source_valid_label):
                source_label_dic[text] = label
            for tg, src in self.trans_dic[self.tgt_lan].items():
                if src in source_label_dic:
                    self.source_train_set.append(tg)
                    self.source_train_label.append(int(source_label_dic[src]))

    def generate_bilingual_word_node(self):
        logger.info("Start generating word node.")

        self.pos_tags = {}
        def single_lan_gen(dedup_text, lang):
            doc_node_cnt = len(self.bilingual_corpus2id)
            tagger = PosTagger(lang, self.tagger_root_path)
            pos_tag_results = tagger.pos_tag_sents(dedup_text)

            exclude_list = PUNCTUATION
            if LANG_NAME[lang] in stopwords.fileids():
                exclude_list += stopwords.words(LANG_NAME[lang])
            
            # tokenize corpus, count words tf and df
            word_count, word_df = {}, {}
            corpus_word_list = []
            for item in pos_tag_results:
                sentence = item["text"]
                tokens = []
                for token, pos_t in item["res"]:
                    if token not in exclude_list and pos_t in POS_TAG_LIST:
                        # add to vocab
                        tokens.append(token)
                        word_count[token] = word_count.get(token, 0) + 1
                for token in set(tokens):
                    word_df[token] = word_df.get(token, 0) + 1
                doc_id = self.bilingual_corpus2id[sentence]
                self.pos_tags[doc_id] = dict(item["res"])
                corpus_word_list.append((doc_id, tokens))

            #word_corpus_tfidf = [(word_count[w] / word_df[w], w)
            #                     if word_count[w] > 0 else (0, w) for w in word_count]
            word_corpus_tfidf = [(word_count[w], w) for w in word_count]
            word_corpus_tfidf = sorted(word_corpus_tfidf, reverse=True)

            selected_word_node = set()
            for score, word in word_corpus_tfidf:
                if (score > 0) and len(selected_word_node) < self.word_node_cnt:
                    selected_word_node.add(word)
                else:
                    break
            return selected_word_node, corpus_word_list, word_df        

        self.source_word_node, self.source_word_list, self.source_word_df = single_lan_gen(
            self.dedup_source_text, self.src_lan)
        self.target_word_node, self.target_word_list, self.target_word_df = single_lan_gen(
            self.dedup_target_text, self.tgt_lan)

        self.bilingual_map = {}
        self.all_word_node = list(self.source_word_node | self.target_word_node)
        self.vocab_dic = {self.src_lan: {}, self.tgt_lan: {}}
        self.ivocab_dic = {}
        for idx, word in enumerate(self.source_word_node):
            self.vocab_dic[self.src_lan][word] = idx
            self.ivocab_dic[idx] = word
        for idx, word in enumerate(self.target_word_node, start=len(self.source_word_node)):
            self.vocab_dic[self.tgt_lan][word] = idx
            self.ivocab_dic[idx] = word

    def _merge_word_nodes(self):
        """
        This function merges word nodes given the bilingual dictionary. For all the target words in the bilingual dictionary,
        they are merged to words in source language.
        """
        self.doc_node_cnt = len(self.bilingual_corpus2id)
        self.id_offset = 0

        self.id_transform = {}
        vocab_old_ids = []
        for i, word in enumerate(self.all_word_node):
            # re-id words in source/target language
            if (word not in self.source_word_node) and (word in self.bilingual_map):
                continue
            if word in self.source_word_node:
                word_new_id = len(self.id_transform)
                self.id_transform[self.vocab_dic[self.src_lan][word]] = self.id_offset + word_new_id
                vocab_old_ids.append(i)
            if word in self.target_word_node:
                word_new_id = len(self.id_transform)
                self.id_transform[self.vocab_dic[self.tgt_lan][word]] = self.id_offset + word_new_id
                vocab_old_ids.append(i)

        # re-order word features
        self.word_embeddings = self.word_embeddings[vocab_old_ids]

    def get_ids(self, mode: str) -> List[int]:
        if mode == "train":
            return self.train_ids
        elif mode == "valid_same_lang":
            return self.valid_src_ids
        elif mode == "valid":
            return self.valid_ids
        else:
            return self.test_ids

    def get_all_labels(self):
        return self.g.nodes[DOC_NODE_TYPE].data["label"]

    def _process_word_doc_edges(self) -> List[Tuple[int, int, float]]:
        """
        Add words with highest corpus tf-idf as word nodes, create word -> doc edges.
        """
        logger.info("Start processing word-doc edges.")

        edges = []
        corpus_word_list = []
        corpus_word_list.extend(self.source_word_list)
        corpus_word_list.extend(self.target_word_list)
        for word_list, lang in zip([self.source_word_list, self.target_word_list], [self.src_lan, self.tgt_lan]):
            for doc_id, tokens in word_list:
                pos_dic = self.pos_tags[doc_id]
                for token in set(tokens):
                    if token in self.vocab_dic[lang]:
                        pos_t = pos_dic.get(token, "X")
                        if pos_t in POS_TAG_LIST:
                            # only keep meaningful words
                            if self.no_pos_tag:
                                pos_t = "word-doc"
                            if (lang == self.tgt_lan) and (token in self.bilingual_map):
                                for src_word in self.bilingual_map[token]:
                                    src_word_id = self.vocab_dic[self.src_lan][src_word]
                                    edges.append((self.id_transform[src_word_id], doc_id, pos_t))
                            else:
                                tid = self.vocab_dic[lang][token]
                                edges.append((self.id_transform[tid], doc_id, pos_t))
        return edges

    def _process_doc_doc_edges(self) -> List[Tuple[int, int, str]]:
        """build edges between bilingual sentences"""
        doc_doc_edge = {DOC_SIM_EDGE_TYPE: [], DOC_TRANS_EDGE_TYPE: []}
        sim_edges = doc_doc_edge[DOC_SIM_EDGE_TYPE]
        trans_edges = doc_doc_edge[DOC_TRANS_EDGE_TYPE]

        # fetch similarity edges
        if self.sim_edges:
            if os.path.exists(self.doc_doc_edge_file):
                with open(self.doc_doc_edge_file, 'r', encoding='UTF-8') as f:
                    for line in f:
                        seg = line.rstrip().split("\t")
                        sim_edges.append((int(seg[0]), int(seg[1])))
            else:
                source_len = len(self.dedup_source_text)
                logger.info(source_len)
                # for index in tqdm(range(source_len,len(self.bilingual_corpus2id)),ncols=80):
                text_embeddings_tensor = torch.Tensor(self.text_embeddings).to(self.device)
                for index in tqdm(range(len(self.bilingual_corpus2id)), desc="building sim edges", ncols=80):
                    vals = torch.nn.functional.cosine_similarity(
                        text_embeddings_tensor[index: index + 1], text_embeddings_tensor)
                    idx = torch.topk(vals, k=4)[1].cpu().numpy().tolist()[1:]
                    for i in idx:
                        sim_edges.append((index, i))
                with open(self.doc_doc_edge_file, 'w', encoding='UTF-8') as f:
                    for n1, n2 in sim_edges:
                        f.write(str(n1))
                        f.write('\t')
                        f.write(str(n2))
                        f.write('\n')

        # build translated edges
        if self.trans_edges:
            for trans_dic in self.trans_dic.values():
                for key, value in trans_dic.items():
                    trans_edges.append((self.bilingual_corpus2id[key],self.bilingual_corpus2id[value]))
        return doc_doc_edge

    def _build_mask(self, idx, shape):
        mask = np.zeros(shape)
        mask[idx] = 1
        return mask

    def _process_mbert_features(self):
        logging.info("Start processing pretrained features")
        encoder = mbert_ft.TransformerEncoder(self.mbert_feature_cache, self.mbert_model_path, self.source_path,
                                              self.transformer_type, self.device, self.encode_maxlen)
        all_texts = [self.ibilingual_corpus2id[i] for i in range(len(self.bilingual_corpus2id))]
        all_texts += self.all_word_node
        all_features = encoder.embed_sentences(all_texts)
        self.text_embeddings = all_features[0: len(self.bilingual_corpus2id)]
        self.word_embeddings = all_features[len(self.bilingual_corpus2id):]

    def _check_edge(self, graph):
        #zero_edge = []
        index_tensor = torch.nonzero((graph.in_degrees() == 0))
        index_list = index_tensor.view(-1).numpy().tolist()
        for index in tqdm(index_list, ncols=80):
            vals = cosine_similarity([self.text_embeddings[index]], self.text_embeddings)
            idx = vals.argsort()[0][-4:-1]
            for i in idx:
                graph.add_edges([i, idx], [idx, i], etype=DOC_SIM_EDGE_TYPE)
        return graph

    def _process_hetero_graph(self):
        # build graph
        self.data_source_read()
        self.data_target_read()
        #self.muse_generate_vocab_dic()
        if self.trans_edges:
            self._generate_trans_doc()
            self.augment_dic()
        self.generate_corpus2id()
        self.data_generate_train_val_test()
        self.generate_bilingual_word_node()

        self._process_mbert_features()
        self._merge_word_nodes()

        edges = {}
        doc_doc_edges = self._process_doc_doc_edges()
        for e_type, doc_etype_edges in doc_doc_edges.items():
            if not doc_etype_edges:
                continue
            doc_doc_edges_n1 = [e[0] for e in doc_etype_edges]
            doc_doc_edges_n2 = [e[1] for e in doc_etype_edges]
            doc_doc_edges_src = doc_doc_edges_n1 + doc_doc_edges_n2
            doc_doc_edges_dst = doc_doc_edges_n2 + doc_doc_edges_n1
            edges[(DOC_NODE_TYPE, e_type, DOC_NODE_TYPE)] = (doc_doc_edges_src, doc_doc_edges_dst)

        word_doc_edges = self._process_word_doc_edges()
        edge_type_dic = {}
        for word_id, doc_id, edge_type in word_doc_edges:
            if edge_type not in edge_type_dic:
                edge_type_dic[edge_type] = ([word_id], [doc_id])
            else:
                edge_type_dic[edge_type][0].append(word_id)
                edge_type_dic[edge_type][1].append(doc_id)
        for e_type in edge_type_dic.keys():
            edges_n1 = edge_type_dic[e_type][0]
            edges_n2 = edge_type_dic[e_type][1]
            edges[(DOC_NODE_TYPE, e_type, WORD_NODE_TYPE)] = (edges_n2, edges_n1)
            edges[(WORD_NODE_TYPE, e_type, DOC_NODE_TYPE)] = (edges_n1, edges_n2)

        g = dgl.heterograph(edges)
        #g = self._check_edge(g)
        logging.info(f"Graph created with {g.num_nodes()} nodes and {g.num_edges()} edges.")

        # features
        #doc_features, word_features = self._process_features()
        g.nodes[DOC_NODE_TYPE].data['feat'] = torch.FloatTensor(self.text_embeddings)
        g.nodes[WORD_NODE_TYPE].data['feat'] = torch.FloatTensor(self.word_embeddings)

        # processing masks and labels
        train_mask = self._build_mask(self.train_ids[DOC_NODE_TYPE], self.doc_node_cnt)
        valid_src_mask = self._build_mask(self.valid_src_ids[DOC_NODE_TYPE], self.doc_node_cnt)
        valid_mask = self._build_mask(self.valid_ids[DOC_NODE_TYPE], self.doc_node_cnt)
        test_mask = self._build_mask(self.test_ids[DOC_NODE_TYPE], self.doc_node_cnt)
        labels = np.zeros(self.doc_node_cnt)
        labels[self.train_ids[DOC_NODE_TYPE]] = self.train_labels
        labels[self.valid_src_ids[DOC_NODE_TYPE]] = self.valid_src_labels
        labels[self.valid_ids[DOC_NODE_TYPE]] = self.valid_labels
        labels[self.test_ids[DOC_NODE_TYPE]] = self.test_labels

        # splitting masks
        g.nodes[DOC_NODE_TYPE].data['train_mask'] = generate_mask_tensor(train_mask)
        g.nodes[DOC_NODE_TYPE].data['valid_src_mask'] = generate_mask_tensor(valid_src_mask)
        g.nodes[DOC_NODE_TYPE].data['valid_mask'] = generate_mask_tensor(valid_mask)
        g.nodes[DOC_NODE_TYPE].data['test_mask'] = generate_mask_tensor(test_mask)
        # node labels
        g.nodes[DOC_NODE_TYPE].data['label'] = torch.LongTensor(labels)
        return g

    def process(self):
        self.g = self._process_hetero_graph()
        logging.info("Process finished.")

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.g

    def __len__(self):
        return 1

    def save(self):
        save_graphs(self.graph_path, self.g)
        save_info(self.info_path, {'text': self.bilingual_corpus2id,
                                   "doc_cnt": self.doc_node_cnt,
                                   "train_ids": self.train_ids,
                                   "valid_src_ids": self.valid_src_ids,
                                   "valid_ids": self.valid_ids,
                                   "test_ids": self.test_ids,
                                   "train_labels": self.train_labels,
                                   "valid_src_labels": self.valid_src_labels,
                                   "valid_labels": self.valid_labels,
                                   "test_labels": self.test_labels})
        logger.info("Processed graph saved to disk.")

    def load(self):
        g, __ = load_graphs(self.graph_path)
        self.g = g[0]
        self.info = load_info(self.info_path)
        self.bilingual_corpus2id = self.info['text']
        self.ibilingual_corpus2id = {v: k for k, v in self.bilingual_corpus2id.items()}
        self.doc_node_cnt = self.info["doc_cnt"]
        self.word_node_cnt = self.g.num_nodes() - self.doc_node_cnt
        self.train_ids = self.info["train_ids"]
        self.valid_src_ids = self.info["valid_src_ids"]
        self.valid_ids = self.info["valid_ids"]
        self.test_ids = self.info["test_ids"]
        self.train_labels = self.info["train_labels"]
        self.valid_src_labels = self.info["valid_src_labels"]
        self.valid_labels = self.info["valid_labels"]
        self.test_labels = self.info["test_labels"]
        logger.info(f"Processed graph loaded {self.g.num_nodes()} nodes and {self.g.num_edges()} edges.")

    def has_cache(self):
        # check whether there are processed data in `self.output_path`
        #print(os.path.exists(self.graph_path) and os.path.exists(self.info_path))
        return os.path.exists(self.graph_path) and os.path.exists(self.info_path)

