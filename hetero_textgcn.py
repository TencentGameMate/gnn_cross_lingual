#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import dgl.nn as dglnn
import torch
from torch import nn
from torch.nn import functional as F

from cross_lingual import DOC_NODE_TYPE, WORD_NODE_TYPE


class HeteroTextGCN(nn.Module):
    def __init__(self, args, TYPE_LIST, dataset):
        super().__init__()
        self.valid_batch_size = args.valid_batch_size
        self.hidden_size = args.hidden_size
        self.out_emb_size = args.out_emb_size
        self.num_classes = args.num_classes
        self.device = args.device
        self.num_workers = args.num_workers
        n_layers = args.num_layers

        #self.conv1 = dglnn.HeteroGraphConv({
        #    t: dglnn.GraphConv(args.emb_size, args.hidden_size) for t in TYPE_LIST
        #}, aggregate="sum")
        #self.conv2 = dglnn.HeteroGraphConv({
        #    t: dglnn.GraphConv(args.hidden_size, args.out_emb_size) for t in TYPE_LIST
        #}, aggregate="sum")
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            t: dglnn.GraphConv(args.emb_size, args.hidden_size) for t in TYPE_LIST
            }, aggregate="sum"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.HeteroGraphConv({
                t: dglnn.GraphConv(args.hidden_size, args.hidden_size) for t in TYPE_LIST
                }, aggregate="sum"))
        self.layers.append(dglnn.HeteroGraphConv({
            t: dglnn.GraphConv(args.hidden_size, args.out_emb_size) for t in TYPE_LIST
            }, aggregate="sum"))
        self.fc = nn.Linear(args.out_emb_size, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, g, blocks, inputs):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = {t: self.dropout(F.leaky_relu(h[t])) for t in h}
        h = {t: self.dropout(h[t]) for t in h}
        logits = self.fc(h[DOC_NODE_TYPE])
        #logits = h[DOC_NODE_TYPE]
        return h, logits

    def inference(self, g):
        nids = {}
        for ntype in g.ntypes:
            nids[ntype] = torch.arange(g.number_of_nodes(ntype))
        x = g.ndata["feat"]
        if not isinstance(x, dict):
            x = {DOC_NODE_TYPE: x}
        for l, layer in enumerate(self.layers):
            y = {t: torch.zeros(g.number_of_nodes(t), self.hidden_size if l != len(self.layers) - 1 else self.out_emb_size)
                    for t in x}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, nids, sampler,
                batch_size=self.valid_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                if not isinstance(input_nodes, dict):
                    input_nodes = {DOC_NODE_TYPE: input_nodes.type(torch.long)}
                    output_nodes = {DOC_NODE_TYPE: output_nodes.type(torch.long)}
                block = blocks[0]
                block = block.int().to(self.device)
                h = {t: x[t][input_nodes[t]].to(self.device) for t in x}
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = {t: self.dropout(F.leaky_relu(h[t])) for t in h}
                for t in h:
                    y[t][output_nodes[t]] = h[t].cpu()
            x = y

        return y[DOC_NODE_TYPE], self.fc(y[DOC_NODE_TYPE].to(self.device)).cpu()
