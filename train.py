import os
import sys
import time
import argparse
import logging

import dgl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from cross_lingual import MQAGraphDataset, DOC_NODE_TYPE
from hetero_textgcn import HeteroTextGCN

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logging.root.addHandler(handler)
logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, dataset, nids, save_badcase=False):
    g = dataset.g
    labels = dataset.get_all_labels()
    model.eval()
    with torch.no_grad():
        __, pred = model.inference(g)
    model.train()
    accs = {}
    for nid_key, nid in nids.items():
        accs[nid_key] = compute_acc(pred[nid], labels[nid])
    if save_badcase:
        with open("badcase.txt", "w") as fout:
            print("pred\ttrue\ttext", file=fout)
            pred_labels = torch.argmax(pred[nids["valid"]], dim=1)
            for pred_label, label, idx in zip(pred_labels, labels[nids["valid"]], nids["valid"]):
                if pred_label != label:
                    text = dataset.ibilingual_corpus2id[idx]
                    print(pred_label.item(), label.item(), text, file=fout, sep="\t")
    return accs

def train(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    file_log_handler = logging.FileHandler(args.log_path)
    file_log_handler.setLevel(logging.DEBUG)
    logging.root.addHandler(file_log_handler)
    logger.info(args)

    dataset = MQAGraphDataset(args)
    logger.info(dataset.g)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    dataloader = dgl.dataloading.NodeDataLoader(
        dataset.g, dataset.get_ids("train"), sampler,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    model = HeteroTextGCN(args, dataset.g.etypes, dataset)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fcn = nn.CrossEntropyLoss().to(args.device)
    model_size = args.emb_size
    noam_lambda = lambda step: (
            1000 * (model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5))))
    noam_scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)

    num_train = len(dataset.get_ids("train")[DOC_NODE_TYPE])
    logger.info(f"Training starts with {num_train} instances.")
    
    iter_tput = []
    best_accs = [0, 0]
    for epoch in range(args.num_epochs):

        tic_step = time.time()
        for step, (__, __, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            blocks = [block.int().to(args.device) for block in blocks]
            batch_inputs = blocks[0].srcdata['feat']
            batch_labels = blocks[-1].dstdata['label']
            if isinstance(batch_labels, dict):
                batch_labels = batch_labels[DOC_NODE_TYPE]
            if not isinstance(batch_inputs, dict):
                batch_inputs = {DOC_NODE_TYPE: batch_inputs}

            # Compute loss and prediction
            __, batch_pred = model(dataset.g, blocks, batch_inputs)
            #print(batch_pred.size())
            #print(batch_labels.size())
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            noam_scheduler.step()

            iter_tput.append(len(batch_labels) / (time.time() - tic_step))
            if (step + 1) % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                logger.info(f"Epoch {epoch} | Step {step + 1:02d} | lr {optimizer.param_groups[0]['lr']:.2e}"
                            f" | Loss {loss.item():.4f} | Train Acc {acc.item():.4f}"
                            f" | Speed (samples/sec) {np.mean(iter_tput[-args.log_every:]):.4f}")
            tic_step = time.time()
        
        # evaluate after each epoch
        accs = evaluate(model, dataset, {
                            "train": dataset.get_ids("train")[DOC_NODE_TYPE],
                            "valid_same_lang": dataset.get_ids("valid_same_lang")[DOC_NODE_TYPE],
                            "valid": dataset.get_ids("valid")[DOC_NODE_TYPE],
                            "test": dataset.get_ids("test")[DOC_NODE_TYPE],
                        },
                        save_badcase=False)
        logger.info(f"Train Acc {accs['train']:.4f}, Valid acc (source lang): {accs['valid_same_lang']:.4f}")
        logger.info(f"Valid Acc {accs['valid']:.4f}, Test Acc {accs['test']:.4f}")
        if accs["valid"] >= best_accs[0]:
            best_accs[0] = accs['valid']
            best_accs[1] = accs['test']

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, os.path.join(args.output_path, f"checkpoint_epoch{epoch + 1}.pt"))

    logger.info("Training finished.")
    logger.info("Best acc for valid: {}, best acc for test: {}".format(*best_accs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #data args
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source language text")
    parser.add_argument("--target_path", type=str, required=True, help="Path to the target language text")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the log file")
    parser.add_argument("--tagger_root_path", type=str, required=True, help="Path to the POS tagger models")
    parser.add_argument("--src_lan", type=str, required=True, help="Source language")
    parser.add_argument("--tgt_lan", type=str, required=True, help="Target language")
    parser.add_argument("--data_cache_path", type=str, required=True, help="Path to cache the processed training data")
    parser.add_argument("--trans_cache_path", type=str, required=True, help="Path where translation files are stored")
    parser.add_argument("--doc_doc_edge_stored", action="store_true", help="Whether need to create doc-doc egdes")
    parser.add_argument("--doc_doc_edge_file", type=str, required=True, help="Path to the doc-doc edges")
    parser.add_argument("--transformer_type", type=str, default="bert", help="Transformer type (bert/xlm/xlmr)")
    parser.add_argument("--mbert_feature_file", type=str, default="", help="Path to mbert features")
    parser.add_argument("--mbert_model_path", type=str, required=True, help="Save path of mbert model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the model output")
    parser.add_argument("--encode_maxlen", type=int, default=512, help="Max length for transformer encoder")
    parser.add_argument("--word_node_cnt", type=int, default=10000, help="Number of word nodes in the graph")
    parser.add_argument("--no_sim_edges", action="store_false", dest="sim_edges", help="Do not add similar edges")
    parser.add_argument("--no_trans_edges", action="store_false", dest="trans_edges", help="Do not add trans edges")
    parser.add_argument("--no_unlabeled", action="store_false", dest="use_unlabeled", help="Do not use unlabeled data")
    parser.add_argument("--no_pos_tag", action="store_true", dest="no_pos_tag", help="Do not separate edges using POS tags")
    parser.add_argument("--merge_word", action="store_true", help="Merge word nodes")
    parser.add_argument("--trans_labels", action="store_true", help="Add translated labels")
    # model args
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size")
    parser.add_argument("--emb_size", type=int, default=768, help="Embedding size")
    parser.add_argument("--out_emb_size", type=int, default=512, help="Embedding size")
    parser.add_argument("--num_layers", type=int, default=3, help="Layers")
    # train args
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=float, default=500, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_epochs', type=int, default=100, help="Total number of epochs to train the model")
    parser.add_argument("--log_every", type=int, default=10, help="Number of iterations between each log")
    parser.add_argument("--eval_every", type=int, default=2, help="Number of epochs when evaluating the model")
    parser.add_argument("--save_every", type=int, default=20, help="Number of epochs when saving the model")
    parser.add_argument("--train_batch_size", type=int, default=1024, help="Batch size for train")
    parser.add_argument("--valid_batch_size", type=int, default=1024, help="Batch size for validation")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout")

    args = parser.parse_args()
    train(args)
