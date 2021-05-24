#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate bert finetune model
"""

import os
import argparse
import torch
from tqdm import tqdm
import bert_finetune as mbert_ft


def load_eval_data(target_test_path):

    eval_data, eval_label = [], []
    eval_file = os.path.join(target_test_path, "test.txt")
    with open(eval_file, 'r', encoding='UTF-8') as fin:
        for line in fin:
            sentence = line.rstrip()
            seg = sentence.split('\t')
            eval_data.append(seg[1])
            eval_label.append(int(seg[0]))
    return eval_data, eval_label


def finetune_model(args):
    tokenizer, mbert_model = mbert_ft.finetune_mode(args.train_path, args.model_save_path, args.transformer_type, args.device,
                        batch_size=args.batch_size, num_labels=args.num_classes, lr=args.lr, epochs=args.epoch_num, max_length=args.max_length,
                        use_trans_label=args.trans_labels, trans_path=args.trans_cache_path, src_lang=args.src_lan, tgt_lang=args.tgt_lan)

    return tokenizer, mbert_model


def eval_bert_ft(tokenizer, mbert_model, test_data_path, max_length=128, bad_case_out_file=None):
    eval_data, eval_label = load_eval_data(test_data_path)

    mbert_model.eval()
    pred_labels = []
    
    sentences = mbert_ft.split_batch(eval_data, 128)
    for batch in tqdm(sentences, desc="mbert encoding"):
        encoded_dict = tokenizer.batch_encode_plus(
                            batch,                      
                            add_special_tokens = True, 
                            max_length = max_length,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
                       )
        with torch.no_grad():
            res = mbert_model(
                encoded_dict["input_ids"].to(mbert_model.device),
                attention_mask=encoded_dict["attention_mask"].to(mbert_model.device)
            )
            pred_labels += torch.argmax(res.logits, dim=1).cpu().numpy().tolist()
    
    if bad_case_out_file:
        bad_case_out = open(bad_case_out_file, "w", encoding="utf-8")

    correct_num = 0
    for (sent, pred, true) in zip(eval_data, pred_labels, eval_label):
        if pred == true:
            correct_num += 1
        elif bad_case_out_file:
            print(f"{pred}\t{true}\t{sent}", file=bad_case_out)
    print(f"Acc: {correct_num / len(eval_data):.4f}")

    if bad_case_out_file:
        bad_case_out.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True, type=str)
    parser.add_argument("--test_path", required=True, type=str)
    parser.add_argument("--transformer_type", required=True, type=str)
    parser.add_argument("--model_save_path", required=True, type=str)
    parser.add_argument("--bad_case_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--src_lan", type=str, required=True, help="Source language")
    parser.add_argument("--tgt_lan", type=str, required=True, help="Target language")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--epoch_num", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--trans_labels", action="store_true", help="Add translated labels")
    parser.add_argument("--trans_cache_path", type=str, default="", help="Path where translation files are stored")
    args = parser.parse_args()

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    tokenizer, model = finetune_model(args)
    eval_bert_ft(tokenizer, model, args.test_path, args.max_length, args.bad_case_path)

if __name__ == "__main__":
    main()
