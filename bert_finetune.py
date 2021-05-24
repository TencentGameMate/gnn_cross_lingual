
import os
import time
import datetime
import random
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (AdamW, \
        BertForSequenceClassification, BertTokenizer, \
        XLMForSequenceClassification, XLMTokenizer, \
        XLMRobertaForSequenceClassification, XLMRobertaTokenizer)
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

BERT_MODEL = "bert-base-multilingual-uncased"
XLM_MODEL = "xlm-mlm-17-1280"
XLMR_MODEL = "xlm-roberta-base"

def num_correct(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_model_cls(model_type):
    if model_type == "bert":
        return BERT_MODEL, BertTokenizer, BertForSequenceClassification
    elif model_type == "xlm":
        return XLM_MODEL, XLMTokenizer, XLMForSequenceClassification
    elif model_type == "xlmr":
        return XLMR_MODEL, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
    else:
        raise NameError(f"Unsupported model {model_type}")

def finetune_mode(file_path, save_path, model="bert", device="cpu",
                  batch_size=32, num_labels=2, lr=2e-5, epochs=4, max_length=128,
                  use_trans_label=False, trans_path=None, src_lang=None, tgt_lang=None):
    print('Loading BERT tokenizer...')
    model_name, tokenizer_cls, model_cls = get_model_cls(model)

    tokenizer = tokenizer_cls.from_pretrained(model_name, do_lower_case=(model == "bert"))
    
    model_save_path = os.path.join(save_path, model_name)
    #if False:
    if os.path.exists(model_save_path):
        model = model_cls.from_pretrained(
            model_save_path,
            output_attentions = False,
            output_hidden_states = True
        )
        model.to(device)
        return tokenizer, model

    def load_translate(trans_cache_path, src_lan, tgt_lan):
        import pandas as pd
        import nltk
        trans_dic = {}
        for mode in ["train", "test"]:
            trans_pair = (src_lan, tgt_lan)
#            for trans_pair in [(self.src_lan, self.tgt_lan)]:
            trans_file = os.path.join(trans_cache_path, f"{mode}_{trans_pair[0]}-{trans_pair[1]}.txt")
            data = pd.read_csv(trans_file, sep='\t', header=None, names=['raw', 'trans'], quoting=3)
            data = data.dropna(axis=0, how='any')
            data = data[~data['trans'].isin(['unsuccessful-trans'])]
            trans_dic_raw = data[["trans", "raw"]].set_index("trans").to_dict()["raw"]
            for trans_key in tqdm(trans_dic_raw, ncols=80):
                trans_dic[trans_dic_raw[trans_key]] = trans_key
        return trans_dic
    if use_trans_label:
        trans_dic = load_translate(trans_path, src_lang, tgt_lang)

    def read_data(file_name, dedup=True, add_trans=False):
        dedup_set = set()
        data, labels = [], []
        with open(file_name, 'r', encoding='UTF-8') as fin:
            for line in fin:
                sentence = line.rstrip()
                seg = sentence.split('\t')
                if (not dedup) or (seg[1] not in dedup_set):
                    data.append(seg[1])
                    labels.append(int(seg[0]))
                    if add_trans:
                        data.append(trans_dic[seg[1]])
                        labels.append(int(seg[0]))
                    dedup_set.add(seg[1])
        encoded_dict = tokenizer.batch_encode_plus(
            data,                           
            add_special_tokens = True,      
            max_length = max_length,        
            pad_to_max_length = True,
            return_attention_mask = True,   
            return_tensors = 'pt',          
        )
        return TensorDataset(encoded_dict["input_ids"], encoded_dict["attention_mask"], torch.LongTensor(labels))

    train_dataset = read_data(os.path.join(file_path, "train.txt"), add_trans=use_trans_label)
    val_dataset = read_data(os.path.join(file_path, "test.txt"), False)

    print('{:>5,} training samples'.format(len(train_dataset)))
    print('{:>5,} validation samples'.format(len(val_dataset)))

    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size 
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size 
            )


    model = model_cls.from_pretrained(
        model_name, 
        num_labels = num_labels, 
        output_attentions = False,
        output_hidden_states = True,
        hidden_dropout_prob = 0.15,
        attention_probs_dropout_prob = 0.15,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr = lr,
                      eps = 1e-8
                    )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
 
    training_stats = []

    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            res = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            loss = res.loss
            logits = res.logits

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_correct = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                res = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels)
                loss = res.loss
                logits = res.logits
                
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_correct += num_correct(logits, label_ids)
            

        avg_val_accuracy = total_eval_correct / len(val_dataset)
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        
    model.save_pretrained(model_save_path)

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
    # Save Model
    #model.save_pretrained(model_save_path)

    return tokenizer, model

def split_batch(sentences, batch_size):
    batches = []
    for i, sent in enumerate(sentences):
        if i % batch_size == 0:
            batches.append([])
        batches[-1].append(sent)
    return batches

def embed_sentences(sentences, tokenizer, model, batch_size=128, encode_length=512):
    model.eval()
    results = []
    
    sentences = split_batch(sentences, batch_size)
    for batch in tqdm(sentences, desc="mbert encoding"):
        encoded_dict = tokenizer.batch_encode_plus(
                            batch,                      
                            add_special_tokens = True, 
                            max_length = encode_length,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
                       )
        with torch.no_grad():
            res = model(
                encoded_dict["input_ids"].to(model.device),
                attention_mask=encoded_dict["attention_mask"].to(model.device)
            )
            cls_hidden_state = res.hidden_states[-1][:, 0, :] 
            results.append(cls_hidden_state.cpu().numpy())
    return np.concatenate(results, axis=0)

class TransformerEncoder:
    def __init__(self, cache_path, model_path, train_file_path, transformer_type, device="cpu", encode_maxlen=512):
        self.cache_path = cache_path
        self.model_path = model_path
        self.train_file_path = train_file_path
        self.transformer_type = transformer_type
        self.device = device
        self.encode_length = encode_maxlen

        text_emb_files = {}
        text_catalog = {}
        for fname in os.listdir(self.cache_path):
            if fname.endswith(".cat"):
                text_catalog[fname[:-4]] = []
                with open(os.path.join(self.cache_path, fname)) as fin:
                    for line in fin.read().splitlines():
                        text_emb_files[line] = fname[:-4]
                        text_catalog[fname[:-4]].append(line)
        self.text_emb_files = text_emb_files
        self.text_catalog = text_catalog

    def _save_cache(self, texts, embeddings):
        save_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        i = 0
        while (i * 10000) < len(texts):
            chunk_texts = texts[i * 10000: (i + 1) * 10000]
            chunk_embeddings = embeddings[i * 10000: (i + 1) * 10000]
            catalog_path = os.path.join(self.cache_path, f"{save_time}_{i}.cat")
            emb_path = os.path.join(self.cache_path, f"{save_time}_{i}.npy")
            np.save(emb_path, chunk_embeddings)
            with open(catalog_path, "w") as fout:
                fout.write("\n".join(chunk_texts))
            i += 1
    
    def embed_sentences(self, texts):
        cache_files = set()
        sentences_to_encode = []
        for text in texts:
            if text in self.text_emb_files:
                cache_files.add(self.text_emb_files[text])
            else:
                sentences_to_encode.append(text)
        
        whole_embedding_matrix = []
        text_id_map = {}
        text_id = 0
        if cache_files:
            for file_name in cache_files:
                for text in self.text_catalog[file_name]:
                    text_id_map[text] = text_id
                    text_id += 1
                whole_embedding_matrix.append(np.load(os.path.join(self.cache_path, file_name + ".npy")))

        if sentences_to_encode:
            tokenizer, model = finetune_mode(
                self.train_file_path, self.model_path, self.transformer_type, self.device)
            embeddings = embed_sentences(sentences_to_encode, tokenizer, model,
                                         batch_size=128, encode_length=self.encode_length)
            self._save_cache(sentences_to_encode, embeddings)
            whole_embedding_matrix.append(embeddings)
            for text in sentences_to_encode:
                text_id_map[text] = text_id
                text_id += 1

        whole_embedding_matrix = np.concatenate(whole_embedding_matrix, axis=0)
        id_transform = []
        for text in texts:
            id_transform.append(text_id_map[text])
        return whole_embedding_matrix[id_transform]
