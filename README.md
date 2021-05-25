# Cross-lingual Text Classification with Heterogeneous Graph Neural Network

Source code for our ACL-2021 paper: [Cross-lingual Text Classification with Heterogeneous Graph Neural Network](https://arxiv.org/abs/2105.11246).

### Dataset
We use three public datasets, which can be downloaded from the links provided below:
1. Amazon Review: [https://webis.de/data/webis-cls-10.html](https://webis.de/data/webis-cls-10.html)
2. XGLUE News Classification: [https://microsoft.github.io/XGLUE/](https://microsoft.github.io/XGLUE/)
3. Multilingual SLU: [https://fb.me/multilingual_task_oriented_data](https://fb.me/multilingual_task_oriented_data)

We also provide the machine-translated texts we use in our experiments. This can be downloaded here: [Google Drive](https://drive.google.com/file/d/1Zu19AX--Ae-I8QsQYMTPbdgfMHOhthiN/view?usp=sharing), [腾讯微云](https://share.weiyun.com/4hGa9C44).

### Dependencies
This code requires Python 3, [PyTorch 1.7+](https://pytorch.org), [transformers](https://github.com/huggingface/transformers), and [dgl 0.5+](https://www.dgl.ai) to run. For a detailed list of dependencies, please refer to `requirements.txt`.

For some POS taggers to work, additional libraries are needed:
* English: Please install [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml) and [Java](https://www.java.com/).
* German: Run `python -m spacy download de_core_news_lg`.
* French: Run `python -m spacy download fr_core_news_lg`.
* Spanish: Run `python -m spacy download es_core_news_lg`.
* Russian: Run `python -c "import nltk; nltk.download('averaged_perceptron_tagger_ru')"`

### Running the code
#### Preparing the data

Download the public datasets, and put them in the folders named like this:
`/path/to/data/{lang}/{domain}/train.txt|dev.txt|eval.txt`, where `{lang}` is the language, and `{domain}` is the domain of the data (`books/dvd/music` for Amazon Review, `xglue` for XGLUE News, and `slu` for Multilingual SLU dataset). Each file looks like the following:

```
8     what's the forecast this week ?
10    change my 3 pm alarm to the next day
3     my alarms
3     show my alarms
6     snooze
```

Each line contains a label id and a document, separated by tab (\t).

Then download the translations and put them into a folder (e.g. `./data/translations/`).

#### Finetune the XLM-R baseline
Modify the paths and other arguments in the `run.sh` script, includes:
```
DATA_ROOT:      path to the top level folder of the data
src_lang:       source language (e.g. en)
tgt_lang:       target language (e.g. de)
domain:         domain name (one of 'music', 'dvd', 'books', 'xglue', 'slu')
xfmr:           transformer type (one of 'xlmr', 'xlm', 'bert')
output:         path to save the results and models (e.g. $DATA_ROOT/output/)
translate_data: path to put the downloaded translated texts
pos_cache_path: path to put Stanford POS Tagger
```

Run `./run.sh finetune` to finetune and evaluate the XLM-R baseline.

#### Run the CLHG model
Run `./run.sh train` to run the CLHG model. This will automatically process doc features, construct the graph, train and evaluate the model.

### Citation

If you use our code, please kindly cite our paper.
```
@inproceedings{wang2021cross,
  title={Cross-lingual Text Classification with Heterogeneous Graph Neural Network},
  author={Wang, Ziyun and Liu, Xuan and Yang, Peiji and Liu, Shixing and Wang, Zhisheng},
  booktitle={Proceedings of ACL 2021},
  year={2021}
}
```

### License
```
MIT License

Copyright (c) 2021 Tencent GameMate

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```