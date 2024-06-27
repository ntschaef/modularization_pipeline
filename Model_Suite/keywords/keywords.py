"""
    Module for adding keywords into DL models.
"""
import json
import random
import torch

import numpy as np


class Keywords():
    """Class for adding keywords and comututing keyword loss.

        Args:
            tasks - list: list of tasks for keywords,
                only set up for: [site, subsite, laterality, histology, behavior]
            id2label - dict: mapping from int to taks label
            id2word - dict: vocab mapping
            device - torch.device: either gpu or cpu, depending on environment
    """
    def __init__(self, tasks: list, id2word: dict, id2label: dict, device: torch.device):
        self.tasks = tasks
        self.id2label = id2label
        self.id2word = id2word
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.device = device
        self.Xs = {}
        self.Ys = {}

    def load_keyword_lists(self):
        """Load dicts of task specific keywords and labels from disk.

        Post-condition:
            Class attributers set.
            Xs - dict: keyword dict mapping
            Ys - dict: keyword label mapping


        """

        word2id = {v: k for k, v in self.id2word.items()}

        for task in self.tasks:
            with open(f'./keywords/{task}_cuis.json', 'r') as f_in:
                kw = json.load(f_in)
            label2id = {v: k for k, v in self.id2label[task].items()}
            X = []
            Y = []
            for label, word_lists in kw.items():
                if label not in label2id:
                    continue

                keyword_id_lists = []

                for word_list in word_lists:
                    id_list = [word2id[w] for w in word_list if w in word2id]
                    if len(id_list) == len(word_list):
                        keyword_id_lists.append(id_list)

                if len(keyword_id_lists) > 0:
                    Y.append(label2id[label])
                    X.append(keyword_id_lists)

            self.Xs[task] = X
            self.Ys[task] = Y

    def keywords_loss(self, model, model_args, alpha=1.0, k=5):
        """Compute keyword loss.

            Args:
                model :torch.nn model (hisan or cnn)
                model_args - dict: model kwargs from model_args.yml
                alpha - float: keyword loss scale
                k - int: ???
        """

        loss = 0

        for t, task in enumerate(self.tasks):
            XY = list(zip(self.Xs[task], self.Ys[task]))
            random.shuffle(XY)
            X_, Y = zip(*XY)
            X_ = X_[:model_args['batch_per_gpu']]
            Y = Y[:model_args['batch_per_gpu']]
            X = np.zeros((len(X_), model_args['max_doc_len']), dtype=np.int32)

            for i, x in enumerate(X_):
                random.shuffle(x)
                flattened_x = np.concatenate(x[:k]).astype(np.int32)
                keyword_len = min(model_args['max_doc_len'], len(flattened_x))
                X[i, :keyword_len] = flattened_x[:keyword_len]

            X = torch.tensor(X, dtype=torch.long).to(self.device)
            Y = torch.tensor(Y, dtype=torch.long).to(self.device)

            logits = model.forward(X)[t]
            loss += self.loss_fct(logits, Y)

        loss /= len(self.tasks)
        loss *= alpha
        return loss
