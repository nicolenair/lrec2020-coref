import re
import os
from collections import Counter
import sys
import argparse
import pandas as pd

import pytorch_pretrained_bert
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import torch
from torch import nn
from bert_coref import LSTMTagger, read_conll, \
    get_data, get_ant_labels, get_distance_bucket, \
        get_inquote, get_matrix, get_mention_width_bucket, \
        vec_get_distance_bucket
import torch.optim as optim
import numpy as np
import random
import calc_coref_metrics

from torch.optim.lr_scheduler import ExponentialLR

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LSTMTaggerRelate(LSTMTagger):
    def __init__(self, config, freeze_bert=False):
        super(LSTMTaggerRelate, self).__init__(config)
        self.device = "cpu"
        self.predict_relationship_layer = nn.Linear(
            self.hidden_dim * 2, self.hidden_dim * 2)
        self.relation_criterion = nn.CrossEntropyLoss()
        self.char_to_token = {}
        self.relation_maps = {}
        self.annotated_characters = []
        self.load_relation_maps()

    def load_relation_maps(self):
        char_relations = pd.read_csv("data/character_relation_annotations.txt", sep='\t')
        print(char_relations)
        for m in range(len(char_relations)):
            self.relation_maps[
                (char_relations.iloc[m]["character_1"], 
                char_relations.iloc[m]["character_2"])] = char_relations.iloc[m]["fine_category"]
        
        self.annotated_characters = list(set(char_relations["character_1"] + char_relations["character_2"]))
        self.char_to_token = {
            k: v for v, k in enumerate(
                self.annotated_characters)}


    def extract_relation(self, a, b):
        if (a, b) in self.relation_maps:
            return self.relation_maps[(a, b)]
        elif (b, a) in self.relation_maps:
            return self.relation_maps[(b, a)]
        else:
            return None

    def extract_pairs_from_assignments(self, set_of_assignments, input_ids):
        set_of_assignments = [
            i for i in set_of_assignments if i in 
            self.annotated_characters]
        all_pairs = []
        all_labels = []

        #map indices to real characters
        print(set_of_assignments)

        real_assignments = [None for i in range(len(set_of_assignments))]
        for e, m in enumerate(set_of_assignments):
            for c in self.annotated_characters:
                if m == c:
                    cluster = set_of_assignments[e]
                    real_assignments[real_assignments==cluster] = self.char_to_token[c]
        pairs = []
        labels = []
        for e, a in enumerate(real_assignments):
            for f, b in enumerate(real_assignments):
                if a!=None and b!=None:
                    if a != b:
                        r = self.extract_relation(a, b)
                        if r != None:
                            pairs.append(e, f)
                            labels.append(r)

        return pairs, labels

    def relationship_loss(self, 
            span_representation, 
            set_of_assignments):
        pairs, true_labels = self.extract_pairs_from_assignments(
                    set_of_assignments, input_ids)
        losses = 0
        for (m, n), true_label in zip(pairs, true_labels):
            mention_representation = torch.cat(
                [span_representation[m], span_representation[n]])
            scores = self.predict_relationship_layer(mention_representation)
            prediction = torch.nn.functional.softmax(scores)
            loss = self.relation_criterion(prediction, true_label)
            losses+=loss
        return losses
        # _, predicted = torch.max(outputs, 1)

    def forward(self, matrix, index, truth=None, names=None, token_positions=None, starts=None, ends=None, widths=None, input_ids=None, attention_mask=None, transforms=None, quotes=None):

        doTrain = False
        if truth is not None:
            doTrain = True

        zeroTensor = torch.FloatTensor([0]).to(self.device)

        all_starts = None
        all_ends = None

        span_representation = None

        all_all = []
        for b in range(len(matrix)):

            span_reps = self.get_mention_reps(input_ids=input_ids[b], attention_mask=attention_mask[b], starts=starts[b], ends=ends[b],
                                              index=index[b], widths=widths[b], quotes=quotes[b], transforms=transforms[b], matrix=matrix[b], doTrain=doTrain)
            if b == 0:
                span_representation = span_reps
                all_starts = starts[b]
                all_ends = ends[b]

            else:

                span_representation = torch.cat(
                    (span_representation, span_reps), 0)

                all_starts = torch.cat((all_starts, starts[b]), 0)
                all_ends = torch.cat((all_ends, ends[b]), 0)

        all_starts = all_starts.to(self.device)
        all_ends = all_ends.to(self.device)

        num_mentions, = all_starts.shape

        running_loss = 0

        curid = -1

        curid += 1

        assignments = []

        seen = {}

        ch = 0

        token_positions = np.array(token_positions)

        mention_index = np.arange(num_mentions)

        unary_scores = self.unary3(self.tanh(self.drop_layer_020(self.unary2(
            self.tanh(self.drop_layer_020(self.unary1(span_representation)))))))

        for i in range(num_mentions):

            if i == 0:
                # the first mention must start a new entity; this doesn't affect training (since the loss must be 0) so we can skip it.
                # if truth is None:

                assignment = curid
                curid += 1
                assignments.append(assignment)

                continue

            MAX_PREVIOUS_MENTIONS = 300

            first = 0
            # if truth is None or :
            # if len(names[i]) == 1 and names[i][0].lower() in {"he", "his", "her", "she", "him", "they", "their", "them", "it", "himself", "its", "herself", "themselves"}:
            #     MAX_PREVIOUS_MENTIONS = 20

            first = i-MAX_PREVIOUS_MENTIONS
            if first < 0:
                first = 0

            targets = span_representation[first:i]
            cp = span_representation[i].expand_as(targets)

            dists = []
            nesteds = []

            # get distance in mentions
            distances = i-mention_index[first:i]
            dists = vec_get_distance_bucket(distances)
            dists = torch.LongTensor(dists).to(self.device)
            distance_embeds = self.distance_embeddings(dists)

            # get distance in sentences
            sent_distances = token_positions[i]-token_positions[first:i]
            sent_dists = vec_get_distance_bucket(sent_distances)
            sent_dists = torch.LongTensor(sent_dists).to(self.device)
            sent_distance_embeds = self.sent_distance_embeddings(sent_dists)

            # is the current mention nested within a previous one?
            res1 = (all_starts[first:i] <= all_starts[i])
            res2 = (all_ends[i] <= all_ends[first:i])

            nesteds = (res1*res2).long()
            nesteds_embeds = self.nested_embeddings(nesteds)

            res1 = (all_starts[i] <= all_starts[first:i])
            res2 = (all_ends[first:i] <= all_ends[i])

            nesteds = (res1*res2).long()
            nesteds_embeds2 = self.nested_embeddings(nesteds)

            elementwise = cp*targets
            concat = torch.cat((cp, targets, elementwise, distance_embeds,
                                sent_distance_embeds, nesteds_embeds, nesteds_embeds2), 1)

            preds = self.mention_mention3(self.tanh(self.drop_layer_020(self.mention_mention2(
                self.tanh(self.drop_layer_020(self.mention_mention1(concat)))))))

            preds = preds + unary_scores[i] + unary_scores[first:i]

            preds = preds.squeeze(-1)

            if truth is not None:

                # zero is the score for the dummy antecedent/new entity
                preds = torch.cat((preds, zeroTensor))

                golds_sum = 0.
                preds_sum = torch.logsumexp(preds, 0)

                if len(truth[i]) == 1 and truth[i][-1] not in seen:
                    golds_sum = 0.
                    seen[truth[i][-1]] = 1
                else:
                    golds = torch.index_select(
                        preds, 0, torch.LongTensor(truth[i]).to(self.device))
                    golds_sum = torch.logsumexp(golds, 0)

                # want to maximize (golds_sum-preds_sum), so minimize (preds_sum-golds_sum)
                diff = preds_sum-golds_sum

                running_loss += diff

                ########
                arg_sorts = torch.argsort(preds, descending=True)
                k = 0
                while k < len(arg_sorts):
                    cand_idx = arg_sorts[k]
                    if preds[cand_idx] > 0:
                        cand_assignment = assignments[cand_idx+first]
                        assignment = cand_assignment
                        ch += 1
                        break

                    else:
                        assignment = curid
                        curid += 1
                        break

                    k += 1
                assignments.append(assignment)

            else:

                assignment = None

                if i == 0:
                    assignment = curid
                    curid += 1

                else:

                    arg_sorts = torch.argsort(preds, descending=True)
                    k = 0
                    while k < len(arg_sorts):
                        cand_idx = arg_sorts[k]
                        if preds[cand_idx] > 0:
                            cand_assignment = assignments[cand_idx+first]
                            assignment = cand_assignment
                            ch += 1
                            break

                        else:
                            assignment = curid
                            curid += 1
                            break

                        k += 1
                assignments.append(assignment)


        if truth is not None:
            relation_loss = self.relationship_loss( 
                span_representation, 
                assignments)
                    
            running_loss += relation_loss
            return running_loss
        else:
            print(assignments)
            return assignments




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainData',
                        help='Folder containing train data', required=False)
    parser.add_argument('-v', '--valData',
                        help='Folder containing test data', required=False)
    parser.add_argument(
        '-m', '--mode', help='mode {train, predict}', required=False)
    parser.add_argument(
        '-w', '--model', help='modelFile (to write to or read from)', required=False)
    parser.add_argument('-o', '--outFile', help='outFile', required=False)
    parser.add_argument('-s', '--path_to_scorer',
                        help='Path to coreference scorer', required=False)

    args = vars(parser.parse_args())

    mode = args["mode"]
    modelFile = args["model"]
    valData = args["valData"]
    outfile = args["outFile"]
    path_to_scorer = args["path_to_scorer"]
    
    model = LSTMTaggerRelate.from_pretrained('bert-base-cased', freeze_bert=True)

    # import sys
    # sys.exit()
    
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model.to("cpu")


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.999)

    if mode == "train":

        trainData = args["trainData"]

        all_docs, all_ents, all_named_ents, all_truth, all_max_words, all_max_ents, doc_ids = read_conll(
            trainData, model)
        test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, test_all_max_words, test_all_max_ents, test_doc_ids = read_conll(
            valData, model)

        best_f1 = 0.
        cur_steps = 0

        best_idx = 0
        patience = 10

        for i in range(100):

            model.train()
            bigloss = 0.
            for idx in range(len(all_docs)):
                # with open("ug.txt", "a+") as f:
                #     f.write("ok")

                if idx % 10 == 0:
                    print(idx, "/", len(all_docs))
                    sys.stdout.flush()
                max_words = all_max_words[idx]
                max_ents = all_max_ents[idx]

                matrix, index, token_positions, ent_spans, starts, ends, widths, input_ids, masks, transforms, quotes = get_data(
                    model, all_docs[idx], all_ents[idx], max_ents, max_words)

                if max_ents > 1:
                    model.zero_grad()
                    loss = model.forward(matrix, index, truth=all_truth[idx], names=None, token_positions=token_positions, starts=starts,
                                         ends=ends, widths=widths, input_ids=input_ids, attention_mask=masks, transforms=transforms, quotes=quotes)
                    loss.backward()
                    optimizer.step()
                    cur_steps += 1
                    if cur_steps % 100 == 0:
                        lr_scheduler.step()
                bigloss += loss.item()

            print(bigloss)

            model.eval()
            doTest = False
            if i >= 2:
                doTest = True

            avg_f1 = test(model, test_all_docs, test_all_ents, test_all_named_ents, test_all_max_words,
                          test_all_max_ents, test_doc_ids, outfile, i, valData, path_to_scorer, doTest=doTest)


            if doTest:
                if avg_f1 > best_f1:
                    torch.save(model.state_dict(), modelFile)
                    print("Saving model ... %.3f is better than %.3f" %
                          (avg_f1, best_f1))
                    best_f1 = avg_f1
                    best_idx = i

                if i-best_idx > patience:
                    print("Stopping training at epoch %s" % i)
                    break

    elif mode == "predict":

        model.load_state_dict(torch.load(modelFile, map_location=self.device))
        model.eval()
        test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, test_all_max_words, test_all_max_ents, test_doc_ids = read_conll(
            valData, model=model)

        test(model, test_all_docs, test_all_ents, test_all_named_ents, test_all_max_words,
             test_all_max_ents, test_doc_ids, outfile, 0, valData, path_to_scorer, doTest=True)
