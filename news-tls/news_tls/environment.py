import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, BartForConditionalGeneration, BartTokenizer, \
    GPT2Tokenizer, GPT2LMHeadModel

sys.path.append('tacl2018-preference-convincing/python/models/')

from gp_pref_learning import *
from gp_classifier_vb import compute_median_lengthscales # use this function to set sensible values for the lengthscale hyperparameters

class Environment:
    def __init__(self, args, device, keywords=None):
        self.keywords = None
        self.keywords_embeddings = None
        self.args = args
        self.lq_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.lq_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        self.timelines_by_topic = [[] for i in range(19)]
        self.pref_model = None
        self.weights = [100, 0.5, 2.5, 0.1]
        self.device = device
        prefdata_path = Path(args.preference)

        for i in range(4):
            datafile = prefdata_path / 'rl-episode{}.json'.format(i)
            with open(datafile, 'r') as jsonfile:
                timelines = json.load(jsonfile)['results']
            for topic_id, tl in enumerate(timelines):
                self.timelines_by_topic[topic_id].append(tl[2])

    def update_keywords(self, keywords):
        self.keywords = list(keywords)
        self.keywords_embeddings = self.encoder.encode(self.keywords)

    def update_pref(self, topic_id):
        item_data = np.zeros((4, 385))
        timelines = self.timelines_by_topic[topic_id]

        for tl_id, tl in enumerate(timelines):
            # print('{}-{}'.format(topic_id, tl_id))
            tl_encoding = []

            for date_summary in tl:
                summary_encodings = []
                filtered_date_summary = date_summary[1][0].replace('–', '')[1:]
                sents = filtered_date_summary.split('. ')

                for sent in sents[-1]:
                    sent_encoding = self.encoder.encode(sent + '.')
                    summary_encodings.append(sent_encoding)

                summary_encoding = np.average(summary_encodings, axis=0)
                tl_encoding.append(summary_encoding)

            item_data[tl_id, 0] = topic_id * 4 + tl_id
            item_data[tl_id, 1:] = np.average(tl_encoding, axis=0)

        # print(item_data.shape)
        item_data = np.array(item_data)
        item_ids = item_data[:, 0].astype(int)  # the first column contains item IDs
        item_feats = item_data[:, 1:]  # the remaining columns contain item features

        pair_data = np.array([[topic_id * 4 + 0, topic_id * 4 + 1, 0],
                              [topic_id * 4 + 0, topic_id * 4 + 2, 0],
                              [topic_id * 4 + 0, topic_id * 4 + 3, 0],
                              [topic_id * 4 + 1, topic_id * 4 + 2, 0],
                              [topic_id * 4 + 1, topic_id * 4 + 3, 0],
                              [topic_id * 4 + 2, topic_id * 4 + 3, 1]])
        items_1_idxs = np.array([np.argwhere(item_ids == iid)[0][0] for iid in pair_data[:, 0].astype(int)])
        items_2_idxs = np.array([np.argwhere(item_ids == iid)[0][0] for iid in pair_data[:, 1].astype(int)])
        prefs = pair_data[:, 2]
        self.pref_model = GPPrefLearning(item_feats.shape[1], shape_s0=2, rate_s0=200)
        self.pref_model.fit(items_1_idxs, items_2_idxs, item_feats, prefs, optimize=False)
        predicted, _ = self.pref_model.predict_f(item_feats)
        self.pref_min = min(predicted)
        self.pref_max = max(predicted)

    def update(self, keywords, topic_id, dataset_name):
        self.update_keywords(keywords)
        if dataset_name =='t17':
            self.update_pref(topic_id)

    # R1
    def factual_consistency(self, source_embedding, summary_embedding):
        ret = sum(cosine_similarity([source_embedding], [summary_embedding])[0] / len(source_embedding))
        return ret

    # R2
    def standardised_pref(self, summary_embedding):
        ret = self.pref_model.predict_f(summary_embedding)
        ret = (ret - self.pref_min) / (self.pref_max - self.pref_min)
        return ret

    def count_keywords(self, summary):
        summary_tokens = summary.lower().split(' ')
        ret = 0
        for k in self.keywords:
            if k in summary_tokens:
                ret += 1
        return ret / len(self.keywords)

    def topical_coherence(self, summary_embedding):
        ret = self.standardised_pref(summary_embedding.reshape(1, -1))[0, 0, 0]
        ret = ret + sum(cosine_similarity([summary_embedding], self.keywords_embeddings)[0] / len(self.keywords))
        return ret

    # R3
    def language_quality(self, summary):
        summary_input_ids = self.lq_tokenizer(summary, truncation=True, return_tensors="pt")['input_ids'].to(self.device)
        outputs = self.lq_model(input_ids=summary_input_ids, labels=summary_input_ids)
        loss = outputs.loss.item()
        ret = 1.0 / loss if loss > 1e-5 else 0
        return ret

    # R4
    def repetition_punishment(self, summary):
        tokens = summary.lower().split()
        cnt = Counter(tokens)
        # print(cnt)
        rep = 1
        for k in cnt:
            v = cnt[k]
            if v > 1:
                rep += v
        if len(tokens) == 0:
            ret = 1
        else:
            ret = min(3, (1.0*len(tokens)) / rep)
        return ret

    def calc(self, input_ids, decoder_input_ids, tokenizer):
        with torch.no_grad():
            source = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
            source_sents = [s.strip() for s in source.split('.')]
            summary_sents = [s.strip() for s in summary.split('.')]

            source_embedding = self.encoder.encode(source_sents)
            summary_embedding = self.encoder.encode(summary_sents)
            source_embedding = np.mean(source_embedding, axis=0)
            summary_embedding = np.mean(summary_embedding, axis=0)

            state = np.concatenate((source_embedding, summary_embedding), axis=0)


            if len(summary) == 0:
                ret_batch = [0, 0, 0, 0, 0]
                return state, ret_batch, None

            #R1 = self.weights[0] * self.factual_consistency(source_embedding=source_embedding, summary_embedding=summary_embedding)
            R1 = 0
            #R2 = 0
            #R2 = self.weights[1] * self.topical_coherence(summary_embedding=summary_embedding)
            R2 = self.count_keywords(summary)
            R3 = self.weights[2] * self.language_quality(summary)
            #R3 = 0
            R4 = self.weights[3] * self.repetition_punishment(summary=summary)
            #R4 = 0
            #print(f'R1={R1} R2={R2} R3={R3} R4={R4}')

            ret = R1 + R2 + R3 + R4
            ret_batch = [ret, R1, R2, R3, R4]
        return state, ret_batch, summary