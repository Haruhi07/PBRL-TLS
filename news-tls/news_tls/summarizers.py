from pprint import pprint
import random

import torch
import numpy as np
import networkx as nx
from scipy import sparse
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import inspect
from news_tls.gpu_mem_track import MemTracker
from torch.distributions import Categorical
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans


def top_k_filtering(logits, top_k=0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    return logits


class Summarizer:

    def summarize(self, sents, k, vectorizer, filters=None):
        raise NotImplementedError

class PegasusSummariser(Summarizer):
    def __init__(self, tokenizer, model, critic, critic_loss_fct, optimizerA, optimizerC, device='cuda', max_new_tokens=90):
        self.device = device
        self.name = 'PEGASUS Summariser'
        self.episodes = 4
        self.max_new_tokens = max_new_tokens
        self.summariser = model
        self.tokenizer = tokenizer
        self.critic = critic
        self.critic_loss_fct = critic_loss_fct
        self.optimizerA = optimizerA
        self.optimizerC = optimizerC

    def concatenate_sents(self, sents):
        ret = None
        for sent in sents:
            if ret is None:
                ret = sent
            else:
                ret = ret + ' ' + sent
        return ret

    def summarize(self, sents, k, vectorizer, filter=None):
        self.summariser.eval()
        raw_sents = [s.raw for s in sents]
        if len(raw_sents) == 0:
            return None
        random.shuffle(raw_sents)
        input_text = self.concatenate_sents(raw_sents)
        input_ids = self.tokenizer(input_text,
                                   truncation=True,
                                   return_tensors='pt').to(self.device)
        #print(input_ids)
        with torch.no_grad():
            output_ids = self.summariser.generate(**input_ids, max_new_tokens=self.max_new_tokens, repetition_penalty=2.0)[0]
        #print(output_ids)
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        #pprint(raw_sents)
        #pprint(output_text)
        return output_text

    def rl(self, args, sents, env, k, filter=None):
        raw_sents = [s.raw for s in sents]
        if len(raw_sents) == 0:
            return None, 0
        input_text = self.concatenate_sents(raw_sents)
        input_ids = self.tokenizer(
            input_text,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].to(self.device)

        self.summariser.train()
        #for p in self.summariser.parameters():
        #    p.requires_grad = False
        #for p in self.summariser.lm_head.parameters():
        #    p.requires_grad = True
        self.critic.train()
        decoder_start_ids = [2]
        explained_vars = []

        for i_episode in range(self.episodes):
            actions = []
            values = torch.zeros((self.max_new_tokens,)).to(self.device)
            rewards = torch.zeros((self.max_new_tokens,)).to(self.device)
            state, _, __ = env.calc(input_ids, [decoder_start_ids], self.tokenizer)
            for t in range(self.max_new_tokens):
                with torch.no_grad():
                    logits = self.summariser(
                        input_ids=input_ids,
                        decoder_input_ids=torch.tensor([decoder_start_ids+actions], dtype=int, device=self.device),
                        return_dict=True,
                    )["logits"][0]

                    prob = Categorical(logits=logits[-1])
                    action = prob.sample()
                    actions.append(action)

                state = torch.from_numpy(state).float().to(self.device)
                value = self.critic(state)
                values[t] = value

                state, reward, summary = env.calc(input_ids, [decoder_start_ids+actions], self.tokenizer)
                #pprint(reward)
                rewards[t] = reward[0]

            # build computing graph
            logits = self.summariser(
                input_ids=input_ids,
                decoder_input_ids=torch.tensor([decoder_start_ids + actions], device=self.device),
                return_dict=True,
            )["logits"][0]

            prob = Categorical(logits=logits[:-1])
            logprobs = prob.log_prob(torch.tensor(actions, device=self.device))
            entropy = prob.entropy()

            print(f"ep: {i_episode}/{self.episodes} reward sum: {rewards.sum()}")
            with torch.no_grad():
                returns = torch.zeros_like(values)
                for t in reversed(range(self.max_new_tokens)):
                    returns[t] = rewards[t] + (returns[t+1]*1 if t+1 < self.max_new_tokens else 0)

                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                advantages = returns - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_losses = torch.mean(-logprobs * advantages) - 2e-4 * entropy.mean()
            value_losses = 0.5*self.critic_loss_fct(values, returns).mean()
            print(f"actor loss: {actor_losses} value loss: {value_losses}")


            self.optimizerA.zero_grad()
            self.optimizerC.zero_grad()

            actor_losses.backward()
            value_losses.backward()

            self.optimizerA.step()
            self.optimizerC.step()

            summary = self.tokenizer.decode(decoder_start_ids + actions, skip_special_tokens=True)
            #print(f'summary: {summary}')

            y_pred, y_true = values.detach().cpu().numpy(), returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            print(f"explained variance: {explained_var}\n")
            explained_vars.append(explained_vars)

        return summary, explained_vars


class TextRank(Summarizer):

    def __init__(self, max_sim=0.9999):
        self.name = 'TextRank Summarizer'
        self.max_sim = max_sim

    def score_sentences(self, X):
        S = cosine_similarity(X)
        nodes = list(range(S.shape[0]))
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for i in range(S.shape[0]):
            for j in range(S.shape[0]):
                graph.add_edge(nodes[i], nodes[j], weight=S[i, j])
        pagerank = nx.pagerank(graph, weight='weight')
        scores = [pagerank[i] for i in nodes]
        return scores

    def summarize(self, sents, k, vectorizer, filter=None):
        raw_sents = [s.raw for s in sents]
        try:
            X = vectorizer.transform(raw_sents)
        except:
            return None

        scores = self.score_sentences(X)

        indices = list(range(len(sents)))
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)

        summary_sents = []
        summary_vectors = []
        for i, _ in ranked:
            if len(summary_sents) >= k:
                break
            new_x = X[i]
            s = sents[i]
            is_redundant = False
            for x in summary_vectors:
                if cosine_similarity(new_x, x)[0, 0] > self.max_sim:
                    is_redundant = True
                    break
            if filter and not filter(s):
                continue
            elif is_redundant:
                continue
            else:
                summary_sents.append(sents[i])
                summary_vectors.append(new_x)

        summary = [s.raw for s in summary_sents]
        return summary


class CentroidRank(Summarizer):

    def __init__(self,  max_sim=0.9999):
        self.name = 'Sentence-Centroid Summarizer'
        self.max_sim = max_sim

    def score_sentences(self, X):
        Xsum = sparse.csr_matrix(X.sum(0))
        centroid = normalize(Xsum)
        scores = cosine_similarity(X, centroid)
        return scores

    def summarize(self, sents, k, vectorizer, filter=None):
        raw_sents = [s.raw for s in sents]
        try:
            X = vectorizer.transform(raw_sents)
            for i, s in enumerate(sents):
                s.vector = X[i]
        except:
            return None

        scores = self.score_sentences(X)
        indices = list(range(len(sents)))
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)

        summary_sents = []
        summary_vectors = []
        for i, _ in ranked:
            if len(summary_sents) >= k:
                break
            new_x = X[i]
            s = sents[i]
            is_redundant = False
            for x in summary_vectors:
                if cosine_similarity(new_x, x)[0, 0] > self.max_sim:
                    is_redundant = True
                    break
            if filter and not filter(s):
                continue
            elif is_redundant:
                continue
            else:
                summary_sents.append(sents[i])
                summary_vectors.append(new_x)

        summary = [s.raw for s in summary_sents]
        return summary


class CentroidOpt(Summarizer):

    def __init__(self, max_sim=0.9999):
        self.name = 'Summary-Centroid Summarizer'
        self.max_sim = max_sim

    def optimise(self, centroid, X, sents, k, filter):
        remaining = set(range(len(sents)))
        selected = []
        while len(remaining) > 0 and len(selected) < k:
            if len(selected) > 0:
                summary_vector = sparse.vstack([X[i] for i in selected])
                summary_vector = sparse.csr_matrix(summary_vector.sum(0))
            i_to_score = {}
            for i in remaining:
                if len(selected) > 0:
                    new_x = X[i]
                    new_summary_vector = sparse.vstack([new_x, summary_vector])
                    new_summary_vector = normalize(np.asarray(new_summary_vector.sum(0)))
                else:
                    new_summary_vector = X[i]
                score = cosine_similarity(new_summary_vector, centroid)[0, 0]
                i_to_score[i] = score

            ranked = sorted(i_to_score.items(), key=lambda x: x[1], reverse=True)
            for i, score in ranked:
                s = sents[i]
                remaining.remove(i)
                if filter and not filter(s):
                    continue
                elif self.is_redundant(i, selected, X):
                    continue
                else:
                    selected.append(i)
                    break
        return selected

    def is_redundant(self, new_i, selected, X):
        summary_vectors = [X[i] for i in selected]
        new_x = X[new_i]
        for x in summary_vectors:
            if cosine_similarity(new_x, x)[0] > self.max_sim:
                return True
        return False

    def summarize(self, sents, k, vectorizer, filter=None):
        raw_sents = [s.raw for s in sents]
        try:
            X = vectorizer.transform(raw_sents)
        except:
            return None
        X = sparse.csr_matrix(X)
        Xsum = sparse.csr_matrix(X.sum(0))
        centroid = normalize(Xsum)
        selected = self.optimise(centroid, X, sents, k, filter)
        summary = [sents[i].raw for i in selected]
        #pprint(raw_sents)
        #pprint(summary)
        return summary


class SubmodularSummarizer(Summarizer):
    """
    Selects a combination of sentences as a summary by greedily optimising
    a submodular function.
    The function models the coverage and diversity of the sentence combination.
    """
    def __init__(self, a=5, div_weight=6, cluster_factor=0.2):
        self.name = 'Submodular Summarizer'
        self.a = a
        self.div_weight = div_weight
        self.cluster_factor = cluster_factor

    def cluster_sentences(self, X):
        n = X.shape[0]
        n_clusters = round(self.cluster_factor * n)
        if n_clusters <= 1 or n <= 2:
            return dict((i, 1) for i in range(n))
        clusterer = MiniBatchKMeans(n_clusters=n_clusters)
        labels = clusterer.fit_predict(X)
        i_to_label = dict((i, l) for i, l in enumerate(labels))
        return i_to_label

    def compute_summary_coverage(self,
                                 alpha,
                                 summary_indices,
                                 sent_coverages,
                                 pairwise_sims):
        cov = 0
        for i, i_generic_cov in enumerate(sent_coverages):
            i_summary_cov = sum([pairwise_sims[i, j] for j in summary_indices])
            i_cov = min(i_summary_cov, alpha * i_generic_cov)
            cov += i_cov
        return cov

    def compute_summary_diversity(self,
                                  summary_indices,
                                  ix_to_label,
                                  avg_sent_sims):

        cluster_to_ixs = collections.defaultdict(list)
        for i in summary_indices:
            l = ix_to_label[i]
            cluster_to_ixs[l].append(i)
        div = 0
        for l, l_indices in cluster_to_ixs.items():
            cluster_score = sum([avg_sent_sims[i] for i in l_indices])
            cluster_score = np.sqrt(cluster_score)
            div += cluster_score
        return div

    def optimise(self,
                 sents,
                 k,
                 filter,
                 ix_to_label,
                 pairwise_sims,
                 sent_coverages,
                 avg_sent_sims):

        alpha = self.a / len(sents)
        remaining = set(range(len(sents)))
        selected = []

        while len(remaining) > 0 and len(selected) < k:

            i_to_score = {}
            for i in remaining:
                summary_indices = selected + [i]
                cov = self.compute_summary_coverage(
                    alpha, summary_indices, sent_coverages, pairwise_sims)
                div = self.compute_summary_diversity(
                    summary_indices, ix_to_label, avg_sent_sims)
                score = cov + self.div_weight * div
                i_to_score[i] = score

            ranked = sorted(i_to_score.items(), key=lambda x: x[1], reverse=True)
            for i, score in ranked:
                s = sents[i]
                remaining.remove(i)
                if filter and not filter(s):
                    continue
                else:
                    selected.append(i)
                    break

        return selected

    def summarize(self, sents, k, vectorizer, filter=None):
        raw_sents = [s.raw for s in sents]
        try:
            X = vectorizer.transform(raw_sents)
        except:
            return None

        ix_to_label = self.cluster_sentences(X)
        pairwise_sims = cosine_similarity(X)
        sent_coverages = pairwise_sims.sum(0)
        avg_sent_sims = sent_coverages / len(sents)

        selected = self.optimise(
            sents, k, filter, ix_to_label,
            pairwise_sims, sent_coverages, avg_sent_sims
        )

        summary = [sents[i].raw for i in selected]
        return summary
