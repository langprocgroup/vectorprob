import sys
import csv
import time

import math
import random
import datetime
import itertools
from collections import Counter

import tqdm
import torch
import rfutils
import opt_einsum 

import feedforward as ff
import readwrite as rw

INF = float('inf')
UNK = "!!!<UNK>!!!"

DEFAULT_NUM_ITER = 10 ** 4
DEFAULT_STRUCTURE = str([300, 300, 300])
DEFAULT_BATCH_SIZE = 10
DEFAULT_LR = 10 ** -3
DEFAULT_ACTIVATION = "relu"
DEFAULT_CHECK_EVERY = 100
DEFAULT_PATIENCE = None
DEFAULT_DROPOUT = 0.0
DEFAULT_FILENAME = "model_%s.pt" % str(datetime.datetime.now()).split(".")[0].replace(" ", "_")

ACTIVATIONS = {
    'relu': torch.nn.ReLU(),
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def identity(x):
    return x

class AdditiveSmoothing:
    def __init__(self, counts):
        self.counts = Counter()
        self.marginal_counts = Counter()
        for (w, *c), v in counts.items():
            c = tuple(c)
            self.counts[w,c] = v
            self.marginal_counts[c] += v
        self.V = len(set(w for w, *_ in self.counts.keys()))

    def surprisal(self, xs, alpha):
        def gen():
            for w, *c in xs:
                c = tuple(c)
                numerator = math.log(self.counts[w,c] + alpha)
                logZ = math.log(self.marginal_counts[c] + alpha*self.V)
                yield logZ - numerator
        return rfutils.mean(gen())

class BackoffSmoothing:
    def __init__(self, counts):
        self.counts = counts
        self.marginal_counts = Counter()
        self.word_counts = Counter()
        self.N = 0
        for (w, c), v in self.counts.items():
            self.marginal_counts[c] += v
            self.word_counts[w] += v
            self.N += v
        self.V = len(self.word_counts)
        
    def surprisal(self, xs, lamda, alpha):
        def gen():
            for w, c in xs:
                if self.marginal_counts[c]:
                    bigram_prob = self.counts[w,c]
                    bigram_prob /= self.marginal_counts[c]
                else:
                    bigram_prob = 0
                unigram_smoothed = (self.word_counts[w] + alpha) / (self.N + alpha*self.V)
                yield -math.log(bigram_prob + lamda * unigram_smoothed)
        return rfutils.mean(gen())

class WordVectors:
    def __init__(self, words, vectors, device=DEVICE):
        self.device = device
        words = set(words)
        vectors = list(vectors)
        assert len(words) == len(vectors)
        self.D = len(rfutils.first(vectors))        
        
        assert UNK not in words
        words_with_unk = words | {UNK} # so UNK will be the last element
        self.word_indices = {w:i for i, w in enumerate(words_with_unk)}
        self.unk_index = self.word_indices[UNK]
        
        unk_vector = torch.randn(self.D)
        unk_vector /= torch.norm(unk_vector)
        vectors.append(list(unk_vector)) # UNK is the last vector
        self.vectors = torch.Tensor(vectors).to(self.device)
        self.unk_vector = self.vectors[self.unk_index]

    def indices_of(self, words, vocab=None):
        if vocab is None:
            vocab = self.word_indices
        indices = [
            self.word_indices[w] if w in vocab and w in self.word_indices else self.unk_index
            for w in words
        ]
        return torch.LongTensor(indices).to(self.device) # this is the biggest bottleneck!

    def embed_words(self, words, vocab=None):
        indices = self.indices_of(words, vocab=vocab)
        return self.vectors[indices] 

    def embed_groups(self, groups, vocab=None, restricted_index=0):
        """ Groups is an iterable of length B of tuples of G words. 
        Return a tensor of size G x B x D with the embeddings of the words.
        
        Apply vocabulary restriction only to the restricted index only.
        """
        columns = list(zip(*groups))
        tensors = []
        for i, column in enumerate(columns):
            restricted_vocab = vocab if i == restricted_index else None
            tensor = self.embed_words(column, vocab=restricted_vocab)
            tensors.append(tensor)
        return torch.stack(tensors)

class MarginalLogLinear(torch.nn.Module):
    def __init__(self, w_encoder_structure, vectors, support=None, activation=DEFAULT_ACTIVATION, dropout=DEFAULT_DROPOUT, device=DEVICE):
        super().__init__()
        self.vectors = vectors
        if not w_encoder_structure:
            self.w_encoder = identity
            K = self.vectors.D
        else:
            self.w_encoder = ff.FeedForward(w_encoder_structure, activation=ACTIVATIONS[activation], dropout=dropout, device=device)
            K = w_encoder_structure[-1]
        self.linear = torch.nn.Linear(K, 1, bias=True, device=device)
        if support is None:
            self.support = self.vectors.word_indices
            self.support_vectors = self.vectors.vectors
        else:
            self.support = set(support) | {UNK}
            self.support_vectors = self.vectors.embed_words(self.support)
        self.device = device

    def forward(self, words):
        """
        This function computes < weights | w_i > - \log \sum_w \exp < weights | w > 
        """
        only_words = [w for w, *_ in words]
        batch = self.vectors.embed_words(only_words, vocab=self.support) # shape B x D
        energy = self.linear(self.w_encoder(batch)).squeeze(-1) # shape B
        logZ = self.linear(self.w_encoder(self.support_vectors)).squeeze(-1).logsumexp(-1) # shape 1, same across batch
        return logZ - energy

class ConditionalLogLinear(torch.nn.Module):
    def __init__(self, w_encoder_structure, c_encoder_structure, vectors, support=None, activation=DEFAULT_ACTIVATION, dropout=DEFAULT_DROPOUT, device=DEVICE):
        super().__init__()
        self.vectors = vectors
        
        if not w_encoder_structure:
            self.w_encoder = identity
            K = self.vectors.D
        else:
            self.w_encoder = ff.FeedForward(
                w_encoder_structure,
                activation=ACTIVATIONS[activation],
                dropout=dropout,
                device=device
            )
            K = w_encoder_structure[-1]
        if c_encoder_structure is None:
            self.c_encoder = self.w_encoder
            L = K
        else:
            self.c_encoder = ff.FeedForward(
                c_encoder_structure,
                activation=ACTIVATIONS[activation],
                dropout=dropout,
                device=device
            )
            L = c_encoder_structure[-1]
        self.linear = torch.nn.Linear(K + L, 1, device=device)
        if support is None:
            self.support = self.vectors.word_indices
            self.support_vectors = self.vectors.vectors
        else:
            self.support = set(support) | {UNK}
            self.support_vectors = self.vectors.embed_words(self.support)
        self.device = device
            

    def forward(self, pairs):
        raise NotImplementedError
        v_w, v_c = self.vectors.embed_groups(pairs, vocab=self.support)
        h_w = self.w_encoder(v_w) # shape B x K
        h_c = self.c_encoder(v_c) # shape B x L
        h_wc = torch.cat([h_w, h_c], dim=-1) # shape B x (K + L)
        h_v = self.w_encoder(self.support_vectors) # shape V x K
        h_vc = ... # shape B x V x (K + L): need "b c, v w -> b v (c + w)" ugh, need to use meshgrid or something? ugly
        # first step b c, v w -> b c v w 
        # energy = <w_i | A | c_i>
        energy = self.linear(h_wc).squeeze(-1) # shape B
        logZ = self.linear(self.w_encoder(self.support_vectors)).squeeze(-1).logsumexp(-1) # shape 1, same across batch
        
        
class ConditionalSoftmax(torch.nn.Module):
    def __init__(self, c_encoder_structure, vectors, support, activation=DEFAULT_ACTIVATION, dropout=DEFAULT_DROPOUT, device=DEVICE):
        super().__init__()
        self.vectors = vectors
        V = len(support)
        if c_encoder_structure is None:
            self.net = ff.FeedForward(
                [self.vectors.D, V],
                activation=None,
                dropout=dropout,
                transform=torch.nn.LogSoftmax(-1),
                device=device,
            )
        else:
            structure = tuple(c_encoder_structure) + (V,)
            self.net = ff.FeedForward(
                structure,
                activation=ACTIVATIONS[activation],
                dropout=dropout,
                transform=torch.nn.LogSoftmax(-1),
                device=device,
            )
        if support is None:
            self.support = self.vectors.word_indices
            self.support_vectors = self.vectors.vectors
        else:
            self.support = set(support) | {UNK}
            self.support_vectors = self.vectors.embed_words(self.support)
        self.support_indices = {w:i for i, w in enumerate(self.support)}
        self.device = device

    def forward(self, pairs):
        ws, cs = zip(*pairs)
        c_vectors = self.vectors.embed_words(cs)
        outputs = self.net(c_vectors) # shape B x V
        w_indices = torch.LongTensor([[
            self.support_indices[w] if w in self.support_indices else self.support_indices[UNK]
            for w in ws
        ]])
        logprobs = torch.gather(outputs, -1, w_indices).squeeze(-2) # shape B
        return -logprobs

class ConditionalLogBilinear(torch.nn.Module):
    def __init__(self, w_encoder_structure, c_encoder_structure, vectors, support=None, activation=DEFAULT_ACTIVATION, dropout=DEFAULT_DROPOUT, device=DEVICE):
        super().__init__()
        self.vectors = vectors
        
        if w_encoder_structure is None:
            self.w_encoder = identity
            K = self.vectors.D
        else:
            self.w_encoder = ff.FeedForward(
                w_encoder_structure,
                activation=ACTIVATIONS[activation],
                dropout=dropout,
                device=device
            )
            K = w_encoder_structure[-1]
        if c_encoder_structure is None:
            self.c_encoder = self.w_encoder
            L = K
        else:
            self.c_encoder = ff.FeedForward(
                c_encoder_structure,
                activation=ACTIVATIONS[activation],
                dropout=dropout,
                device=device
            )
            L = c_encoder_structure[-1]
        self.bilinear = torch.nn.Bilinear(K, L, 1, bias=False, device=device)
        self.w_linear = torch.nn.Linear(K, 1, bias=False, device=device)
        if support is None:
            self.support = self.vectors.word_indices
            self.support_vectors = self.vectors.vectors
        else:
            self.support = set(support) | {UNK}
            self.support_vectors = self.vectors.embed_words(self.support)
        self.device = device

    def forward(self, pairs):
        """ Batch is an iterable of <w, c>.
        This function computes <w_i | A | c_i> - \log \sum_w \exp <w | A | c_i> """
        v_w, v_c = self.vectors.embed_groups(pairs, vocab=self.support)
        h_w = self.w_encoder(v_w) # shape B x E
        h_c = self.c_encoder(v_c) # shape B x E
        h_v = self.w_encoder(self.support_vectors) # shape V x E
        # energy = <w | A | c> + <B|w> + <C|c> + D; but <C|c>+D cancels out so not included
        energy = (self.bilinear(h_w, h_c) + self.w_linear(h_w)).squeeze(-1) # shape B
        # logZ = ln \sum_w exp <w | A | c_i> -- numerical b
        logZ = (
            opt_einsum.contract("vi,ij,bj->bv", h_v, self.bilinear.weight.squeeze(0), h_c) + # shape B x V
            self.w_linear(h_v).T # shape 1 x V
        ).logsumexp(-1) # shape B
        result = logZ - energy
        #if (result < -1).any():
        #    import pdb; pdb.set_trace()
        return result

# With [300, 25, 25], dev loss gets negative and train loss increases
# Suggests spurious energy is being transfered to unseen combos

def sample_from_histogram(histogram, k):
    # Very slow
    values = list(histogram.keys())
    indices = {x:i for i,x in enumerate(values)}
    counts = list(histogram.values())
    Z = sum(counts)
    assert k <= Z
    for _ in range(k):
        sample, = random.choices(values, weights=counts)
        yield sample
        counts[indices[sample]] -= 1

def minibatches(elements, k, verbose=True):
    elements = list(elements)
    for j in itertools.count():
        these_elements = elements.copy()
        random.shuffle(these_elements)
        while these_elements:
            this_sample = []
            for i in range(k):
                if these_elements:
                    this_sample.append(these_elements.pop())
            yield this_sample
        if verbose:
            print("Finished epoch %d" % j, file=sys.stderr)

def filter_dict(d, ok_keys):
    return {k:v for k, v in d.items() if k in ok_keys}

def dev_split(train, dev):
    train_w = {w for w,c in train}
    train_c = {c for w,c in train}
    unseen = {group for group in dev if group not in train}
    unseen_combo = [(w,c) for w,c in unseen if w in train_w and c in train_c]
    unseen_w = [(w,c) for w,c in unseen if w not in train_w and c in train_c]
    unseen_c = [(w,c) for w,c in unseen if w in train_w and c not in train_c]
    unseen_both = [(w,c) for w,c in unseen if w not in train_w and c not in train_c]
    print("N unseen combo: %d" % len(unseen_combo), file=sys.stderr)
    print("N unseen w: %d" % len(unseen_w), file=sys.stderr)
    print("N unseen c: %d" % len(unseen_c), file=sys.stderr)
    print("N unseen both: %d" % len(unseen_both), file=sys.stderr)    
    return (
        Counter(filter_dict(dev, unseen_combo)),
        Counter(filter_dict(dev, unseen_w)),
        Counter(filter_dict(dev, unseen_c)),
        Counter(filter_dict(dev, unseen_both)),
    )

def train(model, train_data, dev_data=None, batch_size=DEFAULT_BATCH_SIZE, num_iter=DEFAULT_NUM_ITER, check_every=DEFAULT_CHECK_EVERY, patience=DEFAULT_PATIENCE, **kwds):

    G = len(rfutils.first(train_data.keys()))    

    train_data_gen = minibatches(list(train_data.elements()), batch_size)    
    if dev_data:
        dev_tokens = list(dev_data.elements())

        if G == 2:
            dev_unseen_combo, dev_unseen_w, dev_unseen_c, dev_unseen_both = dev_split(train_data, dev_data)        

            dev_unseen_combo_tokens = list(dev_unseen_combo.elements())
            dev_unseen_w_tokens = list(dev_unseen_w.elements())
            dev_unseen_c_tokens = list(dev_unseen_c.elements())
            dev_unseen_both_tokens = list(dev_unseen_both.elements())

    opt = torch.optim.Adam(params=list(model.parameters()), **kwds)
    diagnostics = []
    old_dev_loss = INF
    excursions = 0

    smoothed = AdditiveSmoothing(train_data)
    if G == 2:
        backoff = BackoffSmoothing(train_data)

    first_line = True
    start = datetime.datetime.now()
    for i in range(num_iter):
        train_batch = list(next(train_data_gen))
        
        opt.zero_grad()            
        loss = model(train_batch).mean()
        loss.backward()
        opt.step()

        if check_every is not None and i % check_every == 0:
            diagnostic = {'step': i, 'train_mb_loss': loss.item()}            
            
            diagnostic['train_mb_mle'] = smoothed.surprisal(train_batch, 0)
            diagnostic['train_mb_smoothed_1.0'] = smoothed.surprisal(train_batch, 1)
            if G == 2:
                diagnostic['train_mb_backoff_0.25'] = backoff.surprisal(train_batch, 1/4, 1)
            
            if dev_data:
                me = model.eval()
                dev_loss = me(dev_tokens).mean().item()
                diagnostic['dev_loss'] = dev_loss
                diagnostic['dev_smoothed_1.0'] = smoothed.surprisal(dev_tokens, 1)
                if G == 2:
                    diagnostic['dev_backoff_0.25'] = backoff.surprisal(dev_tokens, 1/4, 1)

                if G == 2 and dev_unseen_combo_tokens:
                    diagnostic['dev_unseen_combo_loss'] = me(dev_unseen_combo_tokens).mean().item()
                    diagnostic['dev_unseen_combo_smoothed_1.0'] = smoothed.surprisal(dev_unseen_combo_tokens, 1)
                    diagnostic['dev_unseen_combo_backoff_0.25'] = backoff.surprisal(dev_unseen_combo_tokens, 1/4, 1)                    

                if G == 2 and dev_unseen_w_tokens:
                    diagnostic['dev_unseen_w_loss'] = me(dev_unseen_w_tokens).mean().item()
                    diagnostic['dev_unseen_w_smoothed_1.0'] = smoothed.surprisal(dev_unseen_w_tokens, 1)
                    diagnostic['dev_unseen_w_backoff_0.25'] = backoff.surprisal(dev_unseen_w_tokens, 1/4, 1)                                        

                if G == 2 and dev_unseen_c_tokens:
                    diagnostic['dev_unseen_c_loss'] = me(dev_unseen_c_tokens).mean().item()
                    diagnostic['dev_unseen_c_smoothed_1.0'] = smoothed.surprisal(dev_unseen_c_tokens, 1)
                    diagnostic['dev_unseen_c_backoff_0.25'] = backoff.surprisal(dev_unseen_c_tokens, 1/4, 1)                                                            

                if G == 2 and dev_unseen_both_tokens:
                    diagnostic['dev_unseen_both_loss'] = me(dev_unseen_both_tokens).mean().item()
                    #diagnostic['dev_unseen_both_smoothed_0.5'] = smoothed.surprisal(dev_unseen_both_tokens, 1/2)
                    diagnostic['dev_unseen_both_smoothed_1.0'] = smoothed.surprisal(dev_unseen_both_tokens, 1)
                    diagnostic['dev_unseen_both_backoff_0.25'] = backoff.surprisal(dev_unseen_both_tokens, 1/4, 1)

                if patience is not None and dev_loss > old_dev_loss:
                    excursions += 1
                    if excursions > patience:
                        break
                    else:
                        old_dev_loss = dev_loss
                diagnostic['dev_loss'] = dev_loss

            curr_time = datetime.datetime.now()
            diagnostic['time'] = str(curr_time - start)
            start = curr_time                

            if first_line:
                writer = csv.DictWriter(sys.stdout, diagnostic.keys())
                writer.writeheader()
                first_line = False
            writer.writerow(diagnostic)

            diagnostics.append(diagnostic)
                
    return model.eval(), diagnostics

def dict_transpose(iterable_of_dicts):
    result = {}
    it = iter(iterable_of_dicts)
    first = next(it)
    for k, v in first.items():
        result[k] = [v]
    for d in it:
        for k, v in d.items():
            result[k].append(v)
    return result

def main(vectors_filename,
            train_filename,
            dev_filename=None,
            vocab=None,
            tie_params=False,
            softmax=False,
            no_encoders=False,
            seed=None,
            output_filename=DEFAULT_FILENAME,            
            phi_structure=DEFAULT_STRUCTURE,
            psi_structure=DEFAULT_STRUCTURE,
            activation=DEFAULT_ACTIVATION,
            dropout=DEFAULT_DROPOUT,
            **kwds):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed+1)
    vectors_dict = rw.read_vectors(vectors_filename)
    vectors = WordVectors(vectors_dict.keys(), vectors_dict.values(), device=DEVICE)
    if vocab:
        vocab_words = set(rw.read_words(vocab)) | {UNK}
    else:
        vocab_words = vectors.word_indices
    print("Support size %d" % len(vocab_words), file=sys.stderr)
    
    train_data = rw.read_counts(train_filename)
    if dev_filename:
        dev_data = rw.read_counts(dev_filename)
    else:
        dev_data = None
    G = len(rfutils.first(train_data.keys()))
    if G == 1:
        model = MarginalLogLinear(
            eval(phi_structure) if not no_encoders else None,
            vectors,
            activation=activation,
            dropout=dropout,
            support=vocab_words,
        )
    elif G == 2:
        if softmax:
            model = ConditionalSoftmax(
                eval(phi_structure) if not no_encoders else None,
                vectors,
                support=vocab_words,
                activation=activation,
                dropout=dropout,
            )
        else:
            model = ConditionalLogBilinear(
                eval(phi_structure) if not no_encoders else None,
                None if tie_params else eval(psi_structure),
                vectors,
                activation=activation,
                dropout=dropout,
                support=vocab_words,
            )
    else:
        raise ValueError("Only works for unigrams or bigrams, but %d-grams detected in training data" % G)
        
    model, diagnostics = train(model, train_data, dev_data=dev_data, **kwds)
    with open(output_filename, 'wb') as outfile:
        torch.save(model, outfile)

    return 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model joint word probabilities using a log-bilinear model')
    parser.add_argument("vectors", type=str, help="Path to word vectors in word2vec format")
    parser.add_argument("train", type=str, help="Path to file containing training counts of word pairs")
    parser.add_argument("--dev", type=str, default=None, help="Path to file containing dev counts of word pairs")
    parser.add_argument("--vocab", type=str, default=None, help="Limit output vocabulary to words in the given file if provided")
    parser.add_argument("--tie_params", action='store_true', help="Set phi = psi")
    parser.add_argument("--softmax", action='store_true', help="Only use vectors for the context word, not the target word")
    parser.add_argument("--num_iter", type=int, default=DEFAULT_NUM_ITER)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="batch size; 0 means full gradient descent with no batches")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="starting learning rate for Adam")
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, help="activation function for networks")
    parser.add_argument("--structure", type=str, default=str(DEFAULT_STRUCTURE), help="network structure, same for phi and psi")
    parser.add_argument("--no_encoders", action='store_true', help="Do not train word encoders. Overrides structure arguments.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="dropout")
    parser.add_argument("--check_every", type=int, default=DEFAULT_CHECK_EVERY, help="Record progress and check for early stopping every x iterations")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Allow n increases in dev loss for early stopping. None means infinite patience. Default None.")
    parser.add_argument("--output_filename", type=str, default=DEFAULT_FILENAME, help="Output filename. If not specified, a default is used which indicates the time the training script was run..")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for minibatches.")
    args = parser.parse_args()
    sys.exit(main(args.vectors, args.train, dev_filename=args.dev, phi_structure=args.structure, psi_structure=args.structure, activation=args.activation, dropout=args.dropout, check_every=args.check_every, patience=args.patience, tie_params=args.tie_params, vocab=args.vocab, num_iter=args.num_iter, softmax=args.softmax, output_filename=args.output_filename, no_encoders=args.no_encoders, seed=args.seed, batch_size=args.batch_size))
    




