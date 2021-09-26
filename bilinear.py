import sys
import csv
import time

import math
import random
import datetime
import itertools
from collections import Counter

import torch
import rfutils
import opt_einsum 

import feedforward as ff
import readwrite as rw

DEBUG = True

INF = float('inf')
UNK = "!!!<UNK>!!!"

DEFAULT_NUM_ITER = 25000
DEFAULT_STRUCTURE = str([300, 300, 300])
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 10 ** -3
DEFAULT_ACTIVATION = "relu"
DEFAULT_CHECK_EVERY = 100
DEFAULT_PATIENCE = None
DEFAULT_DROPOUT = 0.0
DEFAULT_FILENAME = None#"output/model_%s.pt" % str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
EPSILON = 10 ** -7

ACTIVATIONS = {
    'relu': torch.nn.ReLU(),
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def identity(x):
    return x

def loglogit(x, eps=EPSILON):
    return x - torch.log(1 - torch.exp(x - eps))

class AdditiveSmoothing:
    def __init__(self, counts, w_vocab, c_vocab):
        self.counts = unkify(counts, w_vocab, c_vocab)
        self.marginal_counts = Counter()
        for (w, *cs), v in self.counts.items():
            cs = tuple(cs)
            self.marginal_counts[cs] += v
        self.w_vocab = w_vocab
        self.c_vocab = c_vocab

    @property
    def V(self):
        return len(self.w_vocab)

    def surprisal(self, xs, alpha):
        def gen():
            for w, *cs in xs:
                cs = tuple(c if c in self.c_vocab else UNK for c in cs)
                w = w if w in self.w_vocab else UNK
                numerator = math.log(self.counts[(w,)+cs] + alpha)
                logZ = math.log(self.marginal_counts[cs] + alpha*self.V)
                yield logZ - numerator
        return rfutils.mean(gen())

class BackoffSmoothing:
    def __init__(self, counts, w_vocab, c_vocab):
        self.counts = unkify(counts, w_vocab, c_vocab) # do thei nputs already have c as tuples??
        self.marginal_counts = Counter()
        self.word_counts = Counter()
        self.N = 0
        for (w, *cs), v in self.counts.items():
            cs = tuple(cs)
            self.marginal_counts[cs] += v
            self.word_counts[w] += v
            self.N += v
        self.w_vocab = w_vocab
        self.c_vocab = c_vocab

    @property
    def V(self):
        return len(self.w_vocab)        
        
    def surprisal(self, xs, lamda, alpha):
        def gen():
            for w, *cs in xs:
                cs = tuple(c if c in self.c_vocab else UNK for c in cs)
                w = w if w in self.w_vocab else UNK                
                if self.marginal_counts[cs]:
                    bigram_prob = self.counts[(w,)+cs]
                    bigram_prob /= self.marginal_counts[cs]
                else:
                    bigram_prob = 0
                unigram_smoothed = (self.word_counts[w] + alpha) / (self.N + alpha*self.V)
                yield -math.log(bigram_prob + lamda * unigram_smoothed)
        return rfutils.mean(gen())

class WordVectors(torch.nn.Module):
    def __init__(self, vectors_dict, device=DEVICE, finetune=False):
        super().__init__()
        self.device = device
        words, vectors = zip(*vectors_dict.items())
        self.D = len(rfutils.first(vectors))        
        
        assert UNK not in words
        words_with_unk = words + (UNK,)
        self.word_indices = {w:i for i, w in enumerate(words_with_unk)}
        self.unk_index = self.word_indices[UNK]
        
        unk_vector = torch.randn(self.D)
        unk_vector /= torch.norm(unk_vector) # needs to be norm 1 like other vectors
        vectors = vectors + (unk_vector,) 
        self.vectors = torch.Tensor(vectors).to(self.device)
        if finetune:
            self.vectors = torch.nn.Parameter(self.vectors)
        self.unk_vector = self.vectors[self.unk_index]

    def indices_of(self, words, vocab=None):
        indices = [
            self.word_indices[w] if (vocab is None or w in vocab) and w in self.word_indices else self.unk_index
            for w in words
        ]
        return torch.LongTensor(indices).to(self.device)

    def embed_words(self, words, vocab=None):
        indices = self.indices_of(words, vocab=vocab)
        return self.vectors[indices]

class MarginalLogLinear(torch.nn.Module):
    def __init__(self, w_encoder_structure, vectors, support=None, activation=DEFAULT_ACTIVATION, dropout=DEFAULT_DROPOUT, device=DEVICE):
        super().__init__()
        self.device = device        
        self.vectors = vectors
        if not w_encoder_structure:
            self.w_encoder = identity
            K = self.vectors.D
        else:
            self.w_encoder = ff.FeedForward(w_encoder_structure, activation=ACTIVATIONS[activation], dropout=dropout, device=device)
            K = w_encoder_structure[-1]
        self.linear = torch.nn.Linear(K, 1, bias=False, device=device)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if support is None:
            self.support = self.vectors.word_indices
            self.support_indices = ... # extract everything
        else:
            self.support = set(support) & set(self.vectors.word_indices) | {UNK}            
            self.support_indices = self.vectors.indices_of(self.support)                       

    def forward(self, words):
        if isinstance(words, torch.Tensor):
            # assumes UNKs already handled
            return self.forward_from_indices(words[:, 0])
        else:
            only_words = [w for w, *_ in words]
            indices = self.vectors.indices_of(only_words, vocab=self.support) # shape B x D
            return self.forward_from_indices(indices)

    def forward_from_indices(self, indices):
        v_w = self.vectors.vectors[indices]
        energy = self.linear(self.w_encoder(v_w)).squeeze(-1) # shape B
        support_vectors = self.vectors.vectors[self.support_indices]
        logZ = self.linear(self.w_encoder(support_vectors)).squeeze(-1).logsumexp(-1) # shape 1, same across batch
        return logZ - energy

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
            self.support = list(self.vectors.word_indices)
        else:
            self.support = list(set(support) & set(self.vectors.word_indices) | {UNK})
            self.support_indices = self.vectors.indices_of(self.support)                        
        self.support_indices_lookup = {w:i for i, w in enumerate(self.support)}
        self.device = device

    def forward(self, pairs):
        ws, cs = zip(*pairs)
        c_vectors = self.vectors.embed_words(cs)
        outputs = self.net(c_vectors) # shape B x V, logspace normalized
        w_indices = torch.LongTensor([[
            self.support_indices_lookup[w] if w in self.support_indices_lookup else self.support_indices_lookup[UNK]
            for w in ws
        ]]).to(self.device)
        logprobs = torch.gather(outputs, -1, w_indices).squeeze(-2) # shape B
        return -logprobs

class ConditionalLogBilinear(torch.nn.Module):
    def __init__(self, w_encoder_structure, c_encoder_structure, vectors, support=None, activation=DEFAULT_ACTIVATION, dropout=DEFAULT_DROPOUT, device=DEVICE):
        super().__init__()
        self.device = device
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
        torch.nn.init.xavier_uniform_(self.bilinear.weight)
        torch.nn.init.xavier_uniform_(self.w_linear.weight)
        if support is None:
            self.support = self.vectors.word_indices
            self.support_indices = ... # extract everything
        else:
            self.support = set(support) & set(self.vectors.word_indices) | {UNK}
            self.support_indices = self.vectors.indices_of(self.support)            

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            # assumes UNKs already handled
            i_w, i_c = batch.T            
            return self.forward_from_indices(i_w, i_c)
        else:
            ws, cs = zip(*batch)
            i_w = self.vectors.indices_of(ws, vocab=self.support)
            i_c = self.vectors.indices_of(cs)
            return self.forward_from_indices(i_w, i_c)            

    def forward_from_indices(self, ws, cs):
        v_w = self.vectors.vectors[ws]
        v_c = self.vectors.vectors[cs]
        h_w = self.w_encoder(v_w) # shape B x K
        h_c = self.c_encoder(v_c) # shape B x L
        support_vectors = self.vectors.vectors[self.support_indices]
        h_v = self.w_encoder(support_vectors) # shape V x L, make contiguous?
        # energy = <w | A | c> + <B | w>
        energy = (self.bilinear(h_w, h_c) + self.w_linear(h_w)).squeeze(-1) # shape B
        # logZ = ln \sum_w exp <w | A | c_i> + <B | w>
        A = self.bilinear.weight.squeeze(0)         
        logZ = (opt_einsum.contract("vi,ij,bj->bv", h_v, A, h_c) + self.w_linear(h_v).T).logsumexp(-1)
        result = logZ - energy
        if DEBUG:
            if (result < -.1).any():
                import pdb; pdb.set_trace()
        return result

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


class WordData:
    def __init__(self, elements):
        self.elements = list(elements)
        self.arity = len(rfutils.first(self.elements))
        self.N = len(self.elements)
        
    def minibatches(self, k, verbose=True):
        for j in itertools.count():
            these_elements = self.elements.copy()
            random.shuffle(these_elements)
            while these_elements:
                this_sample = []
                for i in range(k):
                    if these_elements:
                        this_sample.append(these_elements.pop())
                    else:
                        break
                yield this_sample
            if verbose:
                print("Finished epoch %d" % j, file=sys.stderr)

class IndexData:
    def __init__(self, elements, model):
        print("Processing training data...", file=sys.stderr, end=" ")
        wss = zip(*elements)
        self.indices = torch.stack([
            model.vectors.indices_of(ws) for ws in wss
        ]).T
        print("Done.", file=sys.stderr)
        self.N, self.arity = self.indices.shape
        self.device = model.device

    def minibatches(self, k, verbose=True):
        for j in itertools.count():
            perm = torch.randperm(self.N, device=self.device)
            for b in itertools.count():
                start = k*b
                end = start + k
                if start >= self.N:
                    break
                the_slice = perm[start:end]
                yield self.indices[the_slice] # make contiguous?
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

def train(model, train_data, dev_data=None, test_data=None, w_vocab=None, c_vocab=None, batch_size=DEFAULT_BATCH_SIZE, num_iter=DEFAULT_NUM_ITER, check_every=DEFAULT_CHECK_EVERY, patience=DEFAULT_PATIENCE, clip=None, data_on_device=False, **kwds):

    if data_on_device and not isinstance(model, ConditionalSoftmax):
        train_data_minibatcher = IndexData(train_data.elements(), model)
    else:
        train_data_minibatcher = WordData(train_data.elements())
        
    G = train_data_minibatcher.arity
    train_data_gen = train_data_minibatcher.minibatches(batch_size)
    if dev_data:
        dev_tokens = list(dev_data.elements())

        if G == 2:
            dev_unseen_combo, dev_unseen_w, dev_unseen_c, dev_unseen_both = dev_split(train_data, dev_data)

            dev_unseen_combo_tokens = list(dev_unseen_combo.elements())
            dev_unseen_w_tokens = list(dev_unseen_w.elements())
            dev_unseen_c_tokens = list(dev_unseen_c.elements())
            dev_unseen_both_tokens = list(dev_unseen_both.elements())

    if test_data:
        test_tokens = list(test_data.keys())
        test_values = torch.Tensor(list(test_data.values())).to(DEVICE)

    opt = torch.optim.Adam(params=list(model.parameters()), **kwds)
    diagnostics = []
    old_dev_loss = INF
    excursions = 0

    print("Initializing baselines...", file=sys.stderr, end=" ")
    smoothed = AdditiveSmoothing(train_data, w_vocab, c_vocab)
    if G == 2:
        backoff = BackoffSmoothing(train_data, w_vocab, c_vocab)
    print("Done.", file=sys.stderr)

    first_line = True
    start = datetime.datetime.now()
    for i in range(num_iter):
        train_batch = next(train_data_gen)
        
        opt.zero_grad()            
        loss = model(train_batch).mean()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        if check_every is not None and i % check_every == 0:
            diagnostic = {'step': i, 'train_mb_loss': loss.item()}            
            
            diagnostic['train_mb_mle'] = smoothed.surprisal(train_batch, 0)
            diagnostic['train_mb_smoothed_1.0'] = smoothed.surprisal(train_batch, 1)
            if G == 2:
                diagnostic['train_mb_backoff_0.25'] = backoff.surprisal(train_batch, 1/4, 1)

            me = model.eval()
            if dev_data:
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
                    diagnostic['dev_unseen_both_smoothed_1.0'] = smoothed.surprisal(dev_unseen_both_tokens, 1)
                    diagnostic['dev_unseen_both_backoff_0.25'] = backoff.surprisal(dev_unseen_both_tokens, 1/4, 1)


                if patience is not None and dev_loss > old_dev_loss:
                    excursions += 1
                    if excursions > patience:
                        break
                    else:
                        old_dev_loss = dev_loss
                diagnostic['dev_loss'] = dev_loss

            if test_data is not None:
               est = loglogit(-me(test_tokens))
               obs = loglogit(test_values)
               diagnostic['test_err'] = ((est - obs)**2).mean().item()

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

def unkify(data, w_vocab, c_vocab):
    result = Counter()
    for parts, value in data.items():
        new_parts = [
            UNK if (i == 0 and part not in w_vocab) or part not in c_vocab else part
            for i, part in enumerate(parts)
        ]
        result[tuple(new_parts)] += value
    return result

def filter_unks(data, w_vocab, verbose=True):
    result = Counter()
    N_unk = 0
    N = 0
    for parts, value in data.items():
        N += value
        w, *_ = parts
        if w in w_vocab:
            result[parts] = value
        else:
            N_unk += value
    if verbose:
        print("UNK proportion: %s" % str(N_unk/N), file=sys.stderr)
    return result

def main(vectors_filename,
            train_filename,
            dev_filename=None,
            test_filename=None,
            vocab=None,
            tie_params=False,
            softmax=False,
            no_encoders=False,
            include_unk=False,            
            seed=None,
            finetune=False,
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
    vectors = WordVectors(vectors_dict, device=DEVICE, finetune=finetune)
    if vocab:
        vocab_words = set(rw.read_words(vocab)) & set(vectors.word_indices) | {UNK}
    else:
        vocab_words = vectors.word_indices
    print("Support size = %d + 1" % len(vocab_words) - 1, file=sys.stderr)
    train_data = rw.read_counts(train_filename, verbose=True)
    train_data = unkify(train_data, vocab_words, vectors.word_indices)
    if dev_filename:
        dev_data = rw.read_counts(dev_filename, verbose=True)
        if not include_unk:
            dev_data = filter_unks(dev_data, vocab_words, verbose=True)
    else:
        dev_data = None
    if test_filename:
        test_data = rw.read_numbers(test_filename)
        if not include_unk:
            test_data = filter_unks(test_data, vocab_words, verbose=True)
    else:
        test_data = None

    print("Initializing model...", file=sys.stderr, end=" ")
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
    print("Done.", file=sys.stderr)
        
    model, diagnostics = train(model, train_data, dev_data=dev_data, test_data=test_data, w_vocab=vocab_words, c_vocab=vectors.word_indices, **kwds)
    if output_filename is not None:
        torch.save(model, output_filename)
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model joint word probabilities using a log-bilinear model')
    parser.add_argument("vectors", type=str, help="Path to word vectors in word2vec format")
    parser.add_argument("train", type=str, help="Path to file containing training counts of word pairs")
    parser.add_argument("--dev", type=str, default=None, help="Path to file containing dev counts of word pairs")
    parser.add_argument("--test", type=str, default=None, help="Path to file containing test log probabilities")
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
    parser.add_argument("--include_unk", action='store_true', help="Include UNK target words in dev and test sets.")
    parser.add_argument("--data_on_device", action='store_true', help="Store training data on GPU (faster for big datasets but uses a lot of GPU memory).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for minibatches.")
    parser.add_argument("--finetune", action="store_true", help="Finetune word vectors.")
    parser.add_argument('--clip', type=float, default=None, help='gradient clipping')
    args = parser.parse_args()
    main(args.vectors, args.train, dev_filename=args.dev, test_filename=args.test, phi_structure=args.structure, psi_structure=args.structure, activation=args.activation, dropout=args.dropout, check_every=args.check_every, patience=args.patience, tie_params=args.tie_params, vocab=args.vocab, num_iter=args.num_iter, softmax=args.softmax, output_filename=args.output_filename, no_encoders=args.no_encoders, seed=args.seed, batch_size=args.batch_size, include_unk=args.include_unk, finetune=args.finetune, clip=args.clip, data_on_device=args.data_on_device)

    

