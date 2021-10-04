import argparse
import math
import struct
import os
import sys
import time
import warnings
import random

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from functools import reduce
from multiprocessing import Pool, Value, Array

class VocabItem:
    def __init__(self, word, grids):
        self.word = word
        self.count = 0
        self.path = None # Path (list of indices) from the root to the word (leaf)
        self.code = None # Huffman encoding
        self.grid = [int(x) for x in grids]

class Vocab:
    def __init__(self, fi, min_count, gi):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fi = open(fi, 'r',encoding='utf-8')

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        # for token in ['<bol>', '<eol>']:
        #     vocab_hash[token] = len(vocab_items)
        #     vocab_items.append(VocabItem(token))
        with open(gi) as f: r2g=f.readlines()
        r2g=[(x[:-1].split("-")[0],x[:-1].split("-")[1].split(",")) for x in r2g]
        r2g1=defaultdict(list)
        self.gridsmap=defaultdict(set)
        for pair in r2g:
            for i in pair[1]:
                self.gridsmap[int(i)].add(pair[0])
            r2g1[pair[0]]=pair[1]
                
        for line in fi:
            tokens = line[:-1].split(",")
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token, r2g1[token]))
                    #vocab_items.append(VocabItem(token, token.split("_")[0]))
                    
                #assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab_items[vocab_hash[token]].count += 1
                word_count += 1
            
                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            # vocab_items[vocab_hash['<bol>']].count += 1
            # vocab_items[vocab_hash['<eol>']].count += 1
            # word_count += 2

        self.bytes = fi.tell()
        self.vocab_items = vocab_items         # List of VocabItem objects
        self.vocab_hash = vocab_hash           # Mapping from each token to its index in vocab
        self.word_count = word_count           # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        #assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print ('Total words in training file: %d' % self.word_count)
        print ('Total bytes in training file: %d' % self.bytes)
        print ('Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        #tmp.append(VocabItem('<unk>'))
        unk_hash = 0
        
        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash


        print ('Unknown vocab size:', count_unk)

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        grids = defaultdict(dict)

        print ('Filling unigram table')
        for i, voc in enumerate(vocab):
            for g in voc.grid:
                grids[g][i]=voc.count
        
        for i in grids.keys():
            neargrids=[i-1,i+1,i-125,i+125,i-126,i-124,i+124,i+126]
            neargrids=[x for x in neargrids if x in grids.keys()]
            for j in neargrids:
                grids[i].update(grids[j])
        
        grids_size=len(grids.keys())
        
        #unigramtable=np.zeros(shape=(vocab_size,grids_size)).astype(np.int32)
        table=defaultdict(defaultdict)
        for i, idict in tqdm(grids.items()):
            
            index=list(idict.keys())
            cp=[]


            norm = sum([math.pow(t, power) for t in idict.values()]) # Normalizing constant
            for j in idict.keys():
                c = math.pow(idict[j], power)/norm
                cp.append(c)
            table[i]["index"]=index
            table[i]["unigram"]=cp
        self.table=table

    def sample(self, count, samplegrids):
        if len(samplegrids)>1:
            samplegrids=np.random.choice(samplegrids)
        else:
            samplegrids=samplegrids[0] 
        index=self.table[samplegrids]["index"]
        p=self.table[samplegrids]["unigram"]
        size=min(count,len(index))
        return np.random.choice(index,p=p,size=size,replace=False)

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

def init_net(dim, vocab_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
    print(tmp.shape)
    if os.path.exists("output.txt"):
        with open("output.txt") as f1:
            line=f1.readlines()
            j=0
            for i in line[1:]:
                vec = np.array(i.strip().split(' ')[1:]).astype(np.float32)
                tmp[j] = vec/np.linalg.norm(vec)
                j+=1
            print(j,tmp.shape)
    
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)
    print(dim,vocab_size)
    return (syn0, syn1)

def train_process(pid):
    # Set fi to point to the right chunk of training file
    # if (pid==0):
    #     start = 0
    #     end=235821031
    # else:
    #     start=235821032
    #     end = vocab.bytes 
    # fi.seek(start)

    #print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)
    l=len(fi)
    print(l)
    if pid == 0:
        start=0
        end=int(l/2)
    else:
        start=int(l/2)
        end=l

    starting_alpha = 0.001

    word_count = 0
    last_word_count = 0

    # while fi.tell() < end:
    #     line = fi.readline().strip()
    #     # Skip blank lines
    #     if not line:
    #         continue
    while start<end:
        line=fi[start].strip()

        # Init sent, a list of indices of words in line
        #sent = vocab.indices(['<bol>'] + line.split(",") + ['<eol>'])
        sent = [vocab.vocab_hash[x] for x in line.split(",")]
        grids= [vocab[x].grid for x in sent]
        tokens=[sent[0]]
        for sent_pos, token in enumerate(sent):
            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count

                # Recalculate alpha
                #alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                alpha = starting_alpha
                #if alpha < starting_alpha * 0.01: alpha = starting_alpha * 0.01
                

                # Print progress info
                sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                 (alpha, global_word_count.value, vocab.word_count,
                                  float(global_word_count.value) / vocab.word_count * 100))
                sys.stdout.flush()

            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win+1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # Turn into an iterator?
            
            # CBOW
            if cbow:
                # Compute neu1
                neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                assert len(neu1) == dim, 'neu1 and dim do not agree'

                if sent_pos != len(sent)-1:
                    tokens.append(sent[sent_pos+1])
                if sent_pos > 1:
                    del(tokens[0])
                # Init neu1e with zeros
                neu1e = np.zeros(dim)
                
                tokens1=list(set(tokens))
                # Compute neu1e and update syn1
                if neg > 0:
                    samplegrids=grids[sent_pos]
                    classifiers = [(t, 1)for t in tokens1] + [(target, 0) for target in table.sample(neg,samplegrids) if target not in  tokens1 ]
                else:
                    classifiers = zip(vocab[token].path, vocab[token].code)
                for target, label in classifiers:
                    z = np.dot(neu1, syn1[target])
                    p = sigmoid(z)
                    g = alpha * (label - p)
                    neu1e += g * syn1[target] # Error to backpropagate to syn0
                    syn1[target] += g * neu1  # Update syn1

                # Update syn0
                for context_word in context:
                    syn0[context_word] += neu1e

            # Skip-gram
            else:
                for context_word in context:
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        z = np.dot(syn0[context_word], syn1[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * syn1[target]              # Error to backpropagate to syn0
                        syn1[target] += g * syn0[context_word] # Update syn1

                    # Update syn0
                    syn0[context_word] += neu1e

            word_count += 1
        start += 1

    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, vocab.word_count,
                      float(global_word_count.value)/vocab.word_count * 100))
    sys.stdout.flush()
    #fi.close()

def save(vocab, syn0, fo, binary):
    print ('Saving model to')   , fo
    l=0
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(vocab, syn0):
            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))
            l+=1
    print("vec num is",l)

    fo.close()

def __init_process(*args):
    global vocab, syn0, syn1, table, cbow, neg, dim, alpha
    global win, num_processes, global_word_count, fi
    
    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, alpha, win, num_processes, global_word_count = args[:-1]
    with open(args[-1], 'r') as f:
        fi=f.readlines()
        random.shuffle(fi)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)

def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary, gi,epoch):
    # Read train file to init vocab
    vocab = Vocab(fi, min_count, gi)

    # Init net
    syn0, syn1 = init_net(dim, len(vocab))
    
    global_word_count = Value('i', 0)   
    table = None
    if neg > 0:
        print ('Initializing unigram table')
        table = UnigramTable(vocab)
    else:
        print ('Initializing Huffman tree')
        vocab.encode_huffman()

    # Begin training using num_processes workers
    t0 = time.time()
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
                        win, num_processes, global_word_count,fi))
    pool.map(train_process, range(num_processes))
    pool.terminate()
    t1 = time.time()
    print
    print ('Completed training. Training took'), (t1 - t0) / 60, 'minutes'
    save(vocab, syn0, fo, binary)

    # Save model to file
    
    print("embedding finish")

if __name__ == '__main__':
    train('walkseq_all20219239.txt', 'output.txt', bool(1), 40, 5, 0.00090, 10,
        0, 2, bool(0), "road2grids.txt",5)