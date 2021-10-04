from utils import generate_net,road_net
from deepwalk import deepwalk
#from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#先根据路网数据生成deepwalk序列数据
G=generate_net('')
d=deepwalk(walknum=20,min_road=8,max_length_perwalk=1000.,alpha=0.75,current_roadnet=G,sample_p=1,min_length_perwalk=50)
d.generate_seq()

#对deepwalk数据word2vec得到道路的向量表示

#seq=pd.read_csv('data\output\walkseq_all20217159.txt',sep='/n',encoding='utf-8',header=None)
#seq.columns=['txt']
#seq['seq']=seq['txt'].apply(lambda x :x.split(','))
#word2v=Word2Vec(min_count=1,vector_size=256,compute_loss=True,negative=8,window=8,workers=4,sentences=[x for x in seq['seq']],epochs=100)
#road_list=[G[x][y]['rskey'] for x,y in G.edges]
#//word2v.build_vocab()
#word2v.train(sentences=[x for x in seq['seq']],total_examples=word2v.corpus_count,epochs=100,)
#word2v.save("road2vec_all_sh.model")


a=1
