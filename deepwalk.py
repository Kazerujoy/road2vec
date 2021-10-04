import numpy as np 
import random
from utils import *
import math 
#from gensim.models import Word2Vec
import time
from tqdm import tqdm
# global current_roadnet
# current_roadnet=road_net()

class deepwalk():
    def __init__(self, walknum, min_road, min_length_perwalk, max_length_perwalk, alpha, current_roadnet, output_dir='data//output', walk_seq=[], sample_p=1):
        """
        walknum: 每个序列的步数
        min_raod: 每次采样至少采的路数量
        max_length_perwalk: 最多采样多少距离
        alpha: 采样策略系数
        sample_p：节点采样概率，调试用
        """
        self.walknum=walknum
        self.min_road = min_road
        self.max_length_perwalk = max_length_perwalk 
        self.alpha = alpha
        self.min_length_perwalk=min_length_perwalk
        if len(walk_seq)==0:
            self.walk_seq= []
        self.output_dir=output_dir+'//'
        assert type(current_roadnet)==road_net
        self.current_roadnet=current_roadnet
        self.sample_p=sample_p
    
    def get_a_walk(self, start_edge):
        path=[]
        path.append((self.current_roadnet[start_edge[0]][start_edge[1]])['rskey'])
        walk_len=round((0.5*random.random()+0.5) * self.max_length_perwalk,0)
        min_len=round((random.random()+1.0 )* self.min_length_perwalk,0)
        wl=0
        start=np.random.choice(start_edge)
        temp_s=start
        seq_len = 0.
        while len(path)<self.min_road:
            road_len=0.
            while road_len<min_len:
                temp_s,road_l,rskey = self.choose_succ(temp_s,seq_len,path)
                road_len+=road_l
            seq_len+=road_len
            path.append(rskey)
        while seq_len<walk_len:
            road_len=0.
            while road_len<min_len:
                temp_s,road_l,rskey = self.choose_succ(temp_s,seq_len,path)
                road_len+=road_l
            seq_len+=road_len
            path.append(rskey)
        self.walk_seq.append(path)
            
    def choose_succ(self,start, seq_len,path):
        #succ=[x[0] for x in list(self.current_roadnet.get_succ(start))]
        succ=[x[0] for x in list(self.current_roadnet.get_neighbour(start))]
        #为了减少轨迹loop的情况，要根据已有的seq来降低选择loop路径的概率
        succ_nei=[list(self.current_roadnet.neighbors(x)) for x in succ]
        #succ_nei=[list(self.current_roadnet.neighbors(x))+[x] for x in succ]

        road_nei=[]
        for i in range(len(succ)):
            road_nei.append([self.current_roadnet[succ[i]][x]['rskey'] for x in succ_nei[i]]+[self.current_roadnet[start][succ[i]]['rskey']])
        loop_list=dict()
        for i in range(len(succ)):
            if set(road_nei[i])&set(path) != set():
                loop_list[succ[i]]=self.alpha
            else:
                loop_list[succ[i]]=0
        s_choice=-1
        choice_p=[self.current_roadnet.degree(x) for x in succ]
        pc=sum(choice_p)
        choice_p=[x/pc for x in choice_p]
        while True:
            s_choice=np.random.choice(succ,p=choice_p)
            rand=random.random()
            if rand>=loop_list[s_choice]:
                break
        road=self.current_roadnet[start][s_choice]
        r,rskey=road["length"],road['rskey']
       
        return s_choice,r,rskey
        
    
    def seq2txt(self):
        def list2str(x):
            string=''
            for i in x:
                    string+=str(i)+','
            string=string[:-1]
            return string
        walkstr= [list2str(x) for x in self.walk_seq]
        t=time.gmtime()
        with open('walkseq_all'+str(t.tm_year)+str(t.tm_mon)+str(t.tm_mday) +str(t.tm_hour)+'.txt','w',encoding='utf-8') as f:    
            for i in walkstr:
                f.write(i+'\n')
        
    def generate_seq(self):
        edges=list(self.current_roadnet.edges)
        #sample_nodes=random.sample(nodes,int(round(self.sample_p*len(nodes),0)))
        for i in tqdm(range(len(edges))):
            for j in range(self.walknum):
                self.get_a_walk(edges[i])
        self.seq2txt()
        
    
    
    
        
            
            