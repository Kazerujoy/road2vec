import pandas as pd 
import networkx as nx 

def cal_distance(x,y):
    a=[0.997080300394605, 4.52309073713495E-4, -1.758468684565673E-4, 5.025780007095211E-7]
    RADIUS = 6371000.0
    [lon1,lat1],[lon2,lat2]=x,y
    dx = lon1 - lon2
    dy = lat1 - lat2
    b = (lat1 + lat2)/2.0
    Lx = RADIUS *(a[3] * b * b * b + a[2] * b * b + a[1] * b + a[0]) * math.radians(dx)
    Ly = RADIUS * math.radians(dy)
    return Lx*Lx+Ly*Ly

def generate_net(path):
    ve=pd.read_csv(path+'vertexs.csv',sep=',')
    #ve=pd.read_csv(path+'vertexs_all_sh.csv',sep=',')
    ed=pd.read_csv(path+'edges.csv',sep=',')
    #ed=pd.read_csv(path+'edges_all_sh.csv',sep=',')
    nodes=tuple(zip(ve['vertexID'].values,tuple(zip(ve['lonlat'],ve['gridID']))))
    edges=tuple(zip(ed['pre'].values,ed['suc'].values,tuple(zip(ed['rskey'],ed['speedlimit'],ed['length'],ed['orientation']))))
    G=road_net()
    for i in range(len(nodes)):
        G.add_node(nodes[i][0],lonlat=nodes[i][1][0],girdID=nodes[i][1][1])
    for i in range(len(edges)):
        G.add_edge(edges[i][0],edges[i][1],rskey=edges[i][2][0],speedlimit=edges[i][2][1],length=edges[i][2][2],orientation=edges[i][2][3])
    return G

class road_net(nx.Graph):
    
    # def get_succ(self,pred):
    #     return [(x,self[pred][x]['length'],self[pred][x]['orientation']) for x in self._succ[pred]]
    
    # def get_pred(self,succ):
    #     return [(x,self[x][succ]['length'],self[x][succ]['orientation']) for x in self._pred[succ]]
    def get_neighbour(self,node):
        return [(x,self[x][node]['length'],self[x][node]['orientation']) for x in self.neighbors(node)]
    
    def subg_by_gridid(self, gridid):
        if type(gridid)==int:
            gridid=[gridid]
        assert type(gridid)==list
        nodes= [x for x in self.nodes if self.nodes[x]['girdID'] in gridid]
        edges= []
        for i in nodes:
            edges += [x for x in self.neighbors(i)]
        nodes += [x for x in edges]
        nodes = list(set(nodes))
        return self.subgraph(nodes)
    
    def cal_lonlat_from_road(self,pred,succ,p):
        if  self.has_edge(pred,succ)==False:
            return [-1.,-1.]
        predl=self.nodes[pred]['lonlat'].split(',')
        succl=self.nodes[pred]['lonlat'].split(',')
        return [float(predl[0])+p*(float(succl[0])-float(predl[0])),float(predl[1])+p*(float(succl[1])-float(predl[1]))]
    

def __main__():
    G=generate_net('data//')
    sg=G.subg_by_gridid(13380)
    r=sg.cal_lonlat_from_road(276172, 288775,0.5)
    a=1
    
    
    