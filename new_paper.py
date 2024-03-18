import pdb
import networkx as nx
import random
import numpy as np
import logging
import copy
from queue import PriorityQueue
import sys
import heapq
import time 
import queue
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO,filename='paper_peeling4test.log',filemode='w')
logger = logging.getLogger()


def get_data(path)->list:
    data = []
    upper = []
    lower = []
    edge = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            if line =="\n":
                continue
            line = line.strip('\n')
            dic = {}
            upper.append('1'+line.split(' ')[0])
            lower.append('2'+line.split(' ')[1])
            l = line.split(' ')[2:]
            for index, v in enumerate(l, start=1):
                dic.update({'weight{}'.format(index):float(v)})
            edge.append(dic)
    for i in zip(upper, lower, edge):
        data.append(i)
    return data

def build_bipartite_graph(data):
    G = nx.Graph()
    G.add_edges_from(data)
    #G.add_weighted_edges_from(data,weight='weight')
    #draw_graph(G)
    return G

        
def edge2weight(u, v, att):
    return (u, v), att


def get_alpha_beta_core_new_include_q(tmpG, q, alpha, beta):
    # 验证输入图是否是二部图
    bipartite_graph = copy.deepcopy(tmpG)
    if not nx.is_bipartite(bipartite_graph):
        raise nx.NetworkXError("The input graph is not bipartite.")

    # 初始化顶点度数
    degrees = dict(bipartite_graph.degree())

    # 初始化待移除顶点集
    to_remove = {n for n, d in degrees.items() if (n.startswith('1') and d < alpha) or (n.startswith('2') and d < beta)}

    while to_remove:
        node = to_remove.pop()
        neighbors = list(bipartite_graph.neighbors(node))
        bipartite_graph.remove_node(node)
        for neighbor in neighbors:
            degrees[neighbor] -= 1
            if (neighbor.startswith('1') and degrees[neighbor] < alpha) or (neighbor.startswith('2') and degrees[neighbor] < beta):
                to_remove.add(neighbor)
    connected_subgraphs = list(nx.connected_components(bipartite_graph))

    # 找到最大的连通子图
    # largest_connected_subgraph = max(connected_subgraphs, key=len)
    largest_connected_subgraph = []
    for i in connected_subgraphs:
        if q in i:
            largest_connected_subgraph = i

    # 创建一个子图，只包含最大连通子图的顶点
    largest_subgraph = bipartite_graph.subgraph(largest_connected_subgraph).copy()
    return largest_subgraph

def del_under_constrict(G, constrict, F=None):
    #edgelist.sort(key=lambda x:G.edges[x]['weight{}'.format(dim)])
    for edge in list(G.edges(data=True)):
        for k, v in constrict.items():
            if edge[2]['weight{}'.format(k)] < v:
                if F is not None and ((edge[0],edge[1]) in F or (edge[1],edge[0]) in F):
                    return False
                G.remove_edge(edge[0],edge[1])
                break
    return True

def peeling(G, q, alpha, beta, dim = 1, constrict = None, F=None):
    tmpG = copy.deepcopy(G)
    if constrict is not None:
        if del_under_constrict(tmpG, constrict, F) == False:
            return None
        else:
            tmpG = get_alpha_beta_core_new_include_q(tmpG, q, alpha, beta)
    logger.info('max include q has got {0} nodes and {1} edges'.format(tmpG.number_of_nodes(), tmpG.number_of_edges()))
    if F is not None:
        for i in F:
            if not tmpG.has_edge(i[0], i[1]):
                return None
    def get_min_edge(connectG, dim=1):
        edgelist = [i for i in connectG.edges(data=True)]
        edgelist.sort(key=lambda x:x[2]['weight{}'.format(dim)])
        for edge in edgelist:
            yield edge
    my_set = []
    my_queue = []
    max_iter = tmpG.number_of_edges()
    pbar = tqdm(total=max_iter, desc='Peeling dim:{}'.format(dim))
    count= 0
    while tmpG.number_of_edges() != 0:
        pbar.update(1)
        try:
            edge = next(get_min_edge(tmpG, dim))
            e, dic = edge2weight(*edge)
            if not tmpG.has_edge(e[0], e[1]):
                continue
            if F is not None and ((e[0],e[1]) in F or (e[1],e[0]) in F):
                return tmpG
            tmpG.remove_edge(e[0], e[1])
            my_set.append(edge)
            up = e[0] if e[0].startswith('1') else e[1]
            low = e[1] if e[1].startswith('2') else e[0]
            if tmpG.degree[up] < alpha and up not in my_queue:
                my_queue.append(up)
            if tmpG.degree[low] < beta and low not in my_queue:
                my_queue.append(low)
        except:
            break
        while len(my_queue) != 0:
            u = my_queue[0]
            my_queue.pop(0)
            if u == q:
                G1 = copy.deepcopy(tmpG)
                G1.add_edges_from(my_set)
                R = list(nx.connected_components(G1))
                for i in R:
                    if q in i:
                        print('****************** note 1 *******************')
                        return G1.subgraph(i).copy()
            for neighbor in list(tmpG[u]):
                edge_data = tmpG.get_edge_data(u, neighbor)
                if F is not None and ((u, neighbor) in F or (neighbor, u) in F):
                    G1 = copy.deepcopy(tmpG)
                    G1.add_edges_from(my_set)
                    R = list(nx.connected_components(G1))
                    for i in R:
                        if q in i:
                            print('****************** note 2 *******************')
                            return G1.subgraph(i).copy()
                tmpG.remove_edge(u, neighbor)
                my_set.append((u, neighbor, edge_data))
                if (neighbor.startswith('1') and tmpG.degree[neighbor] < alpha) or (neighbor.startswith('2') and tmpG.degree[neighbor] < beta):
                    my_queue.append(neighbor)
                    if neighbor == q:
                        G1 = copy.deepcopy(tmpG)
                        G1.add_edges_from(my_set)
                        R = list(nx.connected_components(G1))
                        for i in R:
                            if q in i:
                                print('****************** note 3 *******************')
                                return G1.subgraph(i).copy()
        my_set = []
        count += 1       
    return None

def expand(G, alpha, beta, q, dim=1):
    tmpG = copy.deepcopy(G)
    print('max include q has got', tmpG)
    
    def get_max_edge(connectG, dim=1):
        edgelist = [i for i in connectG.edges(data=True)]
        edgelist.sort(key=lambda x:x[2]['weight{}'.format(dim)], reverse=True)
        return edgelist
            
    def lemma1(connectG):
        return alpha * beta - alpha - beta <= connectG.number_of_edges() - connectG.number_of_nodes()

    def lemma2(connectG):
        number_upper, number_lower = 0, 0
        for item in list(connectG.nodes):
            if item.startswith('1') and connectG.degree(item) >= alpha:
                number_upper += 1
            elif item.startswith('2') and connectG.degree(item) >= beta:
                number_lower += 1
        return number_upper >= beta and number_lower >= alpha
    
    G1 = nx.Graph()
    node_edges = sorted(tmpG.edges(q, data=True), key=lambda x: x[2]['weight{}'.format(dim)], reverse=True)
    # 找到第4大的值，如果存在的话
    limt = alpha if q.startswith('1') else beta
    if len(node_edges) >= limt:
        th_largest_value = node_edges[limt-1][2]['weight{}'.format(dim)]
    else:
        th_largest_value = node_edges[-1][2]['weight{}'.format(dim)]
    print('find q th val', th_largest_value)
    
    edgelist = [i  for i in tmpG.edges(data=True) if i[2]['weight{}'.format(dim)] > th_largest_value]
    G1.add_edges_from(edgelist)
    tmpG.remove_edges_from(edgelist)
    print(G1.number_of_edges())
    print(tmpG.number_of_edges())
    print(G1.has_node(q))
    
    pre_len = 0
    edgelist = get_max_edge(tmpG, dim)
    index = 0
    max_iter = tmpG.number_of_edges()
    pbar = tqdm(total=max_iter, desc='expanding')
    while tmpG.number_of_edges() != 0:
        pbar.update(1)
        # print(tmpG.number_of_edges())
        try:
            edge = edgelist[index]
            index += 1
            e, dic = edge2weight(*edge)
            if not tmpG.has_edge(e[0], e[1]):
                continue
            tmpG.remove_edge(e[0], e[1])
            G1.add_edge(*e, **dic)
            C1 = []
            C1_list = list(nx.connected_components(G1))
            for i in C1_list:
                if q in i:
                    C1 = i
            C1G = G1.subgraph(C1).copy()
            if C1G.number_of_edges() == pre_len or not lemma1(C1G) or not lemma2(C1G):
                continue
            if C1G.number_of_edges() >= 1.05*pre_len:
                pre_len = C1G.number_of_edges()
            else:
                continue
            C1G = get_alpha_beta_core_new_include_q(C1G, q, alpha, beta)
            if C1G.has_node(q):
                return peeling(C1G, q, alpha, beta, dim)
        except:
            break
    C1 = []
    C1_list = list(nx.connected_components(G1))
    for i in C1_list:
        if q in i:
            C1 = i
    C1G = G1.subgraph(C1).copy()
    C1G = get_alpha_beta_core_new_include_q(C1G, q, alpha, beta)
    if C1G.has_node(q):
        return peeling(C1G, q, alpha, beta, dim)
    return None
            
def gpeel1D2f(G, dim):
    if G is None:
        return -1
    else:
        minedge = min(G.edges(data=True), key=lambda x:x[2]['weight{}'.format(dim)])
    return minedge[2]['weight{}'.format(dim)]


def peeling2D(G, q, alpha, beta, F=None, I=None):
    tmpG = copy.deepcopy(G)
    f2 = gpeel1D2f(peeling(tmpG, q, alpha, beta, 2, I, F), 2)
    R = []
    if I is None:
        constrict = {'1':0, '2':0}
    else:
        constrict = copy.deepcopy(I)
        constrict.update({'1':0, '2':0})
    while f2 > 0:
        constrict['1'] = 0
        constrict['2'] = f2
        f1 = gpeel1D2f(peeling(tmpG, q, alpha, beta, 1, constrict, F), 1)
        if f1 == -1:
            break
        for i in R:
            if f1 <= i[0] and f2 <= i[1]:
                return R
        R.append((f1, f2))
        logger.info('当前凝聚子图：{}'.format(R))
        constrict['2'] = 0
        constrict['1'] = f1 + 0.01
        f2 = gpeel1D2f(peeling(tmpG, q, alpha, beta, 2, constrict, F), 2)
    return R
    

def GetCandVal(G, alpha, beta, q, dim=3):
    tmpG = get_alpha_beta_core_new_include_q(G, q, alpha, beta)
    logger.info('max include q has got {}'.format(tmpG))
    Fdim = []
    def get_min_edge(connectG, dim=1):
        edgelist = [i for i in connectG.edges(data=True)]
        edgelist.sort(key=lambda x:x[2]['weight{}'.format(dim)])
        for edge in edgelist:
            yield edge
    my_queue = []
    max_iter = tmpG.number_of_edges()
    pbar = tqdm(total=max_iter, desc='Finding F{}'.format(dim))
    while tmpG.number_of_edges() != 0:
        pbar.update(1)
        try:
            edge = next(get_min_edge(tmpG, dim))
            e, dic = edge2weight(*edge)
            Fdim.append(dic['weight{}'.format(dim)])
            if not tmpG.has_edge(e[0], e[1]):
                continue
            tmpG.remove_edge(e[0], e[1])
            up = e[0] if e[0].startswith('1') else e[1]
            low = e[1] if e[1].startswith('2') else e[0]
            if tmpG.degree[up] < alpha and up not in my_queue:
                my_queue.append(up)
            if tmpG.degree[low] < beta and low not in my_queue:
                my_queue.append(low)
        except:
            break
        while len(my_queue) != 0:
            u = my_queue[0]
            my_queue.pop(0)
            for neighbor in list(tmpG[u]):
                tmpG.remove_edge(u, neighbor)
                if (neighbor.startswith('1') and tmpG.degree[neighbor] < alpha) or (neighbor.startswith('2') and tmpG.degree[neighbor] < beta):
                    my_queue.append(neighbor)
        tmpG = get_alpha_beta_core_new_include_q(tmpG, q, alpha, beta)      
    return Fdim


def peeling3D(G, q, alpha, beta, F=None, I=None):
    tmpG = copy.deepcopy(G)
    edgedic= {}
    for edge in tmpG.edges():
        try:
            edgedic[tmpG.edges[edge]['weight{}'.format(3)]].append(edge)
        except:
            edgedic.update({tmpG.edges[edge]['weight{}'.format(3)]:[edge]})
    F3 = GetCandVal(tmpG, alpha, beta, q, 3)
    R = []
    S = []
    F3.reverse()
    logger.info('F3总个数{}'.format(len(F3)))
    if I is None:
        tI = {'3':0}
    else:
        tI = copy.deepcopy(I)
        tI.update({'3':0})
    for f3 in tqdm(F3, desc='F3总轮次'):
        if F is None:
            tF = []
        else:
            tF = copy.deepcopy(F)
        edgeslist = edgedic[f3]
        for edge in edgeslist:
            tF.append(edge)
            tmpf1 = tmpG.edges[edge]['weight{}'.format(1)]
            tmpf2 = tmpG.edges[edge]['weight{}'.format(2)]
            ifcontinue = False
            for prer in S:
                if tmpf1 <= prer[0] and tmpf2 <= prer[1]:
                    ifcontinue = True
                    break
            if ifcontinue == True:
                break
            tI['3'] = f3
            T = peeling2D(tmpG, q, alpha, beta, tF, tI)
            for item in T:
                flag = False
                for prer in S:
                    if item[0] <= prer[0] and item[1] <= prer[1]:
                        flag = True
                        break
                if flag == False:
                    S.append(item)
                    R.append(item+(f3,))
    return R
                
def peelingHighD(G, q, alpha, beta, dim=4, F=None, I=None):
    tmpG = copy.deepcopy(G)
    if dim == 3:
        return peeling3D(tmpG, q, alpha, beta, F, I)
    edgedic= {}
    for edge in tmpG.edges():
        try:
            edgedic[tmpG.edges[edge]['weight{}'.format(dim)]].append(edge)
        except:
            edgedic.update({tmpG.edges[edge]['weight{}'.format(dim)]:[edge]})
    Fd = GetCandVal(tmpG, alpha, beta, q, dim)
    R = []
    S = [] 
    Fd.reverse()
    logger.info('F{}总个数{}'.format(dim, len(Fd)))
    if I is None:
        tI = {str(dim):0}
    else:
        tI = copy.deepcopy(I)
        tI.update({str(dim):0})
    for fd in tqdm(Fd, desc='F{}总轮次'.format(dim)):
        if F is None:
            tF = []
        else:
            tF = copy.deepcopy(F)
        edgeslist = edgedic[fd]
        for edge in edgeslist:
            tF.append(edge)
            tmpf_lst = []
            for idx in range(dim-1):
                tmpf_lst.append(tmpG.edges[edge]['weight{}'.format(idx+1)])
            ifcontinue = False
            for prer in S:
                if all(x <= y for x, y in zip(tmpf_lst, prer)):
                    ifcontinue = True
                    break
            if ifcontinue == True:
                break
            tI[str(dim)] = fd
            T = peelingHighD(tmpG, q, alpha, beta, dim-1, tF, tI)
            for item in T:
                flag = False
                for prer in S:
                    if all(x <= y for x, y in zip(item, prer)):
                        flag = True
                        break
                if flag == False:
                    S.append(item)
                    R.append(item+(fd,))
    return R


if __name__ == "__main__":
    path = './arxiv4dim.txt'
    q = '116'
    alpha, beta = 8, 2
    # path = './dim4graph.txt'
    # q = '22'
    # alpha, beta = 2, 3
    data = get_data(path)
    G = build_bipartite_graph(data)
    connect_subgraph = get_alpha_beta_core_new_include_q(G, q, alpha, beta)
    origG = G.subgraph(connect_subgraph).copy()
    nx.set_edge_attributes(G, 0, "visited")
    starttime = time.time()
    # print(GetCandVal(origG, alpha, beta, q, 3))
    # print(get_alpha_beta_core_new_include_q(G,q,alpha,beta))
    # pdb.set_trace()
    # print(peeling(origG, q, alpha, beta, 1))
    # print(expand(origG, alpha, beta, q, 1))
    # print(peeling2D(origG, q, alpha, beta))
    res = peelingHighD(origG, q, alpha, beta, 4)
    print(res)
    logger.info(res)
    endtime = time.time()
    print(endtime - starttime)
    logger.info(endtime - starttime)
