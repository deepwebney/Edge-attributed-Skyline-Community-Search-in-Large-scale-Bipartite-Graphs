import pdb
import networkx as nx
import random
import numpy as np
#import matplotlib.pyplot as plt
import logging
import copy
from queue import PriorityQueue
import sys
from utils import process_dataset, graph_from_dataset, origraph2maxsub
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,filename='paper.log',filemode='a')
logger = logging.getLogger()


upper = []
lower = []
edge = []


def get_data(path)->list:
    data = []
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
                dic.update({'weight{}'.format(index):int(v)})
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

def add_new_dim(number, lower, upper, file):
    """
    args
    number:the number of edges
    lower:random lower bound
    upper:random upper bound
    file:the merged file
    """
    a = random.sample(range(lower, upper), number)
    with open(file, 'r') as f:
        for n, line in zip(a, f.readlines()):
            new_line = line.rstrip('\n').split(' ')
            new_line.append(str(n))
            new_l = ' '.join(new_line)
            print(new_l)



def get_alpha_beta_core(G,alpha,beta):
    if alpha>beta:
        uporlow = "1"
    else:
        uporlow = "2"
    res = nx.k_core(G,k=min(alpha,beta))
    if alpha==beta:
        return res
    flag = 1
    while flag == 1:
        tf = 0
        degree = res.degree()
        nodes = list(res.nodes())
        for node in nodes:
            if degree[node] < max(alpha,beta) and node[0]==uporlow:
                res.remove_node(node)
                tf = 1
        if tf == 0:
            flag = 0
        res = nx.k_core(res,k = min(alpha,beta))
    return res

def get_max_include_q_com(G,q,alpha,beta,F=None):
    maxcount = 0
    maxresG = []
    resG = []
    temp = list(nx.connected_components(G))
    for i in temp:
        if q in i:
            tempG = get_alpha_beta_core(G.subgraph(i).copy(),alpha,beta)
            if q not in tempG.nodes:
                continue
            if len(tempG.nodes) > maxcount:
                resG = tempG.nodes
                if len(resG) > maxcount:
                    maxcount = len(resG)
                    maxresG = resG
    return G.subgraph(maxresG).copy()
    
    """
    res =[]
    temp = list(nx.connected_components(G))
    for i in temp:
        if q in i:
            print(i)
            res.append(i)
    largest = max(res,key=len)
    return G.subgraph(largest).copy()
    """

def get_min_edge(G, dim=1):
    edgelist = [i for i in G.edges()]
    edgelist.sort(key=lambda x:G.edges[x]['weight{}'.format(dim)])
    edgeWeig = [G.edges[i]['weight{}'.format(dim)] for i in edgelist]
    res = []
    for i in edgelist:
        if G.edges[i]['weight{}'.format(dim)] == edgeWeig[0]:
            res.append(i)
    return res, edgeWeig[0] if len(edgeWeig)>0 else 0

def get_max_edge(G, dim=1):
    edgelist = [i for i in G.edges(data=True)]
    edgelist.sort(key=lambda x:G.edges[(x[0], x[1])]['weight{}'.format(dim)], reverse=True)
    for edge in edgelist:
        yield edge

def DFSCom(G, u, alpha, beta, q, F=None)->int:
    yueshu = alpha
    if u[0] == '2':
        yueshu = beta
    degree = G.degree(u)-1
    if u==q and G.degree(u)<=yueshu:
        return 0
    if degree <yueshu:
        whetherdel = []
        edge = []
        for neighbor in list(G[u]):
            if G.edges[neighbor,u]['visited']==1:
                continue
            logger.info("visit edge is:{}".format(G.edges[neighbor,u]))
            G.edges[neighbor,u]['visited']=1
            if F is not None and ((neighbor,u) in F or (u, neighbor) in F):
                whetherdel.append(0)
            else:
                whetherdel.append(DFSCom(G, neighbor, alpha, beta, q, F))
            edge.append((u, neighbor))
            if sum(whetherdel)!=len(whetherdel):
                return 0
        if sum(whetherdel)==len(whetherdel):
            for e in edge:
                G.remove_edge(e[0], e[1])
            return 1
        else:
            return 0
    return 1


def del_under_constrict(G, constrict, dim=None, F=None):
    edgelist = [i for i in G.edges()]
    #edgelist.sort(key=lambda x:G.edges[x]['weight{}'.format(dim)])
    for edge in edgelist:
        if G.edges[edge]['weight{}'.format(dim)] < constrict:
            if F is not None and ((edge[0],edge[1]) in F or (edge[1],edge[0]) in F):
                return False
            G.remove_edge(edge[0],edge[1])
    return True

def peeling(G, q, alpha, beta, dim = 1, constrict = None, F=None):
    tmpG = copy.deepcopy(G)
    if constrict is not None:
        for key, value in constrict.items():
            checkpoint = del_under_constrict(tmpG, value, int(key), F)
            if checkpoint == False:
                return 0
            #print("去除不满足约束的边，图还剩下的边",tmpG.edges)
    teG = get_max_include_q_com(tmpG, q, alpha, beta)
    if F is not None:
        el = list(teG.edges)
        for u, v in F:
            if (u, v) not in el and (v, u) not in el:
                return 0
    fi = 0
    check = True
    while(teG.size()!=0 and check):
        minedgelist, new_fi = get_min_edge(teG, dim)
        fi = max(fi, new_fi)
        nx.set_edge_attributes(teG, 0, "visited")
        for minedge in minedgelist:
            teG.edges[minedge]['visited']=1
            flag = DFSCom(teG, minedge[0], alpha, beta, q, F)
            if flag == 0:
                check = False
                break
            flag = DFSCom(teG, minedge[1], alpha, beta, q, F)
            if flag == 0:
                check = False
                break
            #print("del-edges:({0},{1})".format(minedge[0],minedge[1]))
            if F is not None:
                if (minedge[0], minedge[1]) in F or (minedge[1], minedge[0]) in F:
                    return fi
            teG.remove_edge(minedge[0], minedge[1])
        teG = get_max_include_q_com(teG, q, alpha, beta)
    #print(teG.edges(data=True))
    return fi

def GetCandVal(G, d, alpha, beta, q):
    def DFS(G, u, alpha, beta):
        yueshu = alpha
        if u[0] == '2':
            yueshu = beta
        if G.degree(u) >= yueshu:
            return
        else:
            for neighbor in list(G[u]):
                if (u, neighbor) in list(G.edges()) or (neighbor, u) in list(G.edges()):
                    G.remove_edge(u, neighbor)
                    DFS(G, neighbor, alpha, beta)
        
    Td = []
    Gq = get_max_include_q_com(G, q, alpha, beta)
    while len(Gq.nodes) > 0:
        edges, fd = get_min_edge(Gq, d)
        Td.append(fd)
        for edge in edges:
            Gq.remove_edge(*edge)
            if edge[0] in list(Gq):
                DFS(Gq, edge[0], alpha, beta)
            if edge[1] in list(Gq):
                DFS(Gq, edge[1], alpha, beta)
            Gq = get_max_include_q_com(Gq, q, alpha, beta)
    return Td


def peelingHighD(G, q, alpha, beta, d, I={}, F=[]):
    if d==3:
        return peeling3D(G, q, alpha, beta, I, F)
    edgedic= {}
    for edge in G.edges():
        try:
            edgedic[G.edges[edge]['weight{}'.format(d)]].append(edge)
        except:
            edgedic.update({G.edges[edge]['weight{}'.format(d)]:[edge]})
    Fd = GetCandVal(G, d, alpha, beta, q)
    Fd.reverse()
    S = []
    R = []
    for fd in Fd:
        tF = copy.deepcopy(F)
        tmpG = copy.deepcopy(G)
        tmpfd = [sys.maxsize for _ in range(d-1)]
        edgeslist = edgedic[fd]
        for edge in edgeslist:
            tF.append(edge)
            for index in range(d-1):
                tmpfd[index] = min(tmpfd[index], int(tmpG.edges[edge]['weight{}'.format(index+1)]))
        tmpfd = tuple(tmpfd)
        flag = 0
        for item in S:
            ttf = False
            for tmp, it in zip(tmpfd, item):
                if tmp > it:
                    ttf = True
                    break
            if ttf == False:
                flag = 1
                break
        if flag == 1:
            continue
        tI = copy.deepcopy(I)
        tI.update({'{}'.format(d):fd})
        T = peelingHighD(tmpG, q, alpha, beta, d-1, tI, tF)
        resT = []
        for i in T:
            if len(S)==0:
                resT.append(i)
            else:
                ff = True
                for s in S:
                    tflag = False
                    for new, old in zip(i, s):
                        if new > old:
                            tflag = True
                            break
                    if tflag == True:
                        continue
                    else:
                        ff = False
                        break
                if ff == True:
                    resT.append(i)
        S.extend(resT)
        for tup in resT:
            R.append(tup + (fd, ))
        print(R)
    return R


def peeling3D(G, q, alpha, beta, I={}, F=[]):
    edgedic= {}
    for edge in G.edges():
        try:
            edgedic[G.edges[edge]['weight{}'.format(3)]].append(edge)
        except:
            edgedic.update({G.edges[edge]['weight{}'.format(3)]:[edge]})
    F3 = GetCandVal(G, 3, alpha, beta, q)
    F3.reverse()
    S = []
    R = []
    for f3 in F3:
        tF = copy.deepcopy(F)
        tmpG = copy.deepcopy(G)
        tmpf1, tmpf2 = sys.maxsize, sys.maxsize
        edgeslist = edgedic[f3]
        for edge in edgeslist:
            tF.append(edge)
            tmpf1 = min(tmpf1, tmpG.edges[edge]['weight{}'.format(1)])
            tmpf2 = min(tmpf2, tmpG.edges[edge]['weight{}'.format(2)])
        flag = 0
        for item in S:
            if tmpf1 <= item[0] and tmpf2 <= item[1]:
                flag = 1
                break
        if flag == 1:
            continue
        I.update({'3':f3})
        T = peeling2D(tmpG, q, alpha, beta, I, tF)
        resT = []
        for i in T:
            if len(S)==0:
                resT.append(i)
            else:
                ff = True
                for s in S:
                    if i[0] > s[0] or i[1] > s[1]:
                        continue
                    else:
                        ff = False
                        break
                if ff == True:
                    resT.append(i)
        S.extend(resT)
        for f1, f2 in resT:
            R.append((f1, f2, f3))
        print(R)
    return R


def peeling2D(G, q, alpha, beta, constrict=None, F=None):
    R = []
    if constrict is not None:
        for key, value in constrict.items():
            checkpoint = del_under_constrict(G, value, int(key), F)
            if checkpoint == False:
                return R
    dic = {'1':0,'2':0}
    #("dim=2")
    f2=peeling(G,q,alpha, beta, dim=2, F=F)
    while f2>0:
        dic['1']=0
        dic['2']=f2
        #print("dim=1")
        f1=peeling(G,q,alpha, beta, dim=1, constrict=dic, F=F)
        if f1 == 0:
            break
        R.append((f1,f2))
        dic['1']=f1 + 1e-5 #I更新x1>f1，删除<=
        dic['2']=0
        #print("dim=2")
        f2 = peeling(G,q,alpha, beta, dim=2,constrict=dic, F=F)
        print(R)
    return R


def generate_bipartite_graph(number, highrange):
    """
    a = random.sample(range(1, highrange), number*number)#generate graph
    nums = np.ones(number*number)
    nums[round(number*number*0.8):] = 0
    np.random.shuffle(nums)
    nums = nums.tolist()
    show = np.zeros((number,number))
    for i in range(len(show)):
        for j in range(len(show[0])):
            if nums[i*number+j]==1:
                show[i][j]=a[i*number+j]
    np.savetxt('generate_graph.txt', show, fmt='%d')
    """
    show = np.loadtxt("generate_graph.txt")
    with open("graph_generate_new.txt","w") as f:
        for i in range(len(show)):
            for j in range(len(show[0])):
                if show[i][j]!=0:
                    f.write("{0} {1} {2}".format(i, j, int(show[i][j]))+"\n")

def draw_graph(G):
    pos = nx.spring_layout(G)
    node_labels = {}
    for i in list(G.nodes):
        node_labels.update({i:i})
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # plt.show()
    # plt.savefig("Graph.jpg", format="JPG")

def significant_com(G, alpha, beta, query):
    """
    这个是sig-com论文用来验证的代码
    """
    tmpG = copy.deepcopy(G)
    s = []
    sw = []
    q = []
    dim = 1
    edgelist = [i for i in tmpG.edges()]
    edgelist.sort(key=lambda x:G.edges[x]['weight{}'.format(dim)])
    edgeWeig = [G.edges[i]['weight{}'.format(dim)] for i in edgelist]
    while(len(tmpG.edges)!=0):
        minedge = edgeWeig[0]
        for edge, w in zip(edgelist, edgeWeig):
            if w == minedge:
                s.append(edge)
                sw.append(w)
                tmpG.remove_edge(edge[0],edge[1])
                edgelist.remove(edge)
                edgeWeig.remove(w)
                if int(edge[0]) < int(edge[1]):
                    if tmpG.degree(edge[0]) < alpha and edge[0] not in q:
                        q.append(edge[0])
                    if tmpG.degree(edge[1]) < beta and edge[1] not in q:
                        q.append(edge[1])
                else:
                    if tmpG.degree(edge[1]) < alpha and edge[1] not in q:
                        q.append(edge[1])
                    if tmpG.degree(edge[0]) < beta and edge[0] not in q:
                        q.append(edge[0])
            else:
                break
        while len(q) != 0:
            pop = q[0]
            del(q[0])
            for neighbor in list(tmpG[pop]):
                tt = (pop, neighbor) if (pop, neighbor) in edgelist else (neighbor, pop)
                s.append(tt)
                sw.append(edgeWeig[edgelist.index(tt)])
                tmpG.remove_edge(pop, neighbor)
                edgeWeig.remove(edgeWeig[edgelist.index(tt)])
                edgelist.remove(tt)
                d = tmpG.degree(neighbor)
                if (neighbor[0] == '1' and d < alpha) or (neighbor[0]=='2' and d < beta):
                    q.append(neighbor)
                    if neighbor == query or pop == query:
                        new = []
                        for i, w1 in zip(s, sw):
                            tmp = i + ({'weight1':w1}, )
                            new.append(tmp)
                        tmpG.add_edges_from(new)
                        print(tmpG.edges())
                        return get_alpha_beta_core(tmpG, alpha, beta)
        s = []
        sw = []

def expand(G, alpha, beta, query, dim=1):
    tmpG = nx.Graph()
    def lemma1(connectG):
        e_G = connectG.number_of_edges()
        u_G, l_G = 0, 0
        for item in list(connectG.nodes):
            if item.startswith('1'):
                u_G += 1
            else:
                l_G += 1
        return alpha * beta - alpha - beta <= e_G - u_G - l_G

    def lemma2(connectG):
        number_upper, number_lower = 0, 0
        for item in list(connectG.nodes):
            if item.startswith('1') and connectG.degree(item) >= alpha:
                number_upper += 1
            elif item.startswith('2') and connectG.degree(item) >= beta:
                number_lower += 1
        return number_upper >= beta and number_lower >= alpha

    pre_connect_subgraph = 0
    for edge in get_max_edge(G, dim):
        e, dic = edge2weight(*edge)
        tmpG.add_edge(*e, **dic)
        connect_subgraphs = [i for i in nx.connected_components(tmpG)]
        connect_subgraphs.sort(key = lambda x:len(x), reverse=True)
        connect_subgraph = tmpG.subgraph(connect_subgraphs[0])
        if connect_subgraph.number_of_edges() > pre_connect_subgraph and lemma1(connect_subgraph) and lemma2(connect_subgraph):
            pre_connect_subgraph = connect_subgraph.number_of_edges()
            if query in list(connect_subgraph.nodes):
                H = get_alpha_beta_core(connect_subgraph, alpha, beta)
                if query in list(H.nodes):
                    return H 
    return None

def community2skylineValue(G, all_dim = 1):
    """
    convert the community to skyline value,for example G->(f1,f2,...)
    all_dim: the number of the attributes.
    """
    if G is None:
        return []
    edgelist = list(G.edges)
    res = []
    for dim in range(all_dim):
        edgelist.sort(key=lambda x:G.edges[x]['weight{}'.format(dim+1)])
        res.append(G.edges[edgelist[0]]['weight{}'.format(dim+1)])
    return res

def edge2weight(u, v, att):
    return (u, v), att

def expand2D(G, alpha, beta, query, fixed_edges):
    edgelist = [i for i in G.edges(data=True)]
    edgelist.sort(key=lambda x:G.edges[x[0], x[1]]['weight{}'.format(2)],reverse=True)
    edgeWeig_dim2 = [G.edges[i[0], i[1]]['weight{}'.format(2)] for i in edgelist]
    edgelist1 = copy.deepcopy(edgelist)
    edgelist1.sort(key=lambda x:G.edges[x[0], x[1]]['weight{}'.format(1)], reverse=True)
    edgeWeig_dim1 = [G.edges[i[0], i[1]]['weight{}'.format(1)] for i in edgelist1]
    tmpG = nx.Graph()
    q_weight = [G.edges[i, query]['weight{}'.format(2)] for i in G[query]]
    q_weight.sort(key=lambda x:x, reverse=True)
    R = []

    def erfen(lst, l, r, target):
        if l>r:
            return -1
        mid = (l+r) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            return erfen(lst, l, mid-1, target)
        else:
            return erfen(lst, mid+1, r, target)
    
    def check(G):
        tmpres = get_alpha_beta_core(G, alpha, beta)
        l = list(tmpres.edges)
        if len(l) > 0:
            if query in list(tmpres.nodes):
                for edge in fixed_edges:
                    if edge not in l and (edge[1], edge[0]) not in l:
                        return False
                return True
            return False
        return False


    f1 = -1
    tar = q_weight[alpha-1] if query[0]=='1' else q_weight[beta-1]
    idx = erfen(edgeWeig_dim2, 0, len(edgeWeig_dim2)-1, tar)
    #edgelistwithW = [edge2weight(*i) for i in edgelist[:idx]]
    tmpG.add_edges_from(edgelist[:idx])


    while tmpG.number_of_edges() < len(edgelist):
        if f1 == -1:
            e, dic = edge2weight(*edgelist[idx])
            tmpG.add_edge(*e, **dic)
            idx += 1
            D = 1
        else:
            t = edgeWeig_dim1.index(f1)
            e, _ = edge2weight(*edgelist1[t])
            tmpG.remove_edge(*e)
            #tmpedgelistwithW = [edge2weight(G, *i) for i in edgelist1[:t]]
            tmpG.add_edges_from(edgelist1[:t])
            D = 2
        tmpres = nx.Graph()
        while check(tmpG):
            tmpres = get_alpha_beta_core(tmpG, alpha, beta)
            edges, _ = get_min_edge(tmpG, dim=D)
            for edge in edges:
                tmpG.remove_edge(*edge)
        if len(list(tmpres)) == 0:
            if f1 == -1:
                continue
            else:
                break
        skylst = community2skylineValue(tmpres, 2)
        for index, sky in enumerate(R):
            flag = True
            for item1, item2 in zip(sky, skylst):
                if item1 > item2:
                    flag = False
                    break
            if flag == True:
                R.pop(index)

                
        R.append(skylst)
        f1 = skylst[0]
        tmpG = tmpres
    
    R = map(lambda x:tuple(x), R)
    return list(R)
        
def expand3D(G, alpha, beta, q, F=[]):
    R = []
    que = PriorityQueue()
    edgelist = [i for i in G.edges(data=True)]
    skyline1 = []
    skyline2 = []
    MAX = 0
    if expand(G, alpha, beta, q, 3) is None:
        return R
    if len(expand(G, alpha, beta, q, 3).edges) != 0:
        maxf3 = community2skylineValue(expand(G, alpha, beta, q, 3), 3)[2]
        MAX = maxf3
        que.put((MAX-maxf3, (0, 0)))
    record_point = []
    pre_f3 = -1
    pre_round_f3 = -1
    while not que.empty():
        S = []
        while not que.empty():
            tmp_tup = que.get()

            f3 = MAX - tmp_tup[0]
            if f3 == pre_round_f3:
                continue
            if f3 != pre_f3 and len(record_point) == 0:
                record_point.append(tmp_tup[1])
                pre_f3 = f3
            elif f3 == pre_f3:
                record_point.append(tmp_tup[1])
            else:
                que.put(tmp_tup)
                break
        f3 = pre_f3
        pre_round_f3 = f3
        pre_f3 = -1
        record_point = list(set(record_point))
        for item in record_point:
            tmpG = nx.Graph()
            f1 = item[0]
            f2 = item[1]
            tmpF = copy.deepcopy(F)
            for edge in edgelist:
                if edge[2]['weight{}'.format(3)] >= f3 and edge[2]['weight{}'.format(2)] > f2 and edge[2]['weight{}'.format(1)] > f1:
                    e, dic = edge2weight(*edge)
                    tmpG.add_edge(*e, **dic)
                if edge[2]['weight{}'.format(3)] == f3:
                    tmpF.append((edge[0], edge[1]))
            # import pdb;pdb.set_trace()
            S.extend(expand2D(tmpG, alpha, beta, q, tmpF))
            for tf1, tf2 in S:
                flag = 0
                for f1s, f2s in zip(skyline1, skyline2):
                    if tf1 >= f1s and tf2 >= f2s:
                        if tf1 == f1s and tf2 == f2s:
                            flag = 1
                        else:
                            skyline1.remove(f1s)
                            skyline2.remove(f2s)
                    
                if flag == 0:
                    R.append((tf1, tf2, f3))
                    skyline1.append(tf1)
                    skyline2.append(tf2)
            new_sky1 = skyline1[:]
            new_sky2 = skyline2[:]
            new_sky1.append(0)
            new_sky2.append(0)
            new_sky2.sort(reverse=True)
            new_sky1.sort()
            for point in zip(new_sky1, new_sky2):
                tmpG = nx.Graph()
                for edge in edgelist:
                    if edge[2]['weight{}'.format(1)] > point[0] and edge[2]['weight{}'.format(2)] > point[1]:
                        e, dic = edge2weight(*edge)
                        tmpG.add_edge(*e, **dic)
                skyline_list = community2skylineValue(expand(tmpG, alpha, beta, q, 3), 3)
                if len(skyline_list) >= 3:
                    maxdim3 = skyline_list[2]
                    que.put((MAX-maxdim3, (point[0], point[1])))
            print(R)
        record_point = []
        # pdb.set_trace()
    R = list(set(R))
    return R

def updatecpoint(C, s, d):
    resC = []
    C = list(map(list, C))
    Chat = []
    for c in C:
        flag = 1
        for a, b in zip(c, s):
            if a > b:
                flag = 0
                break
        if flag == 1:
            Chat.append(c)
    C = list(filter(lambda x: x not in Chat, C))
    C = list(map(tuple, C))
    for i in range(0, d):
        #import pdb;pdb.set_trace()
        restmpc = []
        ctmp = []
        Chat1 = copy.deepcopy(Chat)
        for c in Chat1:
            c[i] = s[i]
            ctmp.append(c)
        ctmp = list(set(map(tuple,ctmp)))
        for j in range(len(ctmp)): #被支配
            isd = False
            for k in range(len(ctmp)):
                if j == k:
                    continue
                dflag = True
                for x, y in zip(ctmp[j], ctmp[k]):
                    if x < y:
                        dflag = False
                        break
                if dflag == True:
                    isd = True
                    break
            if isd == False:
                restmpc.append(ctmp[j])
        resC.extend(restmpc)
    resC.extend(C)
    return resC
            


def expandHighD(G, alpha, beta, q, d, F=[]):
    if d == 3:
        return expand3D(G, alpha, beta, q, F)
    R = []
    globalS = []
    C = [tuple([0 for _ in range(d-1)])]
    que = PriorityQueue()
    edgelist = [i for i in G.edges(data=True)]
    MAX = 0
    if len(expand(G, alpha, beta, q, d).edges) != 0:
        maxfd = community2skylineValue(expand(G, alpha, beta, q, d), d)[d-1]
        MAX = maxfd
        tmptuple = tuple([0 for _ in range(d-1)])
        que.put((MAX-maxfd, tmptuple))
    record_point = []
    pre_fd = -1
    pre_round_fd = -1
    while not que.empty():
        S = []
        while not que.empty():
            tmp_tup = que.get()

            fd = MAX - tmp_tup[0]
            if fd == pre_round_fd:
                continue
            if fd != pre_fd and len(record_point) == 0:
                record_point.append(tmp_tup[1])
                pre_fd = fd
            elif fd == pre_fd:
                record_point.append(tmp_tup[1])
            else:
                que.put(tmp_tup)
                break
        fd = pre_fd
        pre_round_fd = fd
        pre_fd = -1
        record_point = list(set(record_point))
        for item in record_point:
            tmpG = nx.Graph()
            tmpF = copy.deepcopy(F)
            for edge in edgelist:
                attri = [edge[2]['weight{}'.format(i+1)] for i in range(d-1)]
                attriflag = True
                for ea, limit in zip(attri, item):
                    if ea <= limit:
                        attriflag = False
                        break
                if edge[2]['weight{}'.format(d)] >= fd and attriflag == True:
                    e, dic = edge2weight(*edge)
                    tmpG.add_edge(*e, **dic)
                if edge[2]['weight{}'.format(d)] == fd:
                    tmpF.append((edge[0], edge[1]))
            #import pdb;pdb.set_trace()
            S.extend(expandHighD(tmpG, alpha, beta, q, d-1, tmpF))
            for p in S:
                flag = 0
                for gp in globalS:
                    checkbing = True
                    for i in range(len(p)):
                        if p[i] < gp[i]:
                            checkbing = False
                            break
                    if checkbing == True:
                        if p == gp:
                            flag = 1
                        else:
                            globalS.remove(gp)
                if flag == 0:
                    R.append(p+(fd,))
                    globalS.append(p)

                C = updatecpoint(C, p, d-1)
            
            for point in C:
                tmpG = nx.Graph()
                for edge in edgelist:
                    attri = [edge[2]['weight{}'.format(i+1)] for i in range(d-1)]
                    attriflag = True
                    for ea, limit in zip(attri, point):
                        if ea <= limit:
                            attriflag = False
                            break
                    if attriflag == True:
                        e, dic = edge2weight(*edge)
                        tmpG.add_edge(*e, **dic)
                skyline_list = community2skylineValue(expand(tmpG, alpha, beta, q, d), d)
                if len(skyline_list) >= d:
                    maxdimd = skyline_list[d-1]
                    que.put((MAX-maxdimd, point))
            print(R)
        record_point = []
        # pdb.set_trace()
    R = list(set(R))
    return R



if __name__ == "__main__":
    #generate_bipartite_graph(10,150)
    data = graph_from_dataset(sys.argv[1], sys.argv[2])
    q = '17'
    alpha, beta = 2, 2
    G1 = build_bipartite_graph(data)
    G = get_alpha_beta_core(G1, alpha, beta)
    print(nx.is_bipartite(G))
    nx.set_edge_attributes(G, 0, "visited")
    starttime = datetime.now()
    # print(peeling3D(G,q,alpha,beta))
    #print(peeling2D(G, q, alpha, beta))
    # print(peelingHighD(G,q,alpha,beta,4))
    # print(peeling(G,q,alpha,beta))
    endtime = datetime.now()
    print('peeling', (endtime - starttime).microseconds)
    starttime = datetime.now()
    # print(expand3D(G, alpha, beta, q))
    # print(expand2D(G, alpha, beta, q, []))
    # print(expandHighD(G,alpha, beta,q,4))
    print(community2skylineValue(expand(G,alpha,beta,q)))
    endtime = datetime.now()
    print('expand', (endtime - starttime).microseconds)
    # 一维用法
    # print(peeling(G,q,alpha,beta))
    # print(community2skylineValue(expand(G,alpha,beta,q)))
    # 二维用法
    # print(peeling2D(G, q, alpha, beta))
    # print(expand2D(G, alpha, beta, q, []))
    # 三维用法
    # print(peeling3D(G,q,alpha,beta))
    # print(expand3D(G, alpha, beta, q))
    # 高维用法
    # print(peelingHighD(G,q,alpha,beta,4))
    # print(expandHighD(G,alpha, beta,q,4))