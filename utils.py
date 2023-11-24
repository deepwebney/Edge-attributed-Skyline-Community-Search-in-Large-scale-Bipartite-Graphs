import sys
import random
import numpy as np
import networkx as nx
import pickle
from tqdm import tqdm


def process_dataset(G):
    mapping = {}
    x,y = nx.algorithms.bipartite.sets(G)
    new_x = map(lambda x:'1'+x, x)
    new_y = map(lambda y:'2'+y, y)
    for key, value in zip(x, new_x):
        mapping.update({key:value})
    for key, value in zip(y, new_y):
        mapping.update({key:value})
    tmp_G = nx.relabel_nodes(G, mapping)
    return tmp_G

def graph_from_dataset(path, name):
    #import pdb;pdb.set_trace()
    #att = generate_attribute(1, 2, 4, 8)
    
    with open(name,"rb") as f:
        att=pickle.load(f)
    res_data = []
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        next(f)
        for line, att1, att2, att3, att4 in tqdm(zip(f.readlines(), att[0], att[1], att[2], att[3])):
            att_tmp = [att1, att2, att3, att4]
            line = line.strip('\n').strip().split(' ')
            begin = '1' + line[0]
            end = '2' + line[1]
            dic = {}
            for i, v in enumerate(att_tmp):
                dic.update({'weight{}'.format(i+1):float(v)})
                res_data.append((begin, end, dic))
    return res_data

def origraph2maxsub(data_lst, path):
    print(path)
    with open(path, 'w') as f:
        for i in data_lst:
            tmp = [i[0],i[1],i[2]['weight1'],i[2]['weight2'],i[2]['weight3'],i[2]['weight4']]
            tmp = list(map(str,tmp))
            f.write(' '.join(tmp)+ '\n')
            f.flush()

        


def generate_attribute(lower, upper, dim, number,name,sample_num):
    """
    下界，上界，维度，边数，数据集名称，采样数量
    """
    res = []
    for i in range(dim):
        lst = np.random.uniform(lower, upper, size=(sample_num,)).tolist()
        random.shuffle(lst)
        dim_lst = random.sample(lst, number)
        res.append(dim_lst)
    
    with open(name,"wb") as f:
        pickle.dump(res, f)
    return res


if __name__ == '__main__':
    #print(graph_from_dataset(sys.argv[1]))
    # generate_attribute(1,200,4,293697,'dbpedia.pkl',300000)
    # generate_attribute(1,200,4,3232134,'tvtropes.pkl',3500000)
    # generate_attribute(1,200,4,1149739,'bookcrossing.pkl',1200000)
    generate_attribute(1,200,4,3782463,'IMDB.pkl',3800000)
    #generate_attribute(1,200,4,1476,'crime.pkl',1500)
    generate_attribute(1,200,4,58595,'arXiv.pkl',59000)
    # data = graph_from_dataset(sys.argv[1], sys.argv[2])
