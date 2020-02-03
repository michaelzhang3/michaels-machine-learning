#coding:utf-8

'''
哈夫曼树
'''
import collections as cl
import numpy as np
class HafmanTree():

    def __init__(self,words_pairs):
        #所有的word都是叶子节点
        self.words_dic_pairs=[dict(zip(['name','count','leaf'],[word,count,True])) for word,count in words_pairs]
    #利用生成器生成每个单词的随机embeding 向量
    def generateRandom(self):
        np.random.seed(20)
        embedings = np.random.randn(len(self.words_dic_pairs), 3)
        for emb in embedings:
            yield emb
    #递归构造哈夫曼树
    def build(self,genetatorRand):
        sorted_pairs=sorted(self.words_dic_pairs,key=lambda x:x['count'])
        if len(sorted_pairs)>1:
            inner_dict=dict(zip(['name','count','leaf'],['inner' if len(sorted_pairs)>2 else 'root',sorted_pairs[0]['count']+sorted_pairs[1]['count'],False]))
            inner_dict['left_child']=sorted_pairs[0]
            inner_dict['right_child']=sorted_pairs[1]
            if not inner_dict['leaf']:
                if genetatorRand==None:
                    genetatorRand=self.generateRandom()
                inner_dict['weight']=list(next(genetatorRand))
            del sorted_pairs[0:2]
            sorted_pairs.insert(0,inner_dict)
            self.words_dic_pairs=sorted_pairs
            self.build(genetatorRand)

if __name__=='__main__':
    haf=HafmanTree([('zhang',10),('zhao',9),('liu',5),('wu',6),('shen',2),('hao',1),('zhou',3)])
    haf.build(None)
    print(haf.words_dic_pairs)



