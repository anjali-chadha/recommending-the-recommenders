'''
Created on Mar 1, 2017
@author: v-lianji
'''

import pandas
import codecs
import pickle
from scipy.sparse import find
import numpy as np

class movie_lens_data_repos:
    def __init__(self, file):
        with codecs.open(file,'rb') as f:
            train, validate,test,user_content,item_content = pickle.load(f)

        # splitting the training matrix into two parts

        print(test.head(10))
        # print(test.isnull().values.any())
        utility_traindf = train.pivot(index='userId', columns='movieId', values='rating')
        utility_traindf = utility_traindf.fillna(0)
        utility_traindf1 = utility_traindf.iloc[:, 0:4000]
        # utility_traindf1.fillna(0)
        utility_traindf1 = utility_traindf.iloc[:, 0:4000]
        utility_traindf1['userId'] = utility_traindf1.index
        train_melt1 = pandas.melt(utility_traindf1, id_vars='userId')
        utility_traindf2 = utility_traindf.iloc[:, 4000:]
        utility_traindf2['userId'] = utility_traindf2.index
        train_melt2 = pandas.melt(utility_traindf2, id_vars='userId')
        train1 = train_melt1
        train2 = train_melt2

        # splitting the testing matrix into two parts
        utility_testdf = test.pivot(index='userId', columns='movieId', values='rating')
        utility_testdf = utility_testdf.fillna(0)
        print("#############################################")
        print(len(utility_testdf))
        print("#############################################")
        # utility_testdf1 = utility_testdf.iloc[:, 0:4000]
        # utility_testdf1.fillna(0)
        utility_testdf1 = utility_testdf.iloc[:, 0:659//2]
        # print(utility_testdf1.isnull().values.any())
        utility_testdf1['userId'] = utility_testdf1.index
        test_melt1 = pandas.melt(utility_testdf1, id_vars='userId')
        utility_testdf2 = utility_testdf.iloc[:, 659//2:]
        # print(utility_testdf2.isnull().values.any())
        utility_testdf2['userId'] = utility_testdf2.index
        test_melt2 = pandas.melt(utility_testdf2, id_vars='userId')
        test1 = test_melt1
        test2 = test_melt2

        # print(test1.isnull().values.any())
        # print(test2.isnull().values.any())

        #train_melt2.head(5)
        # print(type(item_content))
        # print(item_content.shape)
        # print("n=", n)

        #item_content_1 = item_content[:int(n/2)][:]
        #item_content_2 = item_content[int(n/2):][:]
        #train = train.reindex(np.random.permutation(train.index))

        self.training_ratings_user_1 = train1.loc[:,'userId']
        self.training_ratings_user_2 = train2.loc[:,'userId']
        self.training_ratings_item_1 = train1.loc[:,'movieId']
        self.training_ratings_item_2 = train2.loc[:,'movieId']
        self.training_ratings_score_1 = train1.loc[:,'value']
        self.training_ratings_score_2 = train2.loc[:,'value']

        self.testing_ratings_user_1 = test1.loc[:,'userId']
        self.testing_ratings_user_2 = test2.loc[:,'userId']
        self.testing_ratings_item_1 = test1.loc[:,'movieId']
        self.testing_ratings_item_2 = test2.loc[:,'movieId']
        self.testing_ratings_score_1 = test1.loc[:,'value']
        self.testing_ratings_score_2 = test2.loc[:,'value']

        # print("LENGTH OFFFFFFFFFFFFFF 1111: ", self.training_ratings_item_1)
        # print("LENGTH OFFFFFFFFFFFFFF 222222222: ", self.training_ratings_item_1)

        # self.test_ratings_user = validate.loc[:,'userId']

        # self.test_ratings_item = validate.loc[:,'movieId']

        # self.test_ratings_score = validate.loc[:,'rating']

        self.eval_ratings_user = test.loc[:,'userId']

        self.eval_ratings_item_1 = test.loc[:,'movieId']
        #self.eval_ratings_item_2 = test.loc[:,'item_2']

        self.eval_ratings_score = test.loc[:,'rating']

        self.n_user = int(max([self.training_ratings_user_1.max(), self.testing_ratings_user_1.max(),self.eval_ratings_user.max()])+1)

        # self.n_item_1 = int(max([self.training_ratings_item_1.max(),self.test_ratings_item_1.max(),self.eval_ratings_item_1.max()])+1)
        self.n_item_1 = self.training_ratings_item_1.max()
        # self.n_item_2 = int(max([self.training_ratings_item_2.max(), self.test_ratings_item_2.max(), self.eval_ratings_item_2.max()]) + 1)
        self.n_item_2 = self.training_ratings_item_2.max()

        self.n_user_attr, self.n_item_attr_1, self.n_item_attr_2 = user_content.shape[1], item_content.shape[1], item_content.shape[1]

        # print('n_user=%d n_item=%d n_user_attr=%d n_item_attr=%d' %(self.n_user,self.n_item_1,self.n_user_attr, self.n_item_attr_1))

        self.user_attr = self.BuildAttributeFromSPMatrix(user_content,self.n_user,self.n_user_attr)

        # print("Shape 1:", item_content.shape)
        # print("Shape 2:", self.n_item_1)
        # print("Shape 3:", self.n_item_attr_1)
        
        self.item_attr = self.BuildAttributeFromSPMatrix(item_content,self.n_item_1+self.n_item_2,self.n_item_attr_1)
        self.item_attr_1 = self.item_attr[:len(self.item_attr)//2]
        self.item_attr_2 = self.item_attr[len(self.item_attr)//2:]
        
        # print("********************************")
        # print(self.item_attr_1)
        # print("********************************")
        # print(self.item_attr_2)
        # print("#################################")
#         self.item_attr_2 = self.BuildAttributeFromSPMatrix(item_content, self.n_item_2, self.n_item_attr_2)

    def BuildAttributeFromSPMatrix(self, sp_matrix, n, m):
        res = []
        for _ in range(int(n)):
            res.append([])
            
        (row,col,value) = find(sp_matrix)
        for r,c,v in zip(row,col,value):
            res[r].append([c,float(v)])
        return res



class sparse_data_repos:
    def __init__(self, n_user, n_item, n_user_attr = 0, n_item_attr = 0):
        self.n_user = n_user
        self.n_item = n_item
        self.n_user_attr = n_user_attr
        self.n_item_attr = n_item_attr
        self.user_attr = []
        self.item_attr = []
        self.training_ratings_user = []
        self.training_ratings_item = []
        self.training_ratings_item02 = []
        self.training_ratings_score = []
        self.test_ratings_user = []
        self.test_ratings_item = []
        self.test_ratings_item02 = []
        self.test_ratings_score = []
        self.eval_ratings_user=[]
        self.eval_ratings_item=[]
        self.eval_ratings_score=[]

    def load_user_attributes(self, infile,spliter='\t'):
        self.load_attributes(self.user_attr, self.n_user, self.n_user_attr, infile,spliter)

    def load_item_attributes(self, infile,spliter='\t'):
        self.load_attributes(self.item_attr, self.n_item, self.n_item_attr, infile,spliter)


    def load_attributes(self, res, n, m, infile,spliter):
        for i in range(n):
            res.append([])

        with open(infile, 'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break
                words = line.replace('\r\n','').replace('\n','').split(spliter)
                uid = int(words[0])
                for i in range(len(words)-1):
                    tokens = words[i+1].split(':')
                    res[uid].append([int(tokens[0]),float(tokens[1])])


    def load_trainging_ratings(self, infile, spliter = '\t'):
        self.load_rating_file(infile,self.training_ratings_user, self.training_ratings_item, self.training_ratings_score, spliter)

    def load_test_ratings(self, infile, spliter = '\t'):
        self.load_rating_file(infile,self.test_ratings_user, self.test_ratings_item, self.test_ratings_score, spliter)

    def load_eval_ratings(self, infile, spliter = '\t'):
        self.load_rating_file(infile,self.eval_ratings_user, self.eval_ratings_item, self.eval_ratings_score, spliter)

    def load_rating_file(self,infile,rating_user, rating_item, rating_score,spliter):
        del rating_user[:]
        del rating_item[:]
        del rating_score[:]

        with open(infile,'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break
                words = line.replace('\r\n','').replace('\n','').split(spliter)
                rating_user.append(int(words[0]))
                rating_item.append(int(words[1]))
                rating_score.append(float(words[2]))
        #print(rating_list)


    def load_trainging_pairwise_ratings(self, infile, spliter = '\t'):
        self.load_pairwise_rating_file(infile,self.training_ratings_user, self.training_ratings_item, self.training_ratings_item02, self.training_ratings_score, spliter)

    def load_test_pairwise_ratings(self, infile, spliter = '\t'):
        self.load_pairwise_rating_file(infile,self.test_ratings_user, self.test_ratings_item, self.test_ratings_item02, self.test_ratings_score, spliter)

    def load_pairwise_rating_file(self,infile,rating_user,rating_item01,rating_item02,rating_score, spliter):
        del rating_user[:]
        del rating_item01[:]
        del rating_item02[:]
        del rating_score[:]


        with open(infile,'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break
                words = line.replace('\r\n','').replace('\n','').split(spliter)
                rating_user.append(int(words[0]))
                rating_item01.append(int(words[1]))
                rating_item02.append(int(words[2]))
                rating_score.append(float(words[3]))



class dense_data_repos:
    def __init__(self, n_user, n_item, n_user_attr = 0, n_item_attr = 0):
        self.n_user = n_user
        self.n_item = n_item
        self.n_user_attr = n_user_attr
        self.n_item_attr = n_item_attr
        self.user_attr = []
        self.item_attr = []
        self.training_ratings = []
        self.test_ratings = []

    def load_user_attributes(self, infile,spliter='\t'):
        self.load_attributes(self.user_attr, self.n_user, self.n_user_attr, infile,spliter)

    def load_item_attributes(self, infile,spliter='\t'):
        self.load_attributes(self.item_attr, self.n_item, self.n_item_attr, infile,spliter)

    def load_attributes(self, res, n, m, infile,spliter):
        #res = [[0.0]*m for i in range(n)]
        for i in range(n):
            res.append([0.0]*m) 

        with open(infile, 'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break 
                words = line.replace('\r\n','').replace('\n','').split(spliter)
                uid = int(words[0])
                for i in range(len(words)-1):
                    tokens = words[i+1].split(':')
                    res[uid][int(tokens[0])] = float(tokens[1])

    def load_trainging_ratings(self, infile, spliter = '\t'):
        self.load_rating_file(infile,self.training_ratings,spliter)

    def load_test_ratings(self, infile, spliter = '\t'):
        self.load_rating_file(infile, self.test_ratings, spliter)

    def load_rating_file(self,infile,rating_list,spliter):
        del rating_list[:]
        with open(infile,'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break 
                words = line.replace('\r\n','').replace('\n','').split(spliter)
                rating_list.append([int(words[0]),int(words[1]),float(words[2])])
        #print(rating_list)


    def load_rating_tsv(filename):
    
        res = []
        with open(filename,'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break 
                words = line.replace('\r\n','').replace('\n','').split('\t')
                res.append([words[0],words[1],float(words[2])])
        return res

    def load_content_tsv(filename):
        '''
        res: dict --> uid : [ [tag,value], ...]
        '''
        res = {}
        with open(filename,'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break 
                words = line.replace('\r\n','').replace('\n','').split('\t')
                res[words[0]]=[]
                for i in range(len(words)-1):
                    tokens = words[i+1].split(':')
                    res[words[0]].append([tokens[0],float(tokens[1])])
        return res


if __name__ == '__main__':
    pass