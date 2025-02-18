import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=False):
            
        self.data = data
        self.user_item_matrix = self.prepare_matrix(self.data)  # pd.DataFrame
        self.sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = \
            self.prepare_dicts(self.user_item_matrix)
        self.pop_items = self.data.loc[self.data['item_id'] != 999999]\
            .groupby('item_id')['quantity'].count().reset_index()\
            .sort_values('quantity', ascending=False)['item_id'].tolist()
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        
        
    @staticmethod
    def prepare_matrix(data):
        user_item_matrix = pd.pivot_table(
            data, 
            index='user_id', columns='item_id', 
            values=['basket_id', 'quantity'],
            aggfunc={
                'basket_id': 'count',
                'quantity': 'count'
            },
            fill_value=0
        )

        user_item_matrix = user_item_matrix.basket_id / user_item_matrix.quantity
        user_item_matrix.fillna(0, inplace=True)
                
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=49, regularization=0.0017, iterations=20, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)
        
        return model
    
    def append_pop_items(self, recs, N=5):
        if len(recs) < N:
            recs = (recs + [item for item in self.pop_items[:N] if item not in recs])[:N]
        return recs
    
    def _get_recommendations(self, user, model, N=5):
                
        if user not in self.userid_to_id:
            return self.pop_items[:N]
        
        recs = [self.id_to_itemid[rec[0]] for rec in model.recommend(
            userid=self.userid_to_id[user], 
            user_items=self.sparse_user_item,
            N=N,                        
            filter_already_liked_items=False, 
            filter_items=[self.itemid_to_id[999999]], 
            recalculate_user=True
        )]
        recs = self.append_pop_items(recs, N=N)
        
        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        
        return recs
    
    def get_als_recommendations(self, user, N=5):
        
        return self._get_recommendations(user, model=self.model, N=N)
    
    def get_own_recommendations(self, user, N=5):

        return self._get_recommendations(user, model=self.own_recommender, N=N)
    
    def get_similar_items_recommendation(self, user, N=5):
        
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        if user not in self.userid_to_id:
            return self.pop_items[:N]
        
        popularity_items = self.data\
            .loc[(self.data['user_id'] == user) & 
                 (self.data['item_id'] != 999999)]\
            .groupby(['item_id'])['quantity'].count().reset_index()\
            .sort_values('quantity', ascending=False).head(N)\
            .item_id.tolist()
    
        recs = []
        for item in popularity_items:
            similar_items = self.model.similar_items(self.itemid_to_id[item], N=3)
            similar_item_id = similar_items[1][0]
            rec = self.id_to_itemid[similar_item_id]
            if rec == 999999:
                rec = self.id_to_itemid[similar_items[2][0]]
            recs.append(rec)
        recs = self.append_pop_items(recs, N=N)
        
        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        
        return recs
    
    def get_similar_users_recommendation(self, user, N=5):
    
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        if user not in self.userid_to_id:
            return self.pop_items[:N]
        
        similar_users = map(lambda x: self.id_to_userid[x], 
                            map(lambda x: x[0], 
                                self.model.similar_users(
                                    self.userid_to_id[user], 
                                    N=N+1)[1:]))

        recs = []
        for n, user in enumerate(similar_users, 1):
            for rec in self.get_own_recommendations(user, N=n):
                if rec not in recs:
                    recs.append(rec)
                    break
        recs = self.append_pop_items(recs, N=N)
        
        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        
        return recs
