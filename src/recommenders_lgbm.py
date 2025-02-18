import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

from src.recommenders import MainRecommender
from src.utils import prefilter_items
from src.preprocessor import DataPreprocessing, FeatureGenetator

class RecommenderLGBM:
    
    def __init__(self, data, item_features, user_features, 
                 val_lvl_1_size_weeks=6, val_lvl_2_size_weeks=3,
                 take_n_popular=5000): 
        
        self.data = data
        self.item_features = item_features
        self.user_features = user_features
        
        self.item_features.columns = [col.lower() for col in item_features.columns]
        self.user_features.columns = [col.lower() for col in user_features.columns]

        self.item_features.rename(columns={'product_id': 'item_id'}, inplace=True)
        self.user_features.rename(columns={'household_key': 'user_id'}, inplace=True)
        
        self.val_lvl_1_size_weeks = val_lvl_1_size_weeks
        self.val_lvl_2_size_weeks = val_lvl_2_size_weeks
        
        self.take_n_popular = take_n_popular
        
        self.data_train_lvl_1 =  self.data[self.data['week_no'] <  self.data['week_no'].max() - (self.val_lvl_1_size_weeks + self.val_lvl_2_size_weeks)]
        self.data_val_lvl_1 =  self.data[( self.data['week_no'] >=  self.data['week_no'].max() - (val_lvl_1_size_weeks + val_lvl_2_size_weeks)) &
                              (self.data['week_no'] <  self.data['week_no'].max() - self.val_lvl_2_size_weeks)]
        self.data_train_lvl_2 =  self.data_val_lvl_1.copy()
        self.data_val_lvl_2 =  self.data[self.data['week_no'] >=  self.data['week_no'].max() - self.val_lvl_2_size_weeks]

        self.data_train_lvl_1 = prefilter_items(
            self.data_train_lvl_1, 
            take_n_popular=self.take_n_popular, 
            item_features=self.item_features)
        
        self.recommender = MainRecommender(self.data_train_lvl_1)
        
        self.prepocessor = DataPreprocessing(self.data_train_lvl_1, self.recommender)
        self.features_gen = FeatureGenetator()
        
    def fit(self, num_leaves = 40, n_estimators = 20, max_depth = 5, learning_rate = 0.1,
            rec_method=MainRecommender.get_own_recommendations, N=20):
               
        self.features_gen.fit(self.data, self.item_features, self.user_features)
        
        X = self.prepocessor.transform(self.data_train_lvl_2, rec_method=rec_method, N=N)
        X = self.features_gen.transform(X)
        
        X_train = X.drop('target', axis=1)
        y_train = X[['target']]
        
        cat_feats = ['commodity_desc', 'sub_commodity_desc', 'curr_size_of_product']
        X_train[cat_feats] = X_train[cat_feats].astype('category')
        
        self.model = LGBMClassifier(
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='binary',
            categorical_column=cat_feats,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        self.users = X_train['user_id'].unique().tolist()
        self.X = X_train
        
    def recommend(self, user, N=5):
        
        if user not in self.users:
            return self.recommender.pop_items[:N]
        
        X = self.X[self.X['user_id']==user]
        X['preds'] = self.model.predict_proba(X)[:, 1]
        rec = X.sort_values(by='preds', ascending=False)[:N].item_id.tolist()
        
        return rec
              
