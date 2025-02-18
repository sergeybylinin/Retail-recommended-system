import pandas as pd
import numpy as np

from src.recommenders import MainRecommender


class DataPreprocessing:
    """Подготовка исходных данных"""

    def __init__(self, X1, recommender):
        
        self.recommender = recommender
        self.X1 = X1
        
    def transform(self, X2, rec_method=MainRecommender.get_own_recommendations, N=20):
        
        users_lvl_2 = pd.DataFrame(X2['user_id'].unique(), columns = ['user_id'])
        train_users = self.X1['user_id'].unique()
        users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(train_users)]
        users_lvl_2['candidates'] = users_lvl_2['user_id'].apply(lambda x: rec_method(self.recommender, x, N=N))
        s = users_lvl_2.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'item_id'
        users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)
        users_lvl_2['flag'] = 1
        X = X2[['user_id', 'item_id']].copy()
        X['target'] = 1
        X = users_lvl_2.merge(X, on=['user_id', 'item_id'], how='left')
        X = X[~X.duplicated()]
        X['target'].fillna(0, inplace=True)
        X.drop('flag', axis=1, inplace=True)
        
        return X
        
        
class FeatureGenetator:
    """Генерация новых фич"""
    
    def __init__(self):
    
        self.avg_chek = None
        self.user_dep = None
        self.dep_2 = None
        self.pf = None
        self.pw = None
        self.sp = None
        self.npw = None
        self.dep_3 = None
        self.aip = None
        self.dep_4 = None
        self.user_dep_2 = None
        self.brand = None
        self.department = None
        self.age_desc = None
        self.income_desc = None
        self.marital_status_code = None
        self.homeowner_desc = None
        self.hh_comp_desc = None
        self.household_size_desc = None
        self.kid_category_desc = None
            
    def fit(self, data, item_features, user_features):
        
        def coding_of_a_cahorial_feature(dt, col, on):
            df = data.join(dt[[on, col]].set_index(on), on=on)\
                   .groupby(col)['quantity', on]\
                   .agg({'quantity': np.sum, on: 'count'})
            df['quantity'] = df['quantity'] / df[on]
            return {i: round(j, 3) for i, j in dict(df['quantity'] / df['quantity'].min()).items()}
        
        data = data.copy()
        self.item_features = item_features.copy()
        self.user_features = user_features.copy()
        
        # Средний чек
        self.avg_chek = data.groupby('user_id')['sales_value', 'basket_id']\
            .agg({'sales_value': np.sum, 'basket_id': 'count'})
        self.avg_chek['avg_chek'] = self.avg_chek['sales_value'] / self.avg_chek['basket_id']
        self.avg_chek.drop(['sales_value', 'basket_id'], axis=1, inplace=True)
        
        # Средняя сумма покупки 1 товара в каждой категории для каждого user_а
        self.user_dep = data\
            .join(self.item_features[['item_id', 'department']].set_index('item_id'), on='item_id')\
            .groupby(['user_id', 'department'])['quantity', 'sales_value']\
            .agg({'quantity': np.sum, 'sales_value': np.sum})
        self.user_dep['avg_sum_purchase_one_item_id_by_department'] = self.user_dep['sales_value'] / self.user_dep['quantity']
        self.user_dep.drop(columns=['quantity', 'sales_value'], inplace=True)
        self.user_dep.dropna(axis=0, inplace=True)
        self.user_dep.reset_index('department', inplace=True)
        self.user_dep = self.user_dep.pivot_table(
            index='user_id', columns='department', 
            values='avg_sum_purchase_one_item_id_by_department', 
            aggfunc='sum', fill_value=0)
        self.user_dep.columns = [f'{col}_avg_cost_by_user' for col in self.user_dep.columns]
        
        # Кол-во покупок в каждой категории
        self.dep_2 = pd.DataFrame(data\
            .join(self.item_features[['item_id', 'department']].set_index('item_id'), on='item_id')\
            .groupby('department')['quantity'].sum())\
            .rename(columns={'quantity': 'n_of_purchases_in_the_category'})
        
        # Частотность покупок раз/месяц
        avg_count_of_days_in_a_month = 365 / 12
        count_months = round(data['day'].max() / avg_count_of_days_in_a_month)
        self.pf = pd.DataFrame(data.groupby('user_id')['quantity'].sum())\
            .rename(columns={'quantity': 'purchase_frequency'})
        self.pf['purchase_frequency'] = self.pf['purchase_frequency']\
            .apply(lambda x: round(x / count_months, 1))
        
        # Долю покупок в выходные дни
        data['days_of_the_week'] = data['day'].apply(lambda x: 'weekends' if x % 7 in [0, 6] else 'workday')
        self.pw = pd.DataFrame(data[data['days_of_the_week'] == 'weekends'].groupby('user_id')['sales_value'].sum()
            / data.groupby('user_id')['sales_value'].sum()).rename(columns={'sales_value': 'purchases in weekend'})

        # Доля покупок утром/днем/вечером

        data['date_time'] = pd.cut(
            data['trans_time'].apply(lambda x: int(f'000{x}'[-4: -2])),
            bins=[0, 6, 12, 18, 24], right=False, 
            labels=['night', 'morning', 'day', 'evening'])
        self.sp = data\
            .groupby(['user_id', 'date_time'])['sales_value'].sum().reset_index()\
            .pivot_table(index='user_id', columns='date_time', 
                         values='sales_value', aggfunc='sum', fill_value=0)
        self.sp.columns = [col for col in self.sp.columns]
        self.sp['sum_sales_value'] = self.sp[self.sp.columns].sum(axis=1)
        for col in self.sp.columns[:-1]: 
            self.sp[col] = round(self.sp[col] / self.sp['sum_sales_value'], 2)
        self.sp.drop('sum_sales_value', axis=1, inplace=True)
        
        # Кол-во покупок в неделю
        n_weeks = data['week_no'].max()
        self.npw = pd.DataFrame(data.groupby('item_id')['quantity'].sum() / n_weeks)\
            .rename(columns={'quantity': 'n_purchases_per_week'})
        
        # Среднее кол-во покупок 1 товара в категории в неделю
        self.dep_3 = pd.DataFrame(data\
            .join(self.item_features[['item_id', 'department']].set_index('item_id'), on='item_id')\
            .groupby('department')['quantity'].sum() / n_weeks)\
            .rename(columns={'quantity': 'avg_n_of_purchases_one_item_in_category_per_week'})
        
        # Средняя цена товара
        self.aip = data.groupby('item_id')[['quantity', 'sales_value']].sum()
        self.aip['avg_item_price'] = 0
        self.aip.loc[self.aip['quantity'] > 0, 'avg_item_price'] = self.aip['sales_value'] / self.aip['quantity']
        self.aip.drop(columns=['quantity', 'sales_value'], inplace=True)
        
        # Средняя цена товара в категории
        self.dep_4 = item_features\
            .join(self.aip, on='item_id', how='left')\
            .groupby('department')['item_id', 'avg_item_price']\
            .agg({'item_id': 'count', 'avg_item_price': np.sum})
        self.dep_4['avg_price_itmem_in_category'] = self.dep_4['avg_item_price'] / self.dep_4['item_id']
        self.dep_4.drop(columns=['avg_item_price', 'item_id'], inplace=True)
        
        # Кол-во покупок юзером в каждой категории
        self.user_dep_2 = data\
            .join(self.item_features[['item_id', 'department']].set_index('item_id'), on='item_id')\
            .groupby(['user_id', 'department'])['quantity'].sum().reset_index()\
            .pivot_table(index='user_id', columns='department', 
                         values='quantity', aggfunc='sum', fill_value=0)
        self.user_dep_2.columns = [f'{col}_count_purchases' for col in self.user_dep_2.columns]
        
        # brand
        self.brand = self.item_features['brand'].unique().tolist()
        self.brand = {i: self.brand.index(i) for i in self.brand}
        
        # department
        dep = data\
            .join(self.item_features[['item_id', 'department']].set_index('item_id'), on='item_id')\
            .groupby('department')['quantity', 'sales_value', 'item_id']\
            .agg({'quantity': np.sum, 'sales_value': np.sum, 'item_id': 'count'})
        dep['quantity'] = dep['quantity'] / dep['item_id']
        dep['sales_value'] = dep['sales_value'] / dep['item_id']
        dep = dep[['quantity', 'sales_value']]
        for col in dep: 
            dep[col] = dep[col] / dep.iloc[1:, :][col].min()
        self.department = {ind: round(dep['quantity'][ind] * dep['sales_value'][ind], 3) for ind in dep.index}
                
        # age_desc
        self.age_desc = self.user_features['age_desc'].sort_values().unique().tolist()
        self.age_desc = {i: self.age_desc.index(i) + 1 for i in self.age_desc}
        
        # income_desc
        self.income_desc = self.user_features['income_desc'].unique().tolist()
        sort_dict = {i: int(''.join([x for x in i.replace('+', '000') if x.isdigit()])) for i in self.income_desc}
        self.income_desc = sorted(self.income_desc, key=lambda x: sort_dict[x])
        self.income_desc = {i: self.income_desc.index(i) + 1 for i in self.income_desc}
        
        # marital_status_code
        self.marital_status_code = coding_of_a_cahorial_feature(
            self.user_features, 'marital_status_code', 'user_id')

        # homeowner_desc
        self.homeowner_desc = coding_of_a_cahorial_feature(
            self.user_features, 'homeowner_desc', 'user_id')

        # hh_comp_desc
        self.hh_comp_desc = coding_of_a_cahorial_feature(
            self.user_features, 'hh_comp_desc', 'user_id')
        
        # household_size_desc
        self.household_size_desc = self.user_features['household_size_desc'].sort_values().unique().tolist()
        self.household_size_desc = {i: self.household_size_desc.index(i) + 1 for i in self.household_size_desc}
        
        # kid_category_desc
        self.kid_category_desc = self.user_features['kid_category_desc'].unique().tolist()
        self.kid_category_desc = {i: self.kid_category_desc.index(i) for i in self.kid_category_desc}
        
    # update
        
        # brand
        self.item_features['brand'] = self.item_features['brand'].replace(self.brand)

        # department
        self.item_features['department_1'] = self.item_features['department'].replace(self.dep_2['n_of_purchases_in_the_category'])
        self.item_features['department_2'] = self.item_features['department'].replace(self.dep_3['avg_n_of_purchases_one_item_in_category_per_week'])
        self.item_features['department_3'] = self.item_features['department'].replace(self.dep_4['avg_price_itmem_in_category'])
        self.item_features['department'] = self.item_features['department'].replace(self.department)
        
        # age_desc
        self.user_features['age_desc'] = self.user_features['age_desc'].replace(self.age_desc)
        
        # income_desc
        self.user_features['income_desc'] = self.user_features['income_desc'].replace(self.income_desc)
        
        # marital_status_code
        self.user_features['marital_status_code'] = self.user_features['marital_status_code'].replace(self.marital_status_code)
        
        # homeowner_desc
        self.user_features['homeowner_desc'] = self.user_features['homeowner_desc'].replace(self.homeowner_desc)
        
        # hh_comp_desc
        self.user_features['hh_comp_desc'] = self.user_features['hh_comp_desc'].replace(self.hh_comp_desc)
        
        # household_size_desc
        self.user_features['household_size_desc'] = self.user_features['household_size_desc'].replace(self.household_size_desc)
        
        # kid_category_desc
        self.user_features['kid_category_desc'] = self.user_features['kid_category_desc'].replace(self.kid_category_desc)

    def transform(self, X):
        
        X = X.merge(self.avg_chek, on='user_id', how='left')
        X = X.merge(self.user_dep, on='user_id', how='left')
        X = X.merge(self.user_dep_2, on='user_id', how='left')
        X = X.merge(self.pf, on='user_id', how='left')
        X = X.merge(self.pw, on='user_id', how='left')
        X = X.merge(self.sp, on='user_id', how='left')

        X = X.merge(self.npw, on='item_id', how='left')
        X = X.merge(self.aip, on='item_id', how='left')

        X = X.merge(self.item_features, on='item_id', how='left')
        X = X.merge(self.user_features, on='user_id', how='left')
        
        return X
