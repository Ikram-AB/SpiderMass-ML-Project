import pandas as pd
import numpy as np
from pyopenms import *
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from collections import Counter

def apply_smote(df):
    smote = SMOTE(k_neighbors=2)
    
    columns_to_drop = ['Class', 'File', 'RT', 'Sum']
    
    X = df.drop(columns_to_drop, axis=1)
    y = df['Class']
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("Distribution des classes après suréchantillonnage avec SMOTE:", Counter(y_resampled))
    
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)
    
    return df_resampled

def apply_adasyn(df):
    adasyn = ADASYN(k_neighbors=2)
    
    columns_to_drop = ['Class', 'File', 'RT', 'Sum']
    
    X = df.drop(columns_to_drop, axis=1)
    y = df['Class']
    
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    print("Distribution des classes après suréchantillonnage avec ADASYN:", Counter(y_resampled))
    
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)
    
    return df_resampled

def apply_pca(data, n_components):
    X = data.iloc[:, 4:].values  
    y = data['Class']  
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])

    final_df = pd.concat([principal_df, y], axis=1)
    return final_df


def apply_svd(data, n_components):
    X = data.iloc[:, 4:].values  
    y = data['Class'] 
    
    svd = TruncatedSVD(n_components=n_components)
    principal_components = svd.fit_transform(X)

    principal_df = pd.DataFrame(data=principal_components, columns=[f'Component_{i}' for i in range(1, n_components + 1)])

    final_df = pd.concat([principal_df, y], axis=1)
    return final_df