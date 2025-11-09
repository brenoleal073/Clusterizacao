import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

url = "D:\001 - POS - IA GENERATIVA\Cluterização\PD-Clusterização\Country-data.csv"
try:
    df = pd.read_csv('Country-data.csv')
except FileNotFoundError:
    print("⚠️ Arquivo 'Country-data.csv' não encontrado. Certifique-se de baixá-lo do Kaggle e colocá-lo no diretório correto.")
    df = pd.read_csv('Country-data.csv')

print("✅ Dados carregados com sucesso.")
print(f"Total de Países (linhas) no Dataset: **{df.shape[0]}**")
print("\nPrimeiras 5 linhas do dataset:")
display(df.head())
print("\nInformações sobre as colunas e tipos de dados:")
df.info()