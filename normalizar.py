# avaliacao_cluster_final.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# =======================================================
# PARTE 1: PREPARAÇÃO E NORMALIZAÇÃO DOS DADOS
# =======================================================

print("--- 1. Carregando e Normalizando Dados ---")
# Carregar o dataset.
try:
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
except FileNotFoundError:
    print("ERRO: Certifique-se de que 'heart_failure_clinical_records_dataset.csv' está na pasta.")
    exit()

# Definir as colunas de características (features) para clusterização.
# Excluímos 'DEATH_EVENT' (target) e 'time' (tempo de acompanhamento).
features_cols = df.columns.drop(['DEATH_EVENT', 'time'])
X = df[features_cols]

# Inicializar e treinar o normalizador (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Salvar o scaler para uso na Etapa 4 (novo paciente)
joblib.dump(scaler, 'scaler_heart_failure.pkl')
print(f"Features utilizadas: {features_cols.tolist()}")
print("Dados normalizados com sucesso.")


# =======================================================
# PARTE 2: MÉTODO DO COTOVELO (ELBOW) PARA K ÓTIMO
# =======================================================

print("\n--- 2. Determinando o Número Ótimo de Clusters (K) ---")
wcss = []
k_range = range(1, 11)  # Testar de 1 a 10 clusters

for i in k_range:
    # random_state fixo para resultados reproduzíveis
    # n_init=10 para garantir reprodutibilidade (necessário no scikit-learn >= 1.2)
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ é o WCSS

# Plotar o Método do Cotovelo
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Método do Cotovelo (WCSS vs. K)')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Baseado na observação do gráfico, você deve escolher o K ótimo.
# (Em muitos datasets, K=3 ou K=4 é um bom ponto de partida)
K_OPTIMO = 3  # ALTERE ESTE VALOR após analisar o gráfico

print(f"K-Means executado de K=1 a K=10. Analise o gráfico para definir o K ÓTIMO.")
print(f"K_OPTIMO escolhido (ajuste se necessário): {K_OPTIMO}")

# =======================================================
# PARTE 3: TREINAMENTO DO MODELO FINAL E DESCRIÇÃO
# =======================================================

print("\n--- 3. Treinando o Modelo Final e Descrevendo Clusters ---")

# Treinar o modelo final com o K ótimo
kmeans_final = KMeans(n_clusters=K_OPTIMO, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Salvar o modelo para uso na Etapa 4
joblib.dump(kmeans_final, 'kmeans_model_heart_failure.pkl')

# Adicionar os rótulos de cluster ao DataFrame original (NÃO normalizado)
df_clustered = X.copy()
df_clustered['Cluster'] = cluster_labels

# Descrição dos Clusters: Centróides na escala original
cluster_analysis = df_clustered.groupby('Cluster').mean()
cluster_sizes = df_clustered['Cluster'].value_counts().sort_index().rename('N_Pacientes')
summary = pd.concat([cluster_sizes, cluster_analysis], axis=1)

# ⭐️ LINHA ADICIONADA PARA SALVAR A TABELA DE CENTRÓIDES EM CSV ⭐️
summary.to_csv('tabela_de_centroides_analise_final.csv', index=True, float_format="%.2f", sep=';')
print("\nArquivo 'tabela_de_centroides_analise_final.csv' SALVO com sucesso.")
# ⭐️ FIM DA LINHA ADICIONADA ⭐️

print("\n==================================================================")
print("DESCRIÇÃO DOS CLUSTERS (Centróides - Média das Características)")
print("==================================================================")
# Usamos to_markdown para continuar exibindo no terminal
print(summary.to_markdown(index=True, floatfmt=".2f"))


# =======================================================
# PARTE 4: DETERMINAR CLUSTER DE NOVO PACIENTE
# =======================================================

print("\n--- 4. Classificação de um Novo Paciente ---")

# Novo Paciente (ATENÇÃO: Deve ter as mesmas 11 colunas na ordem correta!)
# Exemplo: Paciente de 65 anos, EF baixa (30), Creatinina alta (1.8) -> ALTO RISCO ESPERADO
novo_paciente_data = {
    'age': 65,
    'anaemia': 0,
    'creatinine_phosphokinase': 150,
    'diabetes': 1,
    'ejection_fraction': 30, # Fator de risco
    'high_blood_pressure': 1,
    'platelets': 250000,
    'serum_creatinine': 1.8, # Fator de risco
    'serum_sodium': 134,
    'sex': 1,
    'smoking': 0
}

# Criar DataFrame (com a ordem de colunas garantida)
novo_paciente_df = pd.DataFrame([novo_paciente_data], columns=features_cols)

# Carregar o scaler e o modelo salvos
scaler_loaded = joblib.load('scaler_heart_failure.pkl')
kmeans_loaded = joblib.load('kmeans_model_heart_failure.pkl')

# 1. Normalizar o novo paciente (USANDO APENAS TRANSFORM!)
novo_paciente_scaled = scaler_loaded.transform(novo_paciente_df)

# 2. Prever o Cluster
cluster_previsto = kmeans_loaded.predict(novo_paciente_scaled)

print("\n==================================================")
print(f"O NOVO PACIENTE pertence ao Cluster: **{cluster_previsto[0]}**")
print("==================================================")
print(f"Para interpretar o risco (Alto/Baixo), consulte a tabela de centróides acima.")