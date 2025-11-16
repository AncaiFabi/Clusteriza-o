# 🧬 Projeto de Clusterização K-Means em Pacientes com Insuficiência Cardíaca

Este projeto utiliza o algoritmo K-Means para identificar e descrever perfis de risco de pacientes com Insuficiência Cardíaca, a partir de dados clínicos. O objetivo é realizar a segmentação da base de dados sem supervisão, validando os perfis de risco encontrados. Projeto realizado pelas alunas Fabiane Ançai, Naara de Oliveira e Sarah Veloso.

---

## 🎯 Objetivo da Clusterização

O principal objetivo deste trabalho é aplicar técnicas de **Aprendizado Não Supervisionado** para:

1.  **Segmentação:** Agrupar pacientes com características clínicas semelhantes.
2.  **Descoberta de Perfis:** Descrever os centróides de cada cluster (média das características) para identificar perfis de risco (ex: "Perfil Fragilidade Renal" vs. "Perfil de Alto Risco Cardíaco").
3.  **Classificação:** Classificar um paciente novo/desconhecido em um dos perfis identificados.

---

## 💾 Dados Utilizados

* **Fonte:** Heart Failure Clinical Records Dataset
* **Arquivo:** `heart_failure_clinical_records_dataset.csv`
* **Descrição:** O dataset contém dados clínicos e laboratoriais de 299 pacientes, coletados durante o acompanhamento de 28 a 244 dias.

### Características Relevantes para a Clusterização

| Variável | Tipo | Notas |
| :--- | :--- | :--- |
| `age` | Numérica | Idade do paciente. |
| `ejection_fraction` | Numérica | Porcentagem de sangue que sai do coração a cada batimento (risco se baixo). |
| `serum_creatinine` | Numérica | Nível de creatinina no sangue (indicador de função renal). |
| `anaemia`, `diabetes`, etc. | Binária (0/1) | Comorbidades e indicadores de estilo de vida. |
| **Excluídas:** | - | `DEATH_EVENT` (target) e `time` (tempo de acompanhamento). |

---

## 🛠️ Metodologia e Pipeline

O projeto foi desenvolvido em Python e segue o seguinte pipeline de Machine Learning:

### 1. Pré-processamento e Normalização
* **Seleção de Features:** Todas as 11 características clínicas foram mantidas, excluindo o target (`DEATH_EVENT`) e o tempo (`time`).
* **Normalização:** Utilizado o **StandardScaler** para padronizar os dados, garantindo que variáveis com escalas muito diferentes (como Plaquetas e Creatinina) tivessem o mesmo peso na métrica de distância do K-Means.

### 2. Determinação do K Ótimo
* **Técnica:** **Método do Cotovelo (Elbow Method)**.
* **Justificativa:** A Curva WCSS (Soma dos Quadrados Dentro do Cluster) foi analisada para determinar o ponto de inflexão que minimiza a distorção.
* **Resultado:** O número ótimo de clusters (K) definido foi: **K = 3**.

### 3. Treinamento e Análise de Centróides
* **Modelo:** K-Means treinado com o K ótimo.
* **Análise:** Os **Centróides** (médias das características) foram calculados na **escala original** dos dados (não normalizada) para facilitar a **interpretação clínica**.

---

## 📊 Resultados e Perfis Identificados

Baseado na análise dos centróides (gerada no arquivo `tabela_de_centroides_analise_final.csv`), os seguintes perfis foram identificados:

| Cluster | N de Pacientes | Perfil Chave | Fatores de Risco Distintivos |
| :---: | :---: | :--- | :--- |
| **[0]** | [95] | **[Perfil do Fumante com Função Cardíaca Comprometida]** | Menor Fração de Ejeção média (36.92%), 100% fumantes. |
| **[1]** | [108] | **[Perfil de Baixo Risco/Saudável]** | Mais jovens, ausência de anemia e tabagismo. |
| **[2]** | [96] | **[Perfil de Maior Fragilidade e Disfunção Renal]** | Mais velhos (61.81), 100% anêmicos, maior Creatinina Sérica (1.45). |

---

## 💻 Como Executar o Projeto

Para replicar esta análise, você precisará ter o Python instalado (versão 3.8+).

### 1. Criar e Ativar Ambiente Virtual
```bash
python -m venv .venv
.venv\Scripts\activate  # Para Windows (PowerShell/CMD)
# source .venv/bin/activate # Para Linux/MacOS
