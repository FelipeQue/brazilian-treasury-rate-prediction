# 🏛️ TreasuryCaster: Modelo Preditivo para Taxas de Leilões do Tesouro Nacional
Modelo preditivo para taxas de leilões das NTN-Bs do Tesouro Nacional, utilizando variáveis macroeconômicas e árvores de decisão. Projeto desenvolvido para o curso *Desenvolvimento de IA para Análise Preditiva* do programa SCTEC.

## 📊 Os Dados

* **📈 Taxas de leilões do Tesouro Nacional:** [Dataset disponível no Kaggle](https://www.kaggle.com/datasets/kayoricardo/leiles-da-dvida-pblica-federal-brasil/data).
* **🏦 Taxas de juros Selic:** Obtidas via API do Banco Central do Brasil.
* **💵 Taxas de câmbio do dólar:** Obtidas via API do Banco Central do Brasil.

## 🛠️ Tecnologias e bibliotecas utilizadas
- Python 3.14.3
- Jupyter Notebook 7.5.7
- Pandas 3.0.3
- Numpy 2.5.1
- Matplotlib 3.11.0
- Seaborn 0.13.2
- Scikit-Learn 1.9.0
- Statsmodels 0.14.6
- XGBoost 3.3.0

- Durante o desenvolvimento deste projeto foi utilizado um ambiente virtual (venv) para gerenciar as dependências do projeto, garantindo que as bibliotecas necessárias estejam isoladas.

## 🚀 Como executar o projeto
1. Clone este repositório em sua máquina local.
2. Crie um ambiente virtual e ative-o.
3. Instale as dependências do projeto utilizando o arquivo requirements.txt.
4. Execute o notebook treasurycaster.ipynb para treinar e avaliar o modelo preditivo.

## 🗺️ Roteiro do notebook
1. 🔍 **Fase 1: Coleta e análise inicial dos dados** - Nesta fase, os dados de taxas de leilões do Tesouro Nacional, taxas de juros Selic e taxas de câmbio do dólar são coletados e agregados. São feitas então análises exploratórias para entender a distribuição e correlação das variáveis.
2. 🧼 **Fase 2: Limpeza dos dados** - Nesta fase, os dados são limpos, tratando valores ausentes e outliers, e transformados para o formato adequado para o treinamento do modelo.
3. 📐 **Fase 3: Engenharia de variáveis** - Nesta fase, são criadas 3 novas variáveis a partir das existentes:
- DURACAO: representa a duração em dias entre a data do leilão e a data de vencimento do título (_days to maturity_).
- REJEICAO_MERCADO: sinaliza o comportamento de rejeição do mercado. É uma variável binária que assume o valor 1 se a proporção ACEITO/OFERTADO for menor que 15%.
- TAXA_ULTIMO_LEILAO: representa a taxa do leilão anterior do título. 
4. 🧪 **Fase 4: Preparação dos dados para o treinamento do modelo** - Nesta fase, os dados são divididos em conjuntos de treino e teste e é feita a análise de multicolinearidade a partir do cálculo do VIF (_Variance Inflation Factor_). Além disso, é feita a padronização das variáveis numéricas. Também nesta etapa foram descartadas todas as colunas que não seriam utilizadas no treinamento do modelo.
5. 🤖 **Fase 5: Treinamento e avaliação dos modelos** - Nesta fase, é feito o treinamento de diferentes modelos sobre os dados preparados. O presente projeto realizou treinos e testes com modelos de:
- Regressão linear
- Ridge (modelo de regressão linear com regularização)
- Lasso (modelo de regressão linear com regularização)
- XGBoost (um modelo de árvore de decisão sequencial)

Cada modelo é avaliado com base nas métricas R2, MAE, MSE e RMSE. Foram treinados e testados modelos com o conjunto completo de variáveis explicativas e também com o conjunto sem a variável de lag TAXA_ULTIMO_LEILAO, para avaliar o impacto desta variável no desempenho do modelo. Também foi feita uma breve discussão sobre o comportamento dos dados levando em conta o período histórico.
6. 💾 **Fase 6: Seleção e versionamento** - Nesta fase, o modelo de melhor desempenho (XGBoost) é versionado e salvo, assim como suas métricas obtidas na fase de treino/teste.

## 📦 Modelo entregue: v1.0

- Versão: v1.0
- Tipo: XGBoost
- Modelo e métricas disponíveis no diretório: models/v1

## 🔮 Possíveis melhorias futuras
- Incluir outras variáveis macroeconômicas, como a direção de tendência da Selic conforme as últimas reuniões do Copom (aumento, diminuição ou manutenção), ou o DI futuro, para tentar ajudar o modelo a antecipar mudanças de regime e reduzir mais o erro.
- Treinar o modelo em outras janelas temporais e regimes econômicos.