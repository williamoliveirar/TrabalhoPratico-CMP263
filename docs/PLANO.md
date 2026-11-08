# Trabalho Final CMP263 — Detecção binária de ataques em redes IoT com o CICIoT2023

**Documento de contexto e plano de execução** · versão 1.1 · 05/11/2026
Disciplina: CMP263 – Aprendizagem de Máquina (PPGC/UFRGS), Profa. Mariana Recamonde Mendoza · Trabalho em dupla
Entrega do artigo e do código: **22/11/2026, 23h59** (meta interna: **20/11/2026**)

> **Versão 1.1 (05/11/2026):** a terminologia foi alinhada à das aulas, e as seções 1.3 e 1.4 foram acrescentadas: o mapa das etapas formais do treinamento de modelos e o glossário. A Etapa 4 foi reescrita a partir da D13. As etapas e decisões posteriores agora citam os nomes das colunas da versão usada (D15). O estado vigente de cada decisão está no quadro "Decisões vigentes" de `docs/DIARIO.md`. Fontes da terminologia: os slides da disciplina, sobretudo "Avaliação de modelos — Divisão de dados" (Aula 08), "Método holdout" (Aula 04), "Métricas de desempenho" (Aula 09), "Nested Cross-Validation & Data Leakage" (Aula 04, material extra), "Limpeza e transformação" e "Desbalanceamento de classes" (Aulas 13–14) e "Redução de dimensionalidade" (Aulas 15–16).

---

## 0. Como usar este documento

### 0.1 Para a dupla

Este arquivo reúne o que é preciso para executar o trabalho do início ao fim: o que o enunciado exige, o problema escolhido, o dataset, as decisões metodológicas, a estrutura do repositório, as etapas com critérios de conclusão, o cronograma e a estrutura do artigo. Ele é o **plano**. O registro do que de fato aconteceu fica no **diário** (seção 6), que é a matéria-prima do artigo.

Para usar com o Claude Code:

1. Salve este arquivo como `docs/PLANO.md` dentro do repositório.
2. Crie na raiz do repositório um `CLAUDE.md` curto, como o modelo da seção 0.2. O Claude Code carrega o `CLAUDE.md` da raiz no início de cada sessão, e o conteúdo dele ocupa contexto em todas as sessões. Por isso ele deve ser curto e apenas apontar para este plano. Não use a sintaxe de importação `@docs/PLANO.md` dentro do `CLAUDE.md`, porque isso carregaria o plano inteiro em toda sessão.
3. Copie o PDF do enunciado para `docs/enunciado.pdf`.
4. Mantenha este plano, o diário e o registro de IA no controle de versão.

### 0.2 Modelo de `CLAUDE.md` (raiz do repositório)

```markdown
# CLAUDE.md — Trabalho final CMP263 (UFRGS): detecção binária de ataques IoT no CICIoT2023

Contexto e plano completos: docs/PLANO.md. Antes de começar, leia a seção da etapa atual
e as últimas entradas de docs/DIARIO.md.

## Regras permanentes
- Converse e escreva o diário em português (pt-BR). Código, comentários, figuras e artigo em inglês.
- Ao concluir uma etapa, tomar uma decisão, encontrar algo inesperado ou encerrar uma sessão,
  acrescente uma entrada em docs/DIARIO.md (modelo na seção 6 do plano). Nunca edite entradas antigas.
- Registre em docs/IA_LOG.md os pedidos relevantes feitos à IA na sessão (exigência do enunciado).
- Os folds externos de teste nunca informam decisões (pré-processamento, atributos, hiperparâmetros, limiar).
- Sementes, caminhos e tamanhos vêm de configs/*.yaml. Nada de números mágicos no código.
- Todo número citado no diário ou no artigo aponta para o arquivo de resultado que o gerou.
- Tabelas e figuras do artigo são geradas por script (src/reporting), nunca copiadas à mão.
- Itens [verificar] do plano só viram fato depois de confirmados na fonte primária; registre a confirmação no diário.
- Pergunte antes de apagar dados brutos ou sobrescrever resultados já registrados.
- Seja crítico: aponte problemas metodológicos sem rodeios; não concorde só por concordar.
```

### 0.3 Para o Claude Code

- Na primeira sessão, leia este plano inteiro. Nas seguintes, leia a seção da etapa em andamento e as últimas entradas do diário.
- Cada etapa da seção 7 tem objetivo, tarefas, saídas e critério de pronto. Não avance para a etapa seguinte sem cumprir o critério de pronto e registrar a entrada correspondente no diário.
- Quando os dados reais contradisserem algo deste plano, os dados vencem. Registre a divergência no diário como `[DESCOBERTA]` e proponha o ajuste à dupla antes de seguir.
- Convenção: **[verificar]** marca informação que veio de memória e ainda não foi confirmada na fonte primária (artigo do dataset, página oficial ou os próprios CSVs).

---

## 1. Contexto

### 1.1 O que o enunciado exige

**Tema:** *Desenvolvimento e Avaliação de Modelos Preditivos em Aprendizado de Máquina: Metodologia, Reprodutibilidade e Interpretabilidade.*

**Objetivo:** selecionar um problema preditivo (classificação ou regressão) e desenvolver uma metodologia de AM para resolvê-lo, da coleta e pré-processamento dos dados ao treinamento, análise e interpretação dos modelos, com ênfase em boas práticas de avaliação e em reprodutibilidade.

**Entregáveis e pesos (nota total 10):**

| Entregável | Peso | Observações |
|---|---|---|
| Artigo científico | 5,5 (4,0 da professora + 1,5 da revisão pelos colegas) | Template do Moodle, coluna única, até 20 páginas incluindo referências, preferencialmente em inglês |
| Código / pipeline | 3,0 | Arquivos com instruções de execução ou link do Colab. Avaliados quanto a corretude metodológica, coerência e robustez em todas as etapas |
| Apresentação oral | 1,5 | Formato de apresentação científica. Deve discutir decisões de reprodutibilidade. A participação nas discussões dos outros grupos é avaliada individualmente |

**Prazos:**

| Item | Data |
|---|---|
| Código + artigo (PDF no Moodle) | 22/11/2026, 23h59 |
| Revisão por pares (cada aluno revisa 2 artigos; nota individual, peso 30% em AT) | 24/11 a 30/11/2026, 23h59 |
| Apresentação oral | 01/12/2026 (tempo máximo a ser informado pela professora) |

Atrasos custam **1 ponto por dia**.

**Requisitos metodológicos** (servem de checklist para todo o plano):

- Definição clara do problema, da pergunta de pesquisa e dos dados, com link para download do dataset público.
- Análise exploratória (EDA) com gráficos e sumarização estatística.
- Pré-processamento conforme o que a EDA revelou, com justificativa (ou justificativa para não fazer).
- Pelo menos 5 algoritmos supervisionados, com vieses indutivos diversos, incluindo algoritmos do programa da disciplina.
- Mais de uma métrica, com uma **métrica principal** de seleção de modelos.
- Estratégia explícita de divisão dos dados para treino, validação (ajuste de hiperparâmetros) e avaliação final.
- Múltiplas repetições, mesmas divisões para todos os algoritmos em cada iteração, ausência de data leakage e sementes fixas.
- Spot-checking de algoritmos antes da otimização de hiperparâmetros (sugerido).
- Resultados sumarizados com gráficos, tabelas e análise estatística, com conclusões apoiadas em dados.
- Interpretação de **um** modelo (o melhor), discutindo relações pertinentes ou inesperadas.
- Reprodutibilidade: disponibilizar código, dados e modelos quando possível, além de relato metodológico detalhado.
- Pouco espaço para teoria básica dos algoritmos; referencial teórico sucinto.

**Critérios de avaliação:** (i) qualidade técnica; (ii) originalidade da pergunta de pesquisa; (iii) comunicação, com equilíbrio entre clareza e abrangência e boas práticas de reprodutibilidade.

**Uso de IA generativa:** permitido como apoio, não como substituto do esforço humano. Deve ser **citado no artigo**, e o grupo deve preparar um **material suplementar com os prompts usados** e as limitações ou potencialidades observadas. Seguir as diretrizes de uso e citação de IA disponíveis no Moodle. Isso é coberto pelo `docs/IA_LOG.md` (seção 6.3).

**Grupo:** dupla. Os dois integrantes devem contribuir em todas as etapas, inclusive escrita e apresentação.

### 1.2 Relação com a dissertação (apenas contexto)

Um dos integrantes desenvolve a dissertação *Detecção e Reação Inteligente a Ciberataques em Redes IoT Residenciais* (título provisório), que propõe detecção por TinyML em microcontroladores ESP32, reação graduada entre dispositivos e autenticação com criptografia pós-quântica. Este trabalho reaproveita o dataset e o subproblema de detecção, mas é **autocontido** e segue só o enunciado da disciplina.

**Fora do escopo deste trabalho:** classificação em 8 ou 34 classes; restrição aos atributos calculáveis no ESP32; tamanho de modelo, quantização e tempo de inferência em microcontrolador; autoencoder e detecção de anomalia; criptografia pós-quântica, hardware, reação coordenada e LLM; testes de generalização em outros datasets.

Se alguma ideia desse tipo surgir durante o trabalho, registre-a no diário com a tag `[IDEIA-DISSERTAÇÃO]` e não a execute.

### 1.3 As etapas formais do treinamento de modelos e onde estão no plano

O enunciado define quatro passos: (i) análise exploratória; (ii) pré-processamento; (iii) treinamento e validação, com divisão dos dados, otimização de hiperparâmetros, seleção de modelos e spot-checking; (iv) interpretação. A eles se somam a definição do problema, a coleta dos dados e a reprodutibilidade. Nas aulas, a ordem de cada passo é dada por uma regra: **quando transformar os dados** (Aulas 13–14).
- Operações que **removem instâncias ou atributos**, ou que aplicam uma regra fixa sem parâmetro estimado dos dados, podem ser feitas no conjunto completo, **antes da divisão**.
- Operações que **estimam parâmetros a partir dos dados**, como imputação, normalização, padronização, discretização, seleção de atributos e reamostragem (*oversampling*), são ajustadas **só no conjunto de treinamento** e depois aplicadas, sem novo ajuste, aos conjuntos de validação e de teste. Na validação cruzada, isso é refeito a cada iteração, dentro de um `Pipeline`.

Seguindo essa regra, o fluxo do projeto é:

| # | Etapa formal (enunciado e aulas) | O que inclui neste trabalho | Etapa do plano | Dados usados |
|---|---|---|---|---|
| 1 | Definição do problema e coleta dos dados | Tarefa, perguntas de pesquisa, dataset, ingestão fiel dos CSVs | §2; Etapas 0 e 1 | Dataset completo (só contagens e esquema) |
| 2 | Amostragem | Duas amostras disjuntas: **E** (exploração) e **M** (modelagem), estratificadas por tipo de ataque | Etapa 2 | Dataset completo → E e M |
| 3 | Análise exploratória dos dados (EDA) | Tipos de atributo, distribuição do atributo alvo, valores ausentes, outliers, redundâncias, atalhos | Etapa 3 | Só E |
| 4 | Pré-processamento **antes da divisão** | Limpeza (duplicatas; atributos redundantes e quase constantes; `Std` ausente → 0); escolha do conjunto de atributos | Etapas 2 e 4 | E e M, linha a linha |
| 5 | Divisão dos dados em treino, validação e teste | Folds estratificados por `label34`, gerados uma vez e congelados | Etapa 5 | E (spot-checking) e M (validação cruzada aninhada) |
| 6 | Pré-processamento **ajustado no treino** | `log1p`, padronização e codificação *one-hot*, ajustados no conjunto de treinamento de cada fold | Etapa 4 (dentro do `Pipeline`) | Treino de cada fold |
| 7 | Treinamento e spot-checking | 8 algoritmos com hiperparâmetros padrão e um classificador de referência (`DummyClassifier`) | Etapa 6 | E |
| 8 | Otimização de hiperparâmetros e seleção de modelos | Busca aleatória no **laço interno** da validação cruzada aninhada | Etapa 7 | Treino de cada fold externo de M (subdividido em treino e validação) |
| 9 | Estimativa de desempenho e comparação de modelos | **Laço externo** da validação cruzada aninhada; métricas; testes de hipóteses | Etapas 7 e 8 | Folds externos de teste de M |
| 10 | Interpretação do modelo | Importância por permutação, SHAP e ablação dos atalhos, só no melhor modelo | Etapa 9 | Folds externos de teste de M |
| 11 | Modelo final | Ajuste dos hiperparâmetros e retreino em toda a amostra M | Etapa 10 | M inteira |
| 12 | Reprodutibilidade e comunicação | `make all`, reprodução cruzada, artigo, revisão por pares, apresentação | Etapas 10 a 12 | — |

Dentro de cada fold, a ordem é sempre a mesma:
1. separar o conjunto de treinamento do conjunto de validação (ou de teste);
2. ajustar o pré-processamento no treino (`fit`) e aplicá-lo ao treino e à validação ou ao teste (`transform`);
3. treinar o modelo no treino;
4. predizer na validação ou no teste;
5. comparar rótulo e predição para obter a estimativa de desempenho.

O `Pipeline` do scikit-learn garante essa ordem nos dois laços.

### 1.4 Glossário (termos das aulas)

O diário, o guia e o plano usam os termos das aulas. O código e o artigo usam o equivalente em inglês.

| Termo das aulas | Em inglês (código e artigo) | Neste trabalho |
|---|---|---|
| Instância, exemplo | instance, sample | Uma janela de pacotes (uma linha do CSV) |
| Atributos preditivos (x) | features | 39 colunas brutas; 27 depois da limpeza (D13) |
| Atributo alvo (y), classe | target, class | `y`: ataque = classe positiva (1), benigno = classe negativa (0) |
| Rótulo | label | `label34` (34 rótulos), `category8` (8 categorias); nunca são atributos |
| Conjunto de treinamento, de validação e de teste | training, validation and test sets | Treino e validação no laço interno; treino e teste no laço externo |
| Holdout (retenção), de 2 ou de 3 vias | holdout (2-way, 3-way) | Não é usado como estimativa principal: depende demais de uma única partição |
| Validação cruzada *k*-fold estratificada | stratified k-fold cross-validation | Estratificada por `label34`, o que preserva também a proporção de `y` |
| Validação cruzada repetida | repeated k-fold cross-validation | Spot-checking: 5 folds × 2 repetições, em E |
| Validação cruzada aninhada | nested cross-validation | Em M: laço externo de 5 folds e laço interno de 3 folds |
| Laço interno | inner loop | Ajuste de hiperparâmetros (papel do conjunto de validação) |
| Laço externo | outer loop | Estimativa de desempenho (papel do conjunto de teste) |
| Otimização (ajuste) de hiperparâmetros | hyperparameter tuning | `RandomizedSearchCV` no laço interno (D9) |
| Configuração de hiperparâmetros | hyperparameter configuration | Um ponto do espaço de busca (`configs/search_spaces.yaml`) |
| Spot-checking | spot-checking | Comparação rápida dos algoritmos com hiperparâmetros padrão (Etapa 6) |
| Seleção de modelos | model selection | Escolha dos 3 finalistas (Etapa 6) e da melhor configuração (Etapa 7) |
| Estimativa de desempenho, capacidade de generalização | performance estimate, generalisation | Média ± desvio padrão nos 5 folds externos de teste |
| Modelo final | final model | Melhor algoritmo e configuração, retreinado em toda a amostra M (Etapa 10) |
| Data leakage (vazamento, contaminação de dados) | data leakage | Qualquer informação de validação ou de teste que chegue ao treino (§10.2) |
| Overfitting (sobreajuste), underfitting (subajuste) | overfitting, underfitting | Diagnosticados pela diferença entre o desempenho no treino e na validação |
| Viés indutivo | inductive bias | Critério de diversidade na escolha dos algoritmos (D7) |
| Limpeza de dados | data cleaning | Duplicatas, valores ausentes, atributos redundantes e quase constantes |
| Tratamento de outliers | outlier handling | Detecção pelo método IQR; sem remoção; `log1p` nos atributos de cauda longa |
| Normalização (min-max), padronização (z-score) | min-max scaling, standardisation | Padronização (`StandardScaler`) nos algoritmos sensíveis à escala |
| Codificação (*one-hot*) | one-hot encoding | `Protocol Type`, que é nominal |
| Discretização | discretisation | Não aplicada (§7, Etapa 4) |
| Desbalanceamento de classes, amostragem de dados | class imbalance, resampling | Resolvido pelo desenho da amostra (D3); ver a proposta D16 no diário |
| Redução de dimensionalidade: seleção e extração de atributos | feature selection, feature extraction (PCA) | Nenhuma seleção supervisionada nem PCA (§7, Etapa 4) |
| Matriz de confusão: VP, VN, FP, FN | confusion matrix: TP, TN, FP, FN | Base de todas as métricas (D5, D6) |
| Sensibilidade, recall, revocação, taxa de verdadeiros positivos (TVP) | sensitivity, recall, TPR | Taxa de detecção de ataques; também por tipo de ataque |
| Especificidade, taxa de verdadeiros negativos | specificity, TNR | Acerto no tráfego benigno |
| Taxa de falsos positivos (TFP = 1 − especificidade) | false positive rate, FPR | Taxa de alarme falso |
| Precisão | precision | Depende da prevalência; não se transfere para uma casa real |
| F1-score, acurácia, acurácia balanceada | F1, accuracy, balanced accuracy | Métricas secundárias (D6) |
| Curva ROC e AUC; curva PR e AUPRC | ROC curve, ROC-AUC; PR curve, PR-AUC | Calculadas a partir das predições fora do fold |
| Coeficiente de correlação de Matthews (MCC) | Matthews correlation coefficient | Métrica principal (D5). Não aparece nos slides; a justificativa é Chicco e Jurman (2020) |
| Teste de hipóteses; erros tipo I e II | hypothesis test; type I and II errors | Comparação dos finalistas (D10) |
| Amostra E, amostra M | exploration sample, modelling sample | Termos do projeto: E serve para decidir, M para ajustar e avaliar |
| Atalho | shortcut | Termo do projeto. Corresponde ao "atributo ilegítimo" (L2) da taxonomia de data leakage (§10.2) |

---

## 2. Problema, objetivo e perguntas de pesquisa

### 2.1 Tarefa

Classificação **binária** supervisionada:

- **Instância:** uma linha do CSV do CICIoT2023, que resume uma janela de pacotes **[verificar no artigo como a janela é definida e qual o seu tamanho]**.
- **Classe positiva:** `y = 1` para ataque (qualquer um dos 33 tipos); `y = 0` para tráfego benigno.
- **Rótulo auxiliar:** o rótulo original (34 valores) fica na coluna `label34`, e a categoria (8 valores) na coluna `category8`. As duas **nunca entram como atributo**. Servem para estratificar a amostragem e os folds e para medir a sensibilidade (recall) por tipo de ataque.

### 2.2 Objetivo (rascunho)

**PT:** Este trabalho desenvolve e avalia modelos de AM para distinguir tráfego benigno de tráfego malicioso em redes IoT, a partir dos atributos estatísticos de janelas de pacotes do CICIoT2023 (disponível em <link>): 39 colunas na versão por captura, das quais 27 ficam depois da limpeza. Além de identificar o melhor modelo com uma estratégia de avaliação sem data leakage, investigamos quais tipos de ataque escapam da detecção e se o desempenho depende de atributos que identificam o ambiente de captura ou o processamento dos dados, e não o ataque.

**EN (para o artigo):** *This work develops and evaluates machine learning models to distinguish benign from malicious traffic in IoT networks, using the statistical features of packet windows in the CICIoT2023 dataset (available at <link>): 39 columns in the per-capture release, 27 after cleaning. Beyond identifying the best model under a leakage-free evaluation strategy, we investigate which attack types escape detection and whether performance relies on features that identify the capture environment or the data processing rather than the attack itself.*

### 2.3 Por que a pergunta não é só "qual modelo acerta mais"

Na literatura, a detecção binária no CICIoT2023 costuma reportar desempenho perto de 99% **[verificar na revisão de literatura]**. Se todos os modelos ficarem no teto, a comparação entre eles diz pouco e o critério de originalidade fica fraco. A contribuição do trabalho está em dois pontos: o rigor do protocolo de avaliação e a explicação de **onde** e **por que** o modelo acerta ou falha.

### 2.4 Perguntas de pesquisa

- **Q1 (desempenho):** com uma estratégia de avaliação sem data leakage (validação cruzada aninhada com folds fixos: laço interno para o ajuste de hiperparâmetros, laço externo para a estimativa de desempenho), qual algoritmo atinge o maior MCC, e as diferenças entre os melhores são estatisticamente significativas?
- **Q2 (onde falha):** quais tipos de ataque o melhor modelo deixa passar (sensibilidade por tipo) e qual a taxa de falsos positivos (TFP, o alarme falso) no tráfego benigno?
- **Q3 (atalhos):** o desempenho depende de atributos que identificam o ambiente de captura (`Duration`, que seria o TTL, e `IAT`)? Quanto o MCC e o recall por tipo caem quando eles são removidos?

> **Revisão da Q3 (D12, 2026-10-25; ver o diário):** a pergunta original fica registrada acima. Nos dados, `Duration` chama-se `Time_To_Live`. A EDA mostrou que o tamanho da janela (`Number`) é um atalho mais forte que o TTL. Por isso, os candidatos a atalho passaram a ser `Number` e `Time_To_Live`, com `IAT` mantido só como comparação, e as métricas são reportadas também por grupo de janela (10 e 100 pacotes).

### 2.5 Hipóteses (para confirmar ou refutar, sem forçar o resultado)

- **H1:** ensembles de árvores (Random Forest, LightGBM) terão o maior MCC, mas a diferença para uma árvore de decisão simples será pequena.
- **H2:** ataques de baixo volume (Web-based, Brute Force e parte do Recon) concentrarão os falsos negativos, por se parecerem com tráfego benigno.
- **H3:** `Duration` (TTL) sozinho separa grande parte do tráfego, por refletir o sistema operacional das máquinas atacantes, e não o comportamento do ataque.

As hipóteses ficam como foram escritas antes de olhar os dados, para não ajustá-las ao resultado. A H3 é testada com `Time_To_Live`, o nome da coluna nesta versão.

---

## 3. O dataset CICIoT2023

### 3.1 Identificação

| Item | Informação |
|---|---|
| Nome | CIC IoT Dataset 2023 (CICIoT2023) |
| Instituição | Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB), Canadá |
| Página oficial | https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| Download | http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/ **[verificar se exige formulário e quais são os termos de uso e redistribuição]** |
| Artigo de referência | Neto, E. C. P.; Dadkhah, S.; Ferreira, R.; Zohourian, A.; Lu, R.; Ghorbani, A. A. *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment*. Sensors, 23(13), 5941, 2023. DOI: 10.3390/s23135941 |
| Link do artigo | https://www.mdpi.com/1424-8220/23/13/5941 |

### 3.2 Como foi gerado

- Topologia com 105 dispositivos IoT reais **[verificar]**.
- 33 ataques em 7 categorias, executados por dispositivos IoT maliciosos contra outros dispositivos IoT.
- Processamento: Mergecap, TCPDump, DPKT para extração de atributos e PySpark **[verificar]**.
- O rótulo parece ser atribuído por captura **[verificar]**. Se for o caso, tráfego de fundo dentro de uma captura de ataque pode estar rotulado como ataque (ruído de rótulo). Isso deve ser registrado como limitação.

### 3.3 Os 33 ataques por categoria

| Categoria | Ataques |
|---|---|
| DDoS (12) | ACK fragmentation, UDP flood, SlowLoris, ICMP flood, ICMP fragmentation, RSTFIN flood, PSHACK flood, HTTP flood, UDP fragmentation, TCP flood, SYN flood, SynonymousIP flood |
| DoS (4) | TCP flood, HTTP flood, SYN flood, UDP flood |
| Recon (5) | Ping sweep, OS scan, Vulnerability scan, Port scan, Host discovery |
| Web-based (6) | SQL injection, Command injection, Backdoor malware, Uploading attack, XSS, Browser hijacking |
| Brute Force (1) | Dictionary brute force |
| Spoofing (2) | ARP spoofing, DNS spoofing |
| Mirai (3) | GREIP flood, Greeth flood, UDPPlain |

A grafia exata dos rótulos nos CSVs (por exemplo, `BenignTraffic`, `DDoS-ICMP_Flood`) precisa ser confirmada na Etapa 1 **[verificar]**. O mapeamento `label34 → category8 → y` deve ser escrito a partir dos rótulos reais, versionado em `src/data/labels.py` e registrado no diário.

### 3.4 Estrutura de diretórios

1. **PCAP:** tráfego original. Não é necessário neste trabalho, a menos que a Etapa 1 conclua que é preciso reprocessar para agrupar por captura.
2. **CSV:** atributos extraídos, divididos em muitos arquivos parciais. **É o que usaremos.**
3. **Example:** notebook de exemplo.
4. **Supplementary material:** código e descrição das ferramentas de coleta e processamento.

### 3.5 Os 46 atributos

> **Correção (Etapa 1, ver `docs/DIARIO.md`, 2026-10-25):** a versão usada (CSVs por captura, D15) tem 39 colunas, e não os 46 atributos abaixo, que são os da versão misturada. `Duration` chama-se `Time_To_Live`; não existem `flow_duration`, `Srate`, `Drate`, `urg_count`, `Magnitue`, `Radius`, `Covariance` e `Weight`; existe `IGMP`. Os nomes exatos estão em `results/inspection/schema.json`.

Nomes exatos a confirmar no cabeçalho dos CSVs **[verificar]**.

| Grupo | Atributos |
|---|---|
| Fluxo | `flow_duration`, `Header_Length`, `Protocol Type`, `Duration`, `Rate`, `Srate`, `Drate` |
| Flags TCP (valor) | `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number` |
| Flags TCP (contagem) | `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count` |
| Protocolos (0/1) | `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC` |
| Tamanho de pacotes | `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size` |
| Tempo | `IAT` |
| Entrada × saída | `Number`, `Magnitue` (grafado assim), `Radius`, `Covariance`, `Variance`, `Weight` |

Segundo o artigo, `Duration` é o TTL dos pacotes **[verificar]**. Valores fracionários em contagens indicam que os atributos são médias sobre janelas de pacotes **[verificar]**.

### 3.6 Observações prévias (a confirmar na EDA)

Vêm das estatísticas publicadas na página oficial:

- **Redundância exata:** `Rate` e `Srate` têm estatísticas idênticas, assim como `IPv` e `LLC`.
- **Quase constantes:** `Drate`, `ece_flag_number`, `cwr_flag_number`, `Telnet`, `SMTP`, `IRC`, `DHCP` e `DNS` têm média praticamente zero.
- **Suspeitas de atalho:**
  - `Duration` (TTL) depende do sistema operacional de quem envia. Se as máquinas atacantes do laboratório tiverem sistemas diferentes dos dispositivos benignos, o modelo pode aprender "quem enviou", e não "o que é um ataque".
  - `IAT` tem valores na casa de dezenas de milhões com dispersão estreita, o que pode indicar um marcador temporal da captura **[hipótese]**.
- **Desbalanceamento:** o ataque domina o dataset, e o tráfego benigno é uma fração pequena **[verificar proporção]**.

### 3.7 Limitações conhecidas (vão para a seção de limitações do artigo)

1. **Ambiente de laboratório:** o tráfego benigno de uma casa real é mais variado.
2. **Prevalência invertida:** no CICIoT2023 o ataque domina; numa casa real, o benigno domina. Precisão e taxas de alarme absolutas não se transferem (falácia da taxa-base, Arp et al., 2022).
3. **Janelas quase duplicadas:** janelas consecutivas da mesma sessão de ataque são muito parecidas. Uma divisão aleatória pode colocar "irmãs" no treino e no teste e inflar o desempenho.
4. **Ruído de rótulo possível:** ver seção 3.2.
5. **Resultados quase perfeitos na literatura:** pedem cautela com data leakage e atalhos.
6. **Tamanho:** o conjunto completo não cabe na memória de máquinas comuns; é preciso amostrar de forma documentada.

---

## 4. Decisões metodológicas

O quadro abaixo traz as decisões como foram planejadas, com os termos das aulas. O estado vigente de cada uma, com as revisões e as confirmações, fica no quadro "Decisões vigentes" de `docs/DIARIO.md`.

| # | Decisão | Escolha | Justificativa | Status |
|---|---|---|---|---|
| D1 | Tarefa | Classificação binária; ataque = classe positiva (1), benigno = classe negativa (0) | Escopo enxuto; foco no rigor e na explicação | Decidido |
| D2 | Instância | Linha do CSV (janela de pacotes) | É a unidade disponibilizada pelo dataset | Decidido |
| D3 | Amostragem | 50% benigno e 50% ataque; ataque dividido igualmente entre os 33 tipos; amostra de exploração (E, ~50 mil instâncias) e de modelagem (M, ~200 mil), disjuntas | Sem estratificar por tipo, o DDoS domina e ataques raros somem. E e M disjuntas impedem que a escolha de algoritmos "veja" os dados de avaliação | Decidido; tamanhos ajustáveis à máquina. A distribuição de classes nos folds de teste é discutida na proposta D16 (diário, 2026-11-05) |
| D4 | Limpeza: instâncias duplicadas | Remover duplicatas exatas e conflitantes (mesmos atributos, rótulos diferentes) antes de qualquer divisão | Evita data leakage por duplicatas entre treino e teste e rótulos contraditórios | Decidido |
| D5 | Métrica principal | MCC | Usa os quatro quadrantes da matriz de confusão; recomendado para classificação binária (Chicco & Jurman, 2020) | Decidido |
| D6 | Métricas secundárias | F1 por classe, acurácia balanceada, AUPRC, AUC-ROC, sensibilidade (TVP, detecção), especificidade, TFP (alarme falso, 1 − especificidade), sensibilidade por tipo de ataque | Sensibilidade, especificidade, TFP e AUC-ROC não dependem da prevalência e são as mais transferíveis para fora do laboratório | Decidido. AUC-ROC e especificidade acrescentadas em 2026-11-06 (P3) |
| D7 | Algoritmos | Regressão logística, Gaussian Naive Bayes, kNN, árvore de decisão, SVM (kernel RBF), MLP, Random Forest, LightGBM, mais `DummyClassifier` como classificador de referência | Vieses indutivos diversos: linear, probabilístico, baseado em instâncias, árvores, kernel (margem máxima), redes neurais e ensembles (bagging e boosting) | SVM incluído em 2026-11-06 (P2), depois da conferência do programa da disciplina |
| D8 | Divisão dos dados (treino, validação e teste) | Spot-checking em E, com validação cruzada repetida (5 folds × 2 repetições); validação cruzada aninhada em M (laço externo de 5 folds para teste, laço interno de 3 folds para validação); folds estratificados por `label34`, gerados uma vez e salvos | Mesmas divisões para todos os algoritmos; o laço interno isola o ajuste de hiperparâmetros da estimativa de desempenho (Cawley & Talbot, 2010; Varma & Simon, 2006) | Decidido; sem agrupamento por captura (Etapa 1). Análises complementares decididas em 2026-11-06: *leave-capture-out* nos 16 tipos com 5 ou mais capturas, na Etapa 9; blocos contíguos de `row_idx` não adotados |
| D9 | Otimização de hiperparâmetros | `RandomizedSearchCV` (busca aleatória), 30 configurações, nos 3 finalistas e na árvore de decisão (P7, só para a H1) | Equilíbrio entre custo e cobertura | Executado na Etapa 7; bordas do espaço em aberto (P8, diário 2026-11-07) |
| D10 | Comparação de modelos (teste de hipóteses) | *Corrected resampled t-test* (Nadeau & Bengio, 2003) com correção de Holm; intervalos de confiança por bootstrap como complemento (P5) | O teste t pareado comum, com folds sobrepostos, tem erro tipo I elevado (Dietterich, 1998). A correção trata essa dependência. Friedman + Nemenyi (Demšar, 2006) é para vários datasets | Decidido (P5 adotada em 07/11) |
| D11 | Interpretação do modelo | Melhor modelo; importância por permutação e SHAP calculados nos folds externos de teste | Interpreta o modelo em dados que ele não viu; permite medir estabilidade entre folds | Decidido |
| D12 | Atalhos | Modelos de um atributo só e ablação do melhor modelo sem os candidatos a atalho | Testa diretamente a H3 e responde a Q3 | Revisado na Etapa 1 (candidatos: `Number` e `Time_To_Live`; ver o diário) |
| D13 | Pré-processamento | Definido a partir da EDA (seção 7, Etapa 4; `configs/features.yaml`) | Depende dos dados | Decidido na Etapa 3 |
| D14 | Reprodutibilidade | Git, `requirements.txt` com versões fixas, configs YAML, Makefile, manifesto de hashes, índices de amostra e de folds versionados | Permite reproduzir do CSV bruto ao artigo | Decidido |

---

## 5. Repositório e ambiente

### 5.1 Estrutura

```
cmp263-ciciot2023/
├── CLAUDE.md                  # instruções curtas para o Claude Code (seção 0.2)
├── README.md                  # como reproduzir tudo do zero
├── LICENSE
├── Makefile                   # alvos de cada etapa e "make all"
├── requirements.txt           # versões fixas
├── configs/
│   ├── base.yaml              # semente, caminhos, tamanhos de amostra, n_jobs
│   ├── features.yaml          # D13: atributos removidos, transformações e conjuntos (full, no_window, no_ttl, no_shortcut)
│   └── search_spaces.yaml     # espaços de busca por algoritmo
├── data/                      # NÃO versionado, exceto os itens marcados
│   ├── raw/                   # CSVs originais
│   ├── raw_manifest.csv       # arquivo, tamanho, SHA-256, nº de linhas (versionado)
│   ├── processed/             # sample_E.parquet, sample_M.parquet
│   ├── sample_index.csv.gz    # (arquivo de origem, linha) de cada linha sorteada (versionado)
│   └── splits/                # índices dos folds (versionado)
├── src/
│   ├── data/                  # inspect_raw.py, make_sample.py, labels.py
│   ├── features/              # (vazio: o pré-processamento fica em models/pipelines.py)
│   ├── models/                # registry.py (algoritmos), pipelines.py (pré-processamento + estimador)
│   ├── experiments/           # make_splits.py, spotcheck.py, nested_cv.py, ablation.py, final_model.py
│   ├── evaluation/            # metrics.py, stats.py
│   ├── interpretation/        # permutation.py, shap_analysis.py
│   ├── reporting/             # tables.py, figures.py
│   └── utils/                 # seed.py, io.py, run_metadata.py
├── notebooks/                 # só exploração e visualização; a lógica fica em src/
│   ├── 01_inspecao.ipynb
│   └── 02_eda.ipynb
├── results/                   # inspection/, eda/, spotcheck/, nested/, ablation/, interpretation/, figures/, tables/
├── models/                    # modelo final + metadados
├── tests/                     # test_splits.py, test_no_leakage.py
└── docs/
    ├── PLANO.md               # este arquivo
    ├── DIARIO.md              # diário de bordo (seção 6)
    ├── IA_LOG.md              # registro de uso de IA (seção 6.3)
    ├── enunciado.pdf
    └── artigo/                # rascunhos, template, figuras finais
```

### 5.2 Ambiente

- Python 3.11 ou 3.12. A versão usada vai para o README. Se o Colab for usado em alguma etapa, alinhar a versão local com a dele.
- Bibliotecas: `numpy`, `pandas`, `pyarrow`, `polars` (opcional, para ler os CSVs grandes), `scikit-learn`, `lightgbm`, `shap`, `scipy`, `matplotlib`, `pyyaml`, `joblib`, `jupyter`, `pytest`.
- Depois de instalar, fixar as versões exatas em `requirements.txt`.
- Determinismo:
  - semente única em `configs/base.yaml`, passada a todo `random_state`;
  - LightGBM com `deterministic=True`, `force_row_wise=True` e `n_jobs` fixo;
  - ordem de leitura dos arquivos sempre alfabética.

### 5.3 Alvos do Makefile

`make env`, `make inspect`, `make sample`, `make eda`, `make splits`, `make spotcheck`, `make tune`, `make evaluate`, `make interpret`, `make ablation`, `make final`, `make report` e `make all`, que executa tudo em sequência a partir dos CSVs brutos.

### 5.4 Regras de versionamento e de resultados

- Os dados brutos e as amostras em parquet não vão para o Git. O que vai: manifesto de hashes, índice da amostra (arquivo e linha) e índices dos folds. Isso permite reproduzir a amostra exata sem redistribuir o dataset.
- Cada execução de experimento grava, junto com os resultados, um `run.json` com data e hora, commit do Git, hash da configuração, versões das bibliotecas, semente e tempo de execução.
- Resultados nunca são sobrescritos silenciosamente. Se um experimento for refeito, a versão anterior é preservada ou o motivo da troca vai para o diário.

---

## 6. Documentação contínua (diário de bordo e registro de IA)

O artigo será escrito a partir do diário. Se uma decisão, um número ou uma descoberta não estiver no diário, ela tende a se perder ou a ser reconstruída de memória na hora de escrever, que é quando surgem erros. Por isso a documentação acontece **durante** o trabalho, e não no fim.

### 6.1 `docs/DIARIO.md`: quando escrever

Acrescente uma entrada:

- ao concluir cada etapa da seção 7 (`[ENTREGA]`);
- a cada decisão metodológica, com a justificativa (`[DECISÃO]`);
- a cada resultado inesperado, confirmação ou refutação de um item `[verificar]` ou de uma hipótese (`[DESCOBERTA]`);
- a cada problema relevante e à forma como foi resolvido (`[PROBLEMA]`);
- ao fim de cada sessão de trabalho, com um resumo curto do que foi feito e do que falta (`[SESSÃO]`);
- a cada ideia que pertence à dissertação, e não a este trabalho (`[IDEIA-DISSERTAÇÃO]`).

### 6.2 Regras e modelo de entrada

- O diário é **somente de acréscimo**. Entradas antigas não são editadas nem apagadas. Uma correção vira uma entrada nova que cita a anterior.
- A única parte editável é o quadro **"Decisões vigentes"**, no topo do arquivo, que resume o estado atual de D1 a D14 e aponta para as entradas que as justificam.
- Todo número traz o caminho do arquivo que o gerou e o commit. Número sem fonte não entra.
- O campo "Para o artigo" indica a seção do artigo onde a informação será usada. Quando possível, inclua uma frase-rascunho em inglês.

Modelo:

```markdown
## AAAA-MM-DD — Etapa N — título curto
**Tipo:** [ENTREGA] | [DECISÃO] | [DESCOBERTA] | [PROBLEMA] | [SESSÃO] | [IDEIA-DISSERTAÇÃO]
**Autor(es):** nome(s) + Claude Code (se usado)
**O que foi feito:** ...
**Resultados (com fonte):** valor — arquivo `results/...` — commit `abc1234`
**Decisões e justificativa:** ...
**Descobertas / surpresas:** ...
**Problemas e solução:** ...
**Para o artigo:** seção [Methods > Sampling] — rascunho: "We sampled ..."
**Pendências / [verificar]:** ...
```

### 6.3 `docs/IA_LOG.md`: registro de uso de IA

É exigência do enunciado: o artigo precisa citar as ferramentas de IA, e o material suplementar precisa trazer os prompts e as limitações observadas. Para cada uso relevante, registre:

```markdown
## AAAA-MM-DD — Ferramenta (ex.: Claude no claude.ai / Claude Code) — etapa
**Finalidade:** ...
**Prompt:** texto integral ou resumo fiel (para prompts longos)
**O que foi aproveitado:** ...
**O que foi rejeitado ou corrigido, e por quê:** ...
**Limitações ou potencialidades observadas:** ...
```

Inclua também as conversas de planejamento feitas no claude.ai antes do início da implementação, que deram origem a este plano.

### 6.4 Consolidação semanal

No fim de cada semana do cronograma, escreva uma entrada `[SESSÃO]` de consolidação com: etapas concluídas, decisões tomadas, números principais e riscos para a semana seguinte. Essas entradas viram o esqueleto das seções de Métodos e Resultados.

---

## 7. Etapas passo a passo

Cada etapa traz **objetivo**, **tarefas**, **saídas**, **critério de pronto** e **o que registrar no diário**. As datas estão na seção 8.

### Etapa 0 — Configuração do projeto

**Objetivo:** repositório, ambiente e documentação prontos antes de tocar nos dados.

**Tarefas:**

1. Criar o repositório no GitHub (privado durante o trabalho, público na entrega) com a estrutura da seção 5.1.
2. Criar o ambiente virtual, instalar as bibliotecas e fixar as versões em `requirements.txt`.
3. Criar `configs/base.yaml` com semente, caminhos, tamanhos de amostra e `n_jobs`.
4. Criar `src/utils/seed.py` e `src/utils/run_metadata.py`.
5. Criar `CLAUDE.md`, `docs/PLANO.md`, `docs/DIARIO.md` (com o quadro "Decisões vigentes"), `docs/IA_LOG.md` e copiar `docs/enunciado.pdf`.
6. Criar o esqueleto do Makefile.
7. Iniciar o download dos CSVs. Ele pode demorar; comece já.

**Saídas:** repositório com o primeiro commit; ambiente funcional.

**Pronto quando:** `make env` funciona numa pasta limpa e o diário tem a primeira entrada.

**Diário:** `[ENTREGA]` com a versão do Python, a máquina usada (CPU, RAM) e a origem do download.

### Etapa 1 — Aquisição e inspeção dos dados brutos

**Objetivo:** conhecer a estrutura real dos CSVs e confirmar ou refutar os itens `[verificar]` antes de qualquer decisão.

**Tarefas:**

1. Baixar a pasta de CSVs da fonte oficial. Registrar a URL, a data e, se houver, os termos de uso. Se usar um espelho (Kaggle, por exemplo), registrar e justificar.
2. Escrever `src/data/inspect_raw.py`, que gera:
   - `data/raw_manifest.csv`: nome, tamanho, SHA-256, número de linhas e de colunas de cada arquivo;
   - `results/inspection/label_counts_by_file.csv`: contagem de cada rótulo em cada arquivo;
   - `results/inspection/label_counts_total.csv`: total por rótulo, por categoria e por classe binária;
   - `results/inspection/schema.json`: nomes e tipos das colunas e consistência entre arquivos.
3. Ler os arquivos em blocos (pandas com `chunksize` ou `polars.scan_csv`), sem carregar tudo na memória.
4. Ler as seções do artigo do dataset sobre extração de atributos e rotulagem.

**Perguntas que esta etapa precisa responder (todas vão para o diário):**

- Os 46 atributos e a coluna de rótulo têm os nomes esperados? Qual a grafia exata dos 34 rótulos?
- Quantas linhas há no total, por rótulo e por categoria? Qual a fração de benigno? Quais são os tipos de ataque mais raros e quantas linhas eles têm?
- Cada arquivo mistura rótulos e capturas? Existe alguma coluna ou informação que identifique captura, sessão, dispositivo ou tempo e permita agrupar linhas?
- Há valores nulos ou infinitos? Em quais colunas?
- Como o artigo define a janela de pacotes? `Duration` é mesmo o TTL? Como o rótulo é atribuído?

**Decisão a tomar:** se houver como agrupar por captura, os folds usam `StratifiedGroupKFold`. Se não houver, a divisão é por linha, e o risco de janelas quase duplicadas entre treino e teste vira uma limitação explícita do artigo.

**Saídas:** manifesto, contagens, esquema e respostas às perguntas acima.

**Pronto quando:** todas as perguntas acima têm resposta no diário, com fonte.

**Diário:** `[ENTREGA]`, um `[DESCOBERTA]` para cada `[verificar]` confirmado ou refutado e um `[DECISÃO]` sobre o agrupamento.

### Etapa 2 — Amostragem

**Objetivo:** gerar, de forma determinística e documentada, as amostras de exploração (E) e de modelagem (M).

**Composição (ajustável à memória da máquina; registrar no diário):**

| Amostra | Benigno | Ataque | Total aproximado |
|---|---|---|---|
| E (exploração e spot-checking) | 25.000 | 25.000, ~758 por tipo | 50.000 |
| M (otimização e avaliação) | 100.000 | 100.000, ~3.030 por tipo | 200.000 |

**Procedimento (`src/data/make_sample.py`):**

1. Ler `label_counts_total.csv` para saber quantas linhas existem de cada rótulo.
2. Calcular o alvo de cada rótulo para E e M. Se um tipo de ataque tiver menos linhas do que E + M exigem, usar todas as disponíveis e dividi-las 20% para E e 80% para M.
3. Fazer uma passada de sorteio. Para cada rótulo, a probabilidade de seleção é `p = min(1, 1,3 × alvo_total / disponível)`; o fator 1,3 dá folga para as duplicatas. Percorrer os arquivos em ordem alfabética, em blocos de tamanho fixo, com um único gerador `numpy.random.default_rng(SEED)` consumido sempre na mesma ordem. Guardar `source_file` e `row_idx` de cada linha sorteada.
4. Remover duplicatas exatas (atributos + `label34`). Identificar duplicatas conflitantes (mesmos atributos e `y` diferente), contar, remover todas as cópias e registrar a contagem.
5. Cortar cada rótulo para o alvo exato com `sample(n, random_state=SEED)` e separar E e M. Como a deduplicação foi feita na união, E e M são disjuntas.
6. Se o total de ataque ficar abaixo do alvo (tipos raros), reduzir o benigno para manter a proporção 1:1.
7. Salvar `data/processed/sample_E.parquet` e `sample_M.parquet` (39 atributos da versão por captura + `label34` + `category8` + `y` + `source_file` + `row_idx`), `data/sample_index.csv.gz`, os SHA-256 dos parquets e `results/sampling/sampling_report.json` com as contagens finais por rótulo em cada amostra.

**O que deve constar no artigo:** a proporção 50/50 e a distribuição uniforme entre os tipos de ataque são escolhas de desenho do estudo, e não reproduzem a distribuição original nem a de uma casa real. Por isso, a sensibilidade (TVP), a TFP e a sensibilidade por tipo são as métricas mais transferíveis, e a precisão não se transfere.

**Relação com as aulas:** pela regra de quando transformar os dados (§1.3), remover instâncias pode ser feito no conjunto completo, e por isso a amostragem não causa data leakage. Já a aula de desbalanceamento de classes pede que os conjuntos de validação e de teste mantenham a distribuição original dos dados. Aqui, como a amostragem vem antes da divisão, os folds de teste ficam com a distribuição do desenho (50/50 e tipos igualados), e não com a do dataset (97,65% de ataque). A proposta D16 (diário, 2026-11-05) trata disso.

**Pronto quando:** rodar o script duas vezes produz arquivos com o mesmo hash.

**Diário:** `[ENTREGA]` com as contagens finais, as duplicatas exatas e conflitantes removidas e qualquer tipo de ataque que ficou abaixo do alvo.

### Etapa 3 — Análise exploratória (somente na amostra E)

**Objetivo:** entender os dados, confirmar ou descartar as suspeitas da seção 3.6 e decidir o pré-processamento.

**Tarefas (funções em `src/`, figuras salvas em `results/eda/` por script, notebook `02_eda.ipynb` só para visualizar):**

1. **Tabela-resumo por atributo:** tipo, mínimo, máximo, média, mediana, % de zeros, nº de valores distintos, % de nulos e infinitos.
2. **Composição das classes** (binária, 8 categorias, 34 tipos) na amostra, mostrando o efeito do desenho amostral.
3. **Atributos constantes e quase constantes:** variância zero, ou valor mais frequente acima de 99,9% das linhas.
4. **Redundância:** correlação de Pearson e de Spearman (mapa de calor), pares com |ρ| > 0,99 e verificação de colunas idênticas (`Rate`/`Srate`, `IPv`/`LLC`).
5. **Distribuição de cada atributo por classe** (benigno × ataque): histogramas em escala log ou violinos para os de cauda pesada. Ranking de separabilidade univariada (AUC ou estatística KS de cada atributo sozinho).
6. **Suspeitos de atalho:**
   - `Duration`: frequência de valores por classe e por tipo de ataque (por exemplo, um mesmo TTL dominando todos os ataques).
   - `IAT`: distribuição por classe e por tipo, verificando se se comporta como marcador temporal.
7. **Modelos de um atributo só:** árvore de profundidade 2 usando apenas `Duration`, depois apenas `IAT`, em validação cruzada na amostra E, medindo o MCC. Um MCC alto com um único atributo é evidência forte de atalho.
8. **Outliers:** resumo por IQR. Por padrão, **não** remover outliers, porque tráfego de ataque é, por natureza, atípico, e removê-lo pode apagar o sinal. Justificar no artigo.
9. **Duplicatas:** reportar as taxas encontradas na Etapa 2.

**Saídas:** figuras e tabelas da EDA, lista de atributos a remover e de transformações, gravada em `configs/features.yaml`.

**Pronto quando:** cada suspeita da seção 3.6 está confirmada ou descartada, com figura ou tabela, e `features.yaml` está preenchido.

**Diário:** `[ENTREGA]`, `[DESCOBERTA]` para cada suspeita e `[DECISÃO]` com a lista de atributos removidos e o motivo de cada um.

### Etapa 4 — Pré-processamento dos dados

**Objetivo:** aplicar o pré-processamento decidido na EDA (D13, `configs/features.yaml`) sem data leakage, seguindo a regra de quando transformar os dados (§1.3). O pré-processamento tem duas partes:
- **antes da divisão**, as operações que removem instâncias ou atributos ou que aplicam uma regra fixa;
- **dentro do `Pipeline`**, as operações cujos parâmetros são estimados no conjunto de treinamento de cada fold.

**As tarefas de pré-processamento das aulas, uma a uma** (os números vêm da EDA em E: `results/eda/`, commit `c3b2bf2`):

| Tarefa (Aulas 13–16) | O que fazemos | Quando | Por quê |
|---|---|---|---|
| Limpeza: instâncias duplicadas | Exatas: fica uma cópia. Conflitantes (mesmos atributos, `y` diferente) e de tipo ambíguo: saem todas as cópias (D4) | Antes da divisão (Etapa 2) | Remove instâncias, sem parâmetro estimado; evita duplicatas entre treino e teste (L1.4, §10.2) |
| Limpeza: atributos redundantes | Saem `Tot size`, `LLC`, `IPv` e `Variance` (identidades exatas), as 4 contagens e `Tot sum` (cópias escaladas pelo tamanho da janela) e `Rate` (Spearman −0,994 com `IAT`) | Antes da divisão | Remove atributos sem usar o atributo alvo. Decidido em E, nunca em M |
| Limpeza: atributos quase constantes | Saem `Telnet` e `SMTP` (mais de 99,9% de zeros) | Antes da divisão | Idem |
| Limpeza: valores ausentes | `Std` ausente vira 0. Só ocorre nas janelas de 1 pacote, que não têm dispersão. O `Rate` infinito sai junto com a coluna | Antes da divisão | Regra fixa, sem parâmetro estimado; não é imputação por média ou mediana. Depois dela não sobra valor ausente nem infinito, e `prepare` interrompe a execução se sobrar |
| Limpeza: dados inconsistentes | `IAT` negativo em 2 instâncias do dataset inteiro (mínimo de −0,018 s), mantido | — | Valor pequeno, compatível com pacotes fora de ordem; `log1p` continua definido |
| Tratamento de outliers | Detecção pelo método IQR na EDA: até 21% das instâncias ficam fora de 1,5 × IQR. Sem remoção nem winsorização. Transformação `log1p` em `Min`, `Max`, `Std` e `IAT` (cauda longa, assimetria > 2) nos algoritmos sensíveis à escala | `log1p` dentro do `Pipeline`; não tem parâmetro | Tráfego de ataque é atípico por natureza, e remover os outliers apagaria o sinal. A transformação log é a estratégia indicada nas aulas para cauda longa à direita |
| Transformação: normalização ou padronização | Padronização (`StandardScaler`) dos atributos numéricos, depois do `log1p`, só em LR, Naive Bayes, kNN, SVM e MLP | Dentro do `Pipeline`, ajustada no treino de cada fold | Médias e desvios são estimados no treino e aplicados à validação e ao teste. Árvores não são sensíveis à escala |
| Transformação: codificação | `Protocol Type` em *one-hot* em todos os pipelines, inclusive nas árvores (P6, 2026-11-06). As categorias são aprendidas no treino, e uma categoria nova vira zeros (`handle_unknown="ignore"`) | Dentro do `Pipeline` | O atributo é nominal: códigos de protocolo IP (0, 1, 6, 17, 47) não têm ordem. Uma árvore conseguiria isolar cada código com vários cortes, mas a regra das aulas para atributo nominal é o *one-hot* |
| Transformação: discretização | Não aplicada | — | Nenhum algoritmo escolhido exige atributos categóricos, e as árvores já particionam os atributos numéricos |
| Balanceamento de classes | Resolvido pelo desenho da amostra (D3: 50/50, tipos igualados). Sem SMOTE nem aprendizado sensível a custo | Antes da divisão (Etapa 2) | Sobre a distribuição dos folds de teste, ver a proposta D16 (diário) |
| Redução de dimensionalidade | Nenhuma seleção de atributos supervisionada nem PCA. Os conjuntos `no_window`, `no_ttl` e `no_shortcut` são definidos por hipótese (D12), e não pelo desempenho | — | Ver o parágrafo abaixo da tabela |

**Redução de dimensionalidade.** Ela não é feita, por três motivos:
- 27 atributos para cerca de 154 mil instâncias de treino não trazem maldição da dimensionalidade;
- o PCA misturaria os atributos e impediria a análise de atalhos atributo a atributo (Q3);
- árvores e LightGBM já fazem seleção embutida.

Se houvesse seleção supervisionada, ela teria de ficar dentro da validação cruzada.

**Implementação** (commit `e16e460`):
- `src/models/pipelines.py`:
  - `feature_names(columns, feature_cfg, feature_set, auxiliary)`: os atributos preditivos do conjunto (`full`, `no_window`, `no_ttl` ou `no_shortcut`), sem as colunas auxiliares (`label34`, `category8`, `y`, `source_file`, `row_idx`);
  - `prepare(df, features, feature_cfg)`: o pré-processamento antes da divisão, linha a linha;
  - `build_pipeline(model_name, features, feature_cfg, search_cfg, seed, n_jobs)`: um `Pipeline` com o pré-processamento ajustado no treino e o estimador. Os hiperparâmetros do estimador ficam acessíveis como `model__<nome>`.
- `src/models/registry.py`: os 8 algoritmos de D7 (o SVM entrou em 2026-11-06, P2) e o `DummyClassifier`, com os hiperparâmetros padrão e os parâmetros fixos de `configs/search_spaces.yaml`.

**Pronto quando** `tests/test_no_leakage.py` confirma:
- que os parâmetros ajustados (médias e desvios da padronização, categorias do *one-hot*) dependem só das instâncias de treino;
- que mudar as instâncias de teste não muda nada do que foi ajustado;
- que a validação cruzada reajusta o pré-processamento em cada fold;
- que o pré-processamento antes da divisão é linha a linha.

**Diário:** `[ENTREGA]` e `[DECISÃO]` para as transformações escolhidas.

### Etapa 5 — Divisão dos dados: treino, validação e teste

**Objetivo:** gerar e congelar todas as divisões antes de treinar qualquer modelo. Assim, todos os algoritmos usam exatamente os mesmos conjuntos de treinamento, de validação e de teste em cada iteração.

**Como os conjuntos das aulas aparecem aqui:**
- **Spot-checking (amostra E):** validação cruzada repetida e estratificada. Em cada iteração, 4 folds formam o conjunto de treinamento e o quinto é avaliado. Como o resultado serve para escolher os finalistas, que é uma decisão de seleção de modelos, esses folds fazem o papel de conjunto de **validação**, e não de teste. A estimativa de desempenho final nunca vem de E.
- **Validação cruzada aninhada (amostra M):**
  - *laço externo:* em cada uma das 5 iterações, 4 folds formam o conjunto de treinamento externo e o quinto é o **conjunto de teste**, que só serve para a estimativa de desempenho;
  - *laço interno:* o treino externo é dividido em 3 folds; em cada iteração, 2 formam o conjunto de **treinamento** e o terceiro é o conjunto de **validação**, usado só para escolher a configuração de hiperparâmetros.

  É o *holdout* de 3 vias (treino, validação e teste) das aulas, com cada partição trocada por uma validação cruzada.

**Tarefas (`src/experiments/make_splits.py`):**

1. **Spot-checking (amostra E):** `RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)`, estratificado por `label34`.
2. **Laço externo (amostra M):** `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`, estratificado por `label34`. Isso preserva tanto a proporção das classes quanto a de cada tipo de ataque em todos os folds.
3. **Laço interno:** para cada fold externo `k`, `StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)` sobre as instâncias de treino de `k`, estratificado por `label34`. Atenção: os índices internos são relativos ao subconjunto de treino do fold externo.
4. Agrupar por captura (`StratifiedGroupKFold`) não se aplica (D8: 15 tipos têm uma só captura). As análises complementares da D8 (*leave-capture-out* e blocos contíguos de `row_idx`) são decididas aqui.
5. Salvar todos os índices em `data/splits/` com hash.
6. Opcional, se houver tempo de máquina: repetir o laço externo com uma segunda semente (5 × 2).

**Pronto quando:** `tests/test_splits.py` confirma:
- que os folds externos são disjuntos e cobrem M;
- que nenhuma instância de teste aparece no treino nem na validação do laço interno;
- que as proporções por `label34` estão preservadas.

**Diário:** `[ENTREGA]` com o tamanho de cada partição.

### Etapa 6 — Spot-checking (amostra E)

**Objetivo:** escolher, de forma informada, os 3 algoritmos que seguem para a otimização de hiperparâmetros. Spot-checking não é ajuste fino: compara rapidamente algoritmos de vieses indutivos diferentes, com hiperparâmetros padrão (enunciado).

**Tarefas (`src/experiments/spotcheck.py`):**

1. Treinar os 8 algoritmos mais o `DummyClassifier` com hiperparâmetros padrão (só o necessário para convergir, como `max_iter`, e `random_state` fixo), nos mesmos folds, com os pipelines da Etapa 4 (`build_pipeline`).
2. Registrar por fold: MCC, F1 por classe, acurácia balanceada, AUPRC, AUC-ROC, sensibilidade (TVP), especificidade, TFP, tempo de treino e de predição. Salvar em `results/spotcheck/metrics_per_fold.csv`. Registrar também o MCC no próprio conjunto de treinamento, para diagnosticar overfitting (treino muito acima da validação) e underfitting (os dois baixos) (P4, 2026-11-06).
3. Gerar a tabela de média ± desvio e o boxplot de MCC por algoritmo.
4. Selecionar os 3 com maior MCC médio. Se as diferenças estiverem dentro de um desvio-padrão (empate no teto), escolher os 3 com vieses indutivos mais diferentes e registrar que o spot-checking não discriminou. Isso também é um resultado.

**Atenção:** o kNN pode ser lento na predição. Se for preciso reduzir dados para ele, registrar.

**Pronto quando:** os 3 finalistas estão escolhidos e justificados.

**Diário:** `[ENTREGA]` com a tabela e `[DECISÃO]` com os finalistas.

### Etapa 7 — Otimização de hiperparâmetros e seleção de modelos com validação cruzada aninhada (amostra M)

**Objetivo:** estimar o desempenho de cada finalista sem o viés da seleção de hiperparâmetros. O laço interno escolhe a configuração usando os conjuntos de validação. O laço externo mede o desempenho dessa escolha num conjunto de teste que não participou dela. Reutilizar o mesmo conjunto para ajustar e para avaliar daria estimativas otimistas (Aula 04, *holdout* de 3 vias).

**Esqueleto:**

```python
mcc_scorer = make_scorer(matthews_corrcoef)
features = feature_names(M.columns, feature_cfg, "full", auxiliary_columns(cfg))
X, y = prepare(M, features, feature_cfg), M["y"].to_numpy()

assign = load_assignment(cfg, NESTED, M)                      # folds congelados da Etapa 5 (make_splits.py)
for k, (train_idx, test_idx) in enumerate(outer_folds(assign)):   # laço externo: treino × teste
    search = RandomizedSearchCV(
        estimator=build_pipeline(model_name, features, feature_cfg, search_cfg, SEED, cfg["n_jobs"]),
        param_distributions=search_spaces[model_name],        # chaves com o prefixo model__
        n_iter=cfg["search"]["n_iter"],                       # 30 configurações
        scoring=mcc_scorer,
        cv=inner_folds(assign, k),    # laço interno: treino × validação, posições relativas ao treino do fold k
        refit=True,                   # retreina a melhor configuração em todo o treino externo
        return_train_score=True,      # diagnóstico de overfitting (proposta P4)
        random_state=SEED,
        n_jobs=cfg["n_jobs"],
    )
    search.fit(X.iloc[train_idx], y[train_idx])
    proba = search.predict_proba(X.iloc[test_idx])[:, 1]      # só agora o fold de teste é usado
    save_fold(model_name, k, search.best_params_, search.best_score_, test_idx, proba)
```

- Salvar por fold: melhores hiperparâmetros, MCC na validação (laço interno), métricas no teste (laço externo) e o escore de cada instância de teste (predições fora do fold). Implementado em `results/nested/<modelo>/`: `fold<k>.json`, `fold<k>_search.csv`, `fold<k>_predictions.csv.gz` e `fold<k>_timing.json`, mais a tabela consolidada `results/nested/folds.csv`.
- Avaliar também o `DummyClassifier` nos folds externos, como piso.
- Implementar retomada: se o arquivo de um fold já existe, pular. A execução é longa.
- Hiperparâmetros diferentes entre folds são esperados. Registrar e discutir a estabilidade.

**Espaços de busca iniciais** (`configs/search_spaces.yaml`; ajustar se a busca bater nos limites). Na execução de 07/11, o MLP e a Random Forest escolheram a borda do espaço em todos os folds; o que fazer está na proposta P8 (diário, entrada "2026-11-07 — Etapa 7"):

| Algoritmo | Hiperparâmetros |
|---|---|
| Regressão logística | `C` ~ log-uniforme(1e-3, 1e2); `solver="lbfgs"`; `max_iter=2000` |
| Gaussian NB | `var_smoothing` ~ log-uniforme(1e-12, 1e-3) |
| kNN | `n_neighbors` ∈ {3, 5, …, 51}; `weights` ∈ {uniform, distance}; `p` ∈ {1, 2} |
| Árvore de decisão | `max_depth` ∈ {3, …, 30, None}; `min_samples_leaf` ∈ [1, 200]; `criterion` ∈ {gini, entropy} |
| SVM (RBF) | `C` ~ log-uniforme(1e-2, 1e3); `gamma` ~ log-uniforme(1e-4, 1); escore = `decision_function`, sem calibração de Platt |
| MLP | `hidden_layer_sizes` ∈ {(32,), (64,), (64, 32), (128, 64)}; `alpha` ~ log-uniforme(1e-6, 1e-2); `learning_rate_init` ~ log-uniforme(1e-4, 1e-2); `early_stopping=True` |
| Random Forest | `n_estimators` ∈ [100, 500]; `max_depth` ∈ {10, 20, 30, None}; `max_features` ∈ {sqrt, log2, 0.5}; `min_samples_leaf` ∈ [1, 50] |
| LightGBM | `n_estimators` ∈ [100, 1000]; `learning_rate` ~ log-uniforme(0.01, 0.3); `num_leaves` ∈ [15, 255]; `min_child_samples` ∈ [10, 200]; `subsample` ∈ [0.6, 1] com `subsample_freq=1`; `colsample_bytree` ∈ [0.6, 1]; `reg_lambda` ~ log-uniforme(1e-3, 10) |

**Pronto quando:** os 3 finalistas, a árvore de decisão (P7) e o Dummy têm resultados nos 5 folds externos. Concluído em 07/11 (resultados no commit `6da18f6`).

**Diário:** `[ENTREGA]` com o tempo de execução e a variação dos hiperparâmetros entre folds.

### Etapa 8 — Estimativa de desempenho e comparação de modelos

**Objetivo:** responder a Q1 e a Q2 com as predições do laço externo, isto é, em conjuntos de teste que não participaram de nenhuma decisão.

**Tarefas (`src/evaluation/`, `src/reporting/`):**

1. **Tabela principal:** média ± desvio padrão nos 5 folds externos de MCC, F1 (ataque), F1 (benigno), acurácia balanceada, AUPRC, AUC-ROC, sensibilidade (TVP), especificidade e TFP (1 − especificidade), para os finalistas e o Dummy.
2. **Gráfico por fold** (boxplot ou pontos ligados) do MCC de cada finalista.
3. **Teste de hipóteses:** *corrected resampled t-test* entre o melhor modelo e cada um dos outros, com correção de Holm. Hipótese nula: os dois modelos têm desempenho equivalente. Com apenas 5 folds o teste tem pouco poder (risco de erro tipo II), e isso deve ser dito no artigo. Complemento adotado (P5, diário, 07/11): intervalos de confiança de 95% por bootstrap (percentis 2,5 e 97,5, B ≥ 1000) sobre as predições fora do fold, também para a diferença entre modelos (reamostragem pareada), como na Aula 08. O bootstrap mede só a variação do conjunto de teste, e o artigo deve dizer isso.

   ```python
   def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
       d = np.asarray(scores_a) - np.asarray(scores_b)
       k = len(d)
       var = np.var(d, ddof=1)
       if var == 0:
           return np.nan, np.nan
       t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
       p = 2 * stats.t.sf(abs(t), df=k - 1)
       return t, p
   ```

4. **Matriz de confusão** do melhor modelo, a partir das predições fora do fold.
5. **Sensibilidade (recall) por tipo de ataque** (33 tipos e 7 categorias) e **TFP no benigno**, com intervalo de confiança de Wilson a 95%. Gráfico de barras horizontais ordenado do pior para o melhor tipo. Também por grupo de janela (10 × 100 pacotes, D12).
6. **Curvas PR e ROC** do melhor modelo, com AUPRC e AUC-ROC, a partir das predições fora do fold.
7. **Distribuição original (D16, P1 adotada em 07/11):** (a) as métricas da tabela principal reponderadas para a distribuição original do dataset, com peso por `label34` igual à fração no dataset dividida pela fração em M; (b) a precisão projetada para outras prevalências de ataque a partir da sensibilidade e da especificidade, PPV = sens·π / (sens·π + (1 − espec)·(1 − π)).

**Pronto quando:** as tabelas e figuras são geradas por `make evaluate` e Q1 e Q2 estão respondidas no diário.

**Diário:** `[ENTREGA]` e `[DESCOBERTA]` para H1 e H2 (confirmadas ou não).

### Etapa 9 — Interpretação do modelo e análise de atalhos

**Objetivo:** responder a Q3 e cumprir o eixo de interpretabilidade, com o **melhor modelo apenas**.

**Princípio:** interpretar os modelos treinados em cada fold externo usando os respectivos dados de teste, que eles não viram. Isso também permite medir a estabilidade das explicações entre folds.

**Tarefas:**

1. **Importância por permutação** (métrica MCC, 10 repetições) em cada fold externo de teste. Reportar média ± desvio entre folds e a estabilidade do ranking (correlação de Spearman entre folds e sobreposição do top 10).
2. **SHAP:**
   - Se o melhor modelo for de árvores, usar `TreeExplainer` numa subamostra de cerca de 2.000 linhas de cada fold de teste. Gerar o gráfico de resumo (beeswarm) e gráficos de dependência para os atributos mais importantes, em especial `Number`, `Time_To_Live` e `IAT`. Os valores SHAP das colunas *one-hot* de `Protocol Type`, se houver, são somados por atributo original.
   - Se não for de árvores, ficar com permutação e gráficos de dependência parcial (PDP), e registrar o motivo.
3. **Explicações locais:** SHAP para 4 a 6 erros (falsos negativos dos tipos de ataque com pior sensibilidade e falsos positivos no benigno), discutidos no artigo.
4. **Testes de atalho:**
   - reavaliar os modelos de um atributo só (`Number`, `Time_To_Live` e, como comparação, `IAT`) nos folds externos de M;
   - **ablação:** repetir a Etapa 7 só para o melhor algoritmo com `no_shortcut` (sem `Number` e `Time_To_Live`), nos mesmos folds. Se houver tempo, `no_window` e `no_ttl`, que removem cada um separadamente;
   - comparar MCC (teste corrigido) e sensibilidade por tipo de ataque entre `full` e `no_shortcut`, separando os grupos de janela 10 e 100. **Essa é a tabela central do artigo.**
5. **Leitura dos resultados:** queda pequena indica que o modelo é robusto a esses atributos; queda grande é evidência de atalho; colapso em tipos específicos de ataque indica quais ataques são reconhecidos pelo ambiente, e não pelo comportamento.
6. ***Leave-capture-out*** (análise complementar da D8, item (a), adotado em 2026-11-06): treinar o melhor algoritmo, com a configuração de hiperparâmetros do modelo final (Etapa 10, tarefa 1, que pode rodar antes desta), nos folds do laço externo (`nested_M`) e nos folds por captura (`capture_M`). Comparar a sensibilidade por tipo, sobre as predições fora do fold, nos 16 tipos agrupados por captura. Uma queda grande indica que o desempenho depende de ver, no treino, janelas da mesma captura que está no teste. Custo: 10 treinos.

**Pronto quando:** Q3 está respondida no diário com números e figuras.

**Diário:** `[ENTREGA]` e `[DESCOBERTA]` para H3.

### Etapa 10 — Modelo final e reprodutibilidade

**Objetivo:** deixar tudo reproduzível do CSV bruto até as figuras do artigo.

**Tarefas:**

1. **Modelo final para disponibilização:** rodar a busca de hiperparâmetros (3 folds) em toda a amostra M com o melhor algoritmo, retreinar e salvar `models/final_model.joblib` com os hiperparâmetros, as versões, a semente, o hash dos dados e o commit. Esse modelo **não é avaliado de novo**: a estimativa de desempenho dele é a da validação cruzada aninhada. É o último passo do *holdout* das aulas (Aula 04): depois de estimar o desempenho, o modelo final é treinado com todos os dados disponíveis. Ele não é nenhum dos modelos dos folds. Escolher o modelo do fold de melhor desempenho seria um erro metodológico. Isso deve ser dito no artigo.
2. **`make all`:** executa tudo, dos CSVs brutos às tabelas (CSV e LaTeX) e figuras, via `src/reporting/`.
3. **README:** visão geral, requisitos, como baixar os dados (com os hashes esperados), comandos `make`, tempo de execução esperado, máquina usada e um mapa de cada figura e tabela do artigo para o script que a gera.
4. **Reprodução cruzada:** cada integrante da dupla clona o repositório num ambiente limpo e roda `make all`. Comparar hashes e métricas. Diferenças (por exemplo, não determinismo com várias threads) são corrigidas ou documentadas. Esse teste vale um parágrafo no artigo.
5. **Publicação:** tornar o repositório público, criar a tag `v1.0` na entrega e, opcionalmente, gerar um DOI no Zenodo.

**Pronto quando:** a reprodução cruzada chega aos mesmos números.

**Diário:** `[ENTREGA]` e `[PROBLEMA]` para qualquer divergência encontrada.

### Etapa 11 — Artigo

**Objetivo:** escrever o artigo a partir do diário (estrutura na seção 9).

**Tarefas:**

1. Confirmar no Moodle o formato do template (LaTeX/Overleaf ou Word).
2. Rascunhar Métodos já a partir da Etapa 5, usando as entradas marcadas "Para o artigo".
3. Escrever Resultados, Interpretação e Discussão após as Etapas 8 e 9.
4. Escrever Introdução e Trabalhos Relacionados a partir da revisão de literatura (seção 8).
5. Escrever a declaração de uso de IA e montar o material suplementar com o `IA_LOG.md`.
6. Passar o checklist da seção 10. Cada integrante revisa o texto inteiro.
7. Gerar o PDF, conferir o limite de 20 páginas e entregar no Moodle junto com o código.

**Pronto quando:** o PDF e o código estão no Moodle.

### Etapa 12 — Revisão por pares e apresentação

1. **Revisão por pares (24 a 30/11):** cada aluno revisa 2 artigos. A nota é individual. O checklist da seção 10 serve de roteiro.
2. **Slides:** problema; dataset e amostragem; divisão em treino, validação e teste; resultados; sensibilidade por tipo; achado sobre atalhos; reprodutibilidade; limitações.
3. **Ensaio com cronômetro**, respeitando o tempo que a professora definir.
4. **Apresentação em 01/12** e participação ativa nas discussões dos outros grupos (nota individual).

---

## 8. Cronograma

Hoje é **25/10/2026 (domingo)**. Restam quatro semanas até a meta interna de entrega (20/11, sexta-feira), com dois dias de folga antes do prazo oficial (22/11, 23h59).

> **Situação em 07/11/2026:** Etapas 0 a 7 concluídas. As Etapas 4 e 5 terminaram em 05/11 (previsto: 01–02/11), a 6 em 06/11 (previsto: 03–04/11) e a 7 em 07/11, dentro da janela prevista (04–08/11). O atraso de cerca de 2 dias foi absorvido, a menos que a proposta P8 leve a refazer algum modelo. A validação cruzada aninhada levou cerca de 5,4 h de cálculo, o dobro do estimado. O plano de contingência (§8.2) continua valendo. O cronograma da disciplina (v2) confirma a revisão por pares em 24/11 e a apresentação em 01/12.

| Período | Etapa(s) | Marco (pronto quando) |
|---|---|---|
| 25–26/10 | 0 | Repositório, ambiente, `CLAUDE.md`, diário e registro de IA criados; download iniciado |
| 26–28/10 | 1 | Manifesto e contagens prontos; `[verificar]` resolvidos; decisão sobre agrupamento registrada |
| 28–29/10 | 2 | Amostras E e M geradas, com hashes estáveis entre execuções |
| 29–31/10 | 3 | EDA concluída; `features.yaml` preenchido; suspeitas de atalho confirmadas ou descartadas |
| 01–02/11 | 4 e 5 | Pipelines e folds congelados; testes de data leakage e de divisão passando |
| 03–04/11 | 6 | Spot-checking concluído; 3 finalistas escolhidos |
| 04–08/11 | 7 | Validação cruzada aninhada concluída para os 3 finalistas e o Dummy |
| 08–10/11 | 8 | Tabelas, figuras e testes estatísticos; Q1 e Q2 respondidas |
| 10–13/11 | 9 | Interpretação e ablação; Q3 respondida |
| 13–15/11 | 10 | Modelo final salvo; `make all` funcionando |
| 14–19/11 | 11 | Artigo completo e revisado; reprodução cruzada concluída |
| **20/11** | Entrega | PDF e código no Moodle |
| 23–30/11 | 12 | Revisões por pares (24–30/11), slides e ensaio |
| **01/12** | 12 | Apresentação oral |

**Em paralelo:**

- **Revisão de literatura (01–07/11),** aproveitando o tempo em que a validação cruzada aninhada roda. Ver seção 8.1.
- **Rascunho de Métodos a partir de 02/11,** com base no diário.
- **Consolidação semanal no diário** (seção 6.4) em 31/10, 07/11, 14/11 e 20/11.

### 8.1 Revisão de literatura (o que procurar)

O objetivo é mostrar a lacuna que o trabalho preenche: a maior parte dos trabalhos reporta métricas altíssimas com divisão aleatória e sem investigar atalhos.

- O artigo do CICIoT2023 e os resultados de referência que ele reporta para 2 classes.
- 4 a 6 trabalhos que usam o CICIoT2023 em detecção binária. Para cada um, anotar: protocolo de divisão, métricas, tratamento de desbalanceamento e se investigam data leakage ou atalhos.
- Problemas metodológicos em datasets de detecção de intrusão: Engelen et al. (2021) sobre o CICIDS2017; Arp et al. (2022).
- Aprendizado por atalho: Geirhos et al. (2020).
- Vazamento e reprodutibilidade em ciência baseada em AM: Kapoor e Narayanan (2023).

Registrar cada leitura no diário com uma linha de resumo e a seção do artigo onde será citada.

### 8.2 Plano de contingência

Se o cronograma atrasar, cortar nesta ordem:

1. Ablações separadas de `Number` e `Time_To_Live` (`no_window`, `no_ttl`; manter só a conjunta, `no_shortcut`).
2. Segunda repetição do laço externo (já é opcional).
3. Iterações da busca: de 30 para 15.
4. Tamanho da amostra M: de 200 mil para 100 mil linhas.
5. Explicações SHAP locais (manter as globais e a permutação).

**Nunca cortar:** a separação entre dados de decisão e de teste; MCC e sensibilidade por tipo de ataque; a análise de atalhos; pelo menos 5 algoritmos; manifesto, sementes, README e `make all`; o diário e o registro de IA.

---

## 9. O artigo

### 9.1 Estrutura e orçamento de páginas (limite de 20 páginas, incluindo referências e apêndices)

| Seção | Conteúdo | Páginas |
|---|---|---|
| Abstract | Problema, método, números principais, achado sobre atalhos | — |
| 1. Introduction | Motivação (IoT doméstica vulnerável), resultados quase perfeitos na literatura, Q1–Q3, contribuições | 1,5 |
| 2. Related Work | Trabalhos com o CICIoT2023; problemas metodológicos em datasets de intrusão; aprendizado por atalho | 1,5–2 |
| 3. Data | Dataset, inspeção, amostragem e deduplicação, principais achados da EDA | 2,5 |
| 4. Methods | Pré-processamento, algoritmos, divisão dos dados (treino, validação e teste), métricas, otimização, teste estatístico, interpretação | 2,5 |
| 5. Results | Spot-checking, validação cruzada aninhada, testes de hipóteses, sensibilidade por tipo de ataque | 3 |
| 6. Interpretation and Shortcut Analysis | Permutação, SHAP, modelos de um atributo, ablação | 2,5 |
| 7. Discussion and Limitations | Respostas a Q1–Q3; limitações da seção 3.7; ameaças à validade | 1,5 |
| 8. Conclusion | Síntese e trabalhos futuros | 0,5 |
| Reproducibility and AI Usage Statements | Repositório, tag, hashes, reprodução cruzada; ferramentas de IA usadas e onde | 0,5 |
| References | | 2 |

### 9.2 Figuras e tabelas previstas

**Figuras:**

1. Diagrama da estratégia de avaliação: amostragem → E e M → spot-checking → validação cruzada aninhada (laço interno: treino × validação; laço externo: treino × teste) → interpretação e ablação.
2. `Number` e `Time_To_Live` por tipo de ataque (EDA: `fig_number_by_type.png`, `fig_by_type.png`).
3. MCC por algoritmo no spot-checking (boxplot).
4. MCC por fold externo dos finalistas.
5. Recall por tipo de ataque, `full` × `no_shortcut` (figura central).
6. Resumo SHAP do melhor modelo.
7. (Opcional) Curva PR do melhor modelo.

**Tabelas:**

1. Composição das amostras E e M.
2. Atributos removidos e motivo.
3. Resultados da validação cruzada aninhada (média ± desvio).
4. Testes estatísticos.
5. Ablação (MCC e sensibilidade por categoria, `full` × `no_shortcut`).

Os espaços de busca e os hiperparâmetros escolhidos em cada fold podem ir para um apêndice curto, que também conta no limite de páginas.

### 9.3 Orientações de escrita

- Inglês, template do Moodle, coluna única.
- Não explicar a teoria dos algoritmos; citar e seguir em frente.
- Todo número do texto deve vir de uma tabela ou figura gerada por script.
- Descrever o protocolo com detalhe suficiente para outra pessoa reproduzir sem o código.
- Citar as ferramentas de IA conforme as diretrizes do Moodle e apontar para o material suplementar.

---

## 10. Armadilhas e checklist final

### 10.1 Armadilhas de Arp et al. (2022) e como o plano trata cada uma

| Armadilha | Como o plano trata |
|---|---|
| Viés de amostragem | Desenho amostral explícito (D3), documentado e discutido como limitação |
| Rótulos imprecisos | Verificação de como o rótulo é atribuído (Etapa 1); duplicatas conflitantes removidas e contadas |
| *Data snooping* (data leakage) | Amostras E e M disjuntas; pré-processamento ajustado dentro do `Pipeline`; folds congelados; testes automatizados (§10.2) |
| Correlações espúrias | Modelos de um atributo só e ablação de `Number` e `Time_To_Live` (D12, Etapa 9) |
| Seleção enviesada de hiperparâmetros | Validação cruzada aninhada |
| Linha de base inadequada | `DummyClassifier` e modelos simples (regressão logística, Naive Bayes, árvore) |
| Métrica inadequada | MCC como principal, mais sensibilidade, TFP e sensibilidade por tipo; nunca só acurácia |
| Falácia da taxa-base | Discussão explícita de que, numa casa real, o benigno domina e mesmo uma TFP baixa gera muitos alarmes falsos |
| Avaliação só em laboratório | Registrada como limitação; generalização fica fora do escopo |

### 10.2 Tipos de data leakage (Kapoor & Narayanan, 2023) e como o plano trata cada um

É a taxonomia do material extra da Aula 04 (*Nested Cross-Validation & Data Leakage*). Este quadro é o roteiro da seção do artigo que argumenta, tipo a tipo, que não há data leakage (as *model info sheets* do mesmo trabalho).

| Código | Tipo | Risco neste trabalho | Como o plano trata |
|---|---|---|---|
| L1.1 | Sem conjunto de teste (treinar e avaliar nos mesmos dados) | — | Validação cruzada aninhada; o desempenho no treino só serve para diagnosticar overfitting |
| L1.2 | Pré-processamento no conjunto completo (imputação, normalização ou *oversampling* antes da divisão) | Padronização e *one-hot* | Ajustados dentro do `Pipeline`, no treino de cada fold (Etapa 4, `tests/test_no_leakage.py`). Antes da divisão, só operações sem parâmetro estimado |
| L1.3 | Seleção de atributos no conjunto completo | Remoção de atributos (D13) | Remoção sem uso do atributo alvo, decidida em E (disjunta de M); nenhuma seleção supervisionada |
| L1.4 | Duplicatas entre treino e teste | Janelas idênticas | Deduplicação antes da divisão (D4); nenhum vetor de atributos se repete em E ∪ M |
| L2 | Atributos ilegítimos (*proxies* do atributo alvo) | `Number` (tamanho da janela) e `Time_To_Live` | Análise de atalhos: modelos de um atributo e ablação (D12, Etapa 9) |
| L3.1 | Data leakage temporal | Ordem das janelas desconhecida **[verificar]** | Nenhum atributo de tempo absoluto; blocos contíguos de `row_idx` como análise complementar (D8) |
| L3.2 | Dependência entre instâncias (a mesma origem no treino e no teste) | Janelas da mesma captura no treino e no teste | Inevitável com 15 tipos de uma só captura (D8). *Leave-capture-out* como análise complementar; limitação no artigo |
| L3.3 | Viés de amostragem no conjunto de teste | Folds de teste 50/50 e com tipos igualados (D3) | Métricas independentes da prevalência; D16 (P1, adotada em 07/11): métricas reponderadas para a distribuição original e precisão projetada para outras prevalências (Etapa 8) |

### 10.3 Checklist antes da entrega

- [ ] Problema, objetivo, perguntas de pesquisa e link do dataset descritos com clareza.
- [ ] EDA com gráficos e sumarização estatística.
- [ ] Pré-processamento justificado item a item, para cada tarefa das aulas, inclusive as não aplicadas: limpeza, outliers, normalização e padronização, codificação, discretização, balanceamento e redução de dimensionalidade (Etapa 4).
- [ ] Pelo menos 5 algoritmos com vieses diversos, constantes do programa da disciplina.
- [ ] Métrica principal e métricas secundárias justificadas.
- [ ] Divisão em treino, validação e teste descrita com os termos das aulas; mesmas divisões para todos os algoritmos; sementes fixas.
- [ ] Nenhuma decisão usou os folds externos de teste.
- [ ] Cada tipo de data leakage do §10.2 discutido no artigo.
- [ ] Spot-checking e otimização descritos (espaços de busca, número de iterações).
- [ ] Resultados com média ± desvio, gráficos e teste estatístico.
- [ ] Interpretação do melhor modelo, com discussão de achados pertinentes ou inesperados.
- [ ] Limitações (laboratório, desenho 50/50, prevalência invertida, janelas quase duplicadas, atalhos).
- [ ] Repositório público, README, `requirements.txt`, `make all` reproduzindo do zero, tag `v1.0`.
- [ ] Reprodução cruzada feita pelos dois integrantes.
- [ ] Uso de IA citado no artigo e material suplementar com os prompts pronto.
- [ ] Até 20 páginas, template correto, em inglês.
- [ ] Todos os itens `[verificar]` resolvidos ou removidos do texto.

---

## 11. Referências

Todas vieram de memória e devem ser **conferidas antes de citar [verificar]**. As marcadas "(slides)" são citadas nos slides da disciplina, mas os dados bibliográficos ainda precisam ser conferidos na fonte.

**Material da disciplina** (fonte da terminologia, §1.3 e §1.4)

- Recamonde-Mendoza, M. Slides de CMP263 – Aprendizagem de Máquina, PPGC/UFRGS, 2026/2: Aula 04 (*Método holdout*) e material extra (*Nested Cross-Validation & Data Leakage in Predictive Modeling*, baseado em Kapoor & Narayanan, 2023); Aula 08 (*Avaliação de modelos — Divisão de dados*); Aula 09 (*Métricas de desempenho*); Aulas 02b e 13–14 (*Pré-processamento: normalização e codificação; limpeza e transformação; desbalanceamento de classes*); Aulas 15–16 (*Redução de dimensionalidade*).

**Dataset**

- Neto, E. C. P.; Dadkhah, S.; Ferreira, R.; Zohourian, A.; Lu, R.; Ghorbani, A. A. *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment*. Sensors, 23(13), 5941, 2023. DOI: 10.3390/s23135941.

**Metodologia em segurança e problemas de datasets**

- Arp, D. et al. *Dos and Don'ts of Machine Learning in Computer Security*. USENIX Security Symposium, 2022.
- Engelen, G.; Rimmer, V.; Joosen, W. *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*. IEEE Security and Privacy Workshops, 2021.
- Geirhos, R. et al. *Shortcut Learning in Deep Neural Networks*. Nature Machine Intelligence, 2, 665–673, 2020.
- Kapoor, S.; Narayanan, A. *Leakage and the Reproducibility Crisis in Machine-Learning-Based Science*. Patterns, 4(9), 2023.

**Avaliação e comparação de modelos**

- Cawley, G. C.; Talbot, N. L. C. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR, 11, 2079–2107, 2010.
- Varma, S.; Simon, R. *Bias in Error Estimation When Using Cross-Validation for Model Selection*. BMC Bioinformatics, 7, 91, 2006.
- Nadeau, C.; Bengio, Y. *Inference for the Generalization Error*. Machine Learning, 52, 239–281, 2003.
- Holm, S. *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 6(2), 65–70, 1979.
- Chicco, D.; Jurman, G. *The Advantages of the Matthews Correlation Coefficient (MCC) over F1 Score and Accuracy in Binary Classification Evaluation*. BMC Genomics, 21, 6, 2020.
- Dietterich, T. G. *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*. Neural Computation, 10(7), 1895–1923, 1998 (slides).
- Demšar, J. *Statistical Comparisons of Classifiers over Multiple Data Sets*. JMLR, 7, 1–30, 2006 (slides).
- Efron, B. *Better Bootstrap Confidence Intervals*. Journal of the American Statistical Association, 82(397), 1987. DOI: 10.2307/2289144 (slides).
- Kuncheva, L. I.; Rodríguez, J. J. *On Feature Selection Protocols for Very Low-Sample-Size Data*. Pattern Recognition, 81, 660–673, 2018 (slides).

**Interpretabilidade**

- Breiman, L. *Random Forests*. Machine Learning, 45, 5–32, 2001 (importância por permutação).
- Lundberg, S. M.; Lee, S.-I. *A Unified Approach to Interpreting Model Predictions*. NeurIPS, 2017.

**Reprodutibilidade**

- Pineau, J. et al. *Improving Reproducibility in Machine Learning Research*. JMLR, 22, 2021.
