# Trabalho Final CMP263 — Detecção binária de ataques em redes IoT com o CICIoT2023

**Documento de contexto e plano de execução** · versão 1.0 · 25/10/2026
Disciplina: CMP263 – Aprendizagem de Máquina (PPGC/UFRGS), Profa. Mariana Recamonde Mendoza · Trabalho em dupla
Entrega do artigo e do código: **22/11/2026, 23h59** (meta interna: **20/11/2026**)

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
- Múltiplas repetições, mesmas divisões para todos os algoritmos em cada iteração, ausência de vazamento e sementes fixas.
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

---

## 2. Problema, objetivo e perguntas de pesquisa

### 2.1 Tarefa

Classificação **binária** supervisionada:

- **Instância:** uma linha do CSV do CICIoT2023, que resume uma janela de pacotes **[verificar no artigo como a janela é definida e qual o seu tamanho]**.
- **Classe positiva:** `y = 1` para ataque (qualquer um dos 33 tipos); `y = 0` para tráfego benigno.
- **Rótulo auxiliar:** o rótulo original (34 valores) fica na coluna `label34`, e a categoria (8 valores) na coluna `category8`. As duas **nunca entram como atributo**. Servem para estratificar a amostragem e os folds e para medir o recall por tipo de ataque.

### 2.2 Objetivo (rascunho)

**PT:** Este trabalho desenvolve e avalia modelos de AM para distinguir tráfego benigno de tráfego malicioso em redes IoT, a partir de 46 atributos estatísticos extraídos de janelas de pacotes do CICIoT2023 (disponível em <link>). Além de identificar o melhor modelo sob um protocolo sem vazamento de dados, investigamos quais tipos de ataque escapam da detecção e se o desempenho depende de atributos que identificam o ambiente de captura, e não o ataque.

**EN (para o artigo):** *This work develops and evaluates machine learning models to distinguish benign from malicious traffic in IoT networks, using 46 statistical features extracted from packet windows of the CICIoT2023 dataset (available at <link>). Beyond identifying the best model under a leakage-free evaluation protocol, we investigate which attack types escape detection and whether performance relies on features that identify the capture environment rather than the attack itself.*

### 2.3 Por que a pergunta não é só "qual modelo acerta mais"

Na literatura, a detecção binária no CICIoT2023 costuma reportar desempenho perto de 99% **[verificar na revisão de literatura]**. Se todos os modelos ficarem no teto, a comparação entre eles diz pouco e o critério de originalidade fica fraco. A contribuição do trabalho está em dois pontos: o rigor do protocolo de avaliação e a explicação de **onde** e **por que** o modelo acerta ou falha.

### 2.4 Perguntas de pesquisa

- **Q1 (desempenho):** sob um protocolo sem vazamento (validação cruzada aninhada com folds fixos), qual algoritmo atinge o maior MCC, e as diferenças entre os melhores são estatisticamente significativas?
- **Q2 (onde falha):** quais tipos de ataque o melhor modelo deixa passar (recall por tipo) e qual a taxa de alarme falso no tráfego benigno?
- **Q3 (atalhos):** o desempenho depende de atributos que identificam o ambiente de captura (`Duration`, que seria o TTL, e `IAT`)? Quanto o MCC e o recall por tipo caem quando eles são removidos?

### 2.5 Hipóteses (para confirmar ou refutar, sem forçar o resultado)

- **H1:** ensembles de árvores (Random Forest, LightGBM) terão o maior MCC, mas a diferença para uma árvore de decisão simples será pequena.
- **H2:** ataques de baixo volume (Web-based, Brute Force e parte do Recon) concentrarão os falsos negativos, por se parecerem com tráfego benigno.
- **H3:** `Duration` (TTL) sozinho separa grande parte do tráfego, por refletir o sistema operacional das máquinas atacantes, e não o comportamento do ataque.

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
5. **Resultados quase perfeitos na literatura:** pedem cautela com vazamento e atalhos.
6. **Tamanho:** o conjunto completo não cabe na memória de máquinas comuns; é preciso amostrar de forma documentada.

---

## 4. Decisões metodológicas

| # | Decisão | Escolha | Justificativa | Status |
|---|---|---|---|---|
| D1 | Tarefa | Binária; ataque = 1, benigno = 0 | Escopo enxuto; foco no rigor e na explicação | Decidido |
| D2 | Unidade | Linha do CSV (janela de pacotes) | É a unidade disponibilizada pelo dataset | Decidido |
| D3 | Amostragem | 50% benigno e 50% ataque; ataque dividido igualmente entre os 33 tipos; amostra de exploração (E, ~50 mil linhas) e de modelagem (M, ~200 mil linhas), disjuntas | Sem estratificar por tipo, o DDoS domina e ataques raros somem. E e M disjuntas impedem que a escolha de algoritmos "veja" os dados de avaliação | Decidido; tamanhos ajustáveis à máquina |
| D4 | Duplicatas | Remover duplicatas exatas e conflitantes (mesmos atributos, rótulos diferentes) antes de qualquer divisão | Evita vazamento trivial e rótulos contraditórios | Decidido |
| D5 | Métrica principal | MCC | Usa os quatro quadrantes da matriz de confusão; recomendado para classificação binária (Chicco & Jurman, 2020) | Decidido |
| D6 | Métricas secundárias | F1 por classe, acurácia balanceada, PR-AUC, TPR (detecção), FPR (alarme falso), recall por tipo de ataque | TPR e FPR não dependem da prevalência e são as mais transferíveis para fora do laboratório | Decidido |
| D7 | Algoritmos | Regressão logística, Gaussian Naive Bayes, kNN, árvore de decisão, MLP, Random Forest, LightGBM, mais `DummyClassifier` como piso | Cobrem viés linear, probabilístico, baseado em instâncias, de árvores, de redes neurais e de ensembles (bagging e boosting) | Confirmar que constam do programa da disciplina |
| D8 | Validação | Spot-checking em E (5 folds × 2 repetições); CV aninhada em M (5 folds externos × 3 internos); folds estratificados por `label34`, gerados uma vez e salvos | Atende "mesmas divisões para todos", evita viés de seleção de hiperparâmetros (Cawley & Talbot, 2010; Varma & Simon, 2006) | Decidido; agrupamento por captura depende da Etapa 1 |
| D9 | Otimização | `RandomizedSearchCV`, 30 iterações, só nos 3 finalistas | Equilíbrio entre custo e cobertura | Ajustável |
| D10 | Estatística | *Corrected resampled t-test* (Nadeau & Bengio, 2003) com correção de Holm | Adequado a um único dataset com folds sobrepostos; Friedman + Nemenyi é para vários datasets | Decidido |
| D11 | Interpretação | Melhor modelo; importância por permutação e SHAP calculados nos folds externos de teste | Interpreta o modelo em dados que ele não viu; permite medir estabilidade entre folds | Decidido |
| D12 | Atalhos | Modelos de um atributo só (`Duration`, `IAT`) e ablação do melhor modelo sem `Duration` e `IAT` | Testa diretamente a H3 e responde a Q3 | Decidido |
| D13 | Pré-processamento | Definido após a EDA (seção 7, Etapa 4) | Depende dos dados | A decidir na Etapa 3 |
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
│   ├── features.yaml          # atributos removidos e conjuntos (full, no_shortcut)
│   └── search_spaces.yaml     # espaços de busca por algoritmo
├── data/                      # NÃO versionado, exceto os itens marcados
│   ├── raw/                   # CSVs originais
│   ├── raw_manifest.csv       # arquivo, tamanho, SHA-256, nº de linhas (versionado)
│   ├── processed/             # sample_E.parquet, sample_M.parquet
│   ├── sample_index.csv.gz    # (arquivo de origem, linha) de cada linha sorteada (versionado)
│   └── splits/                # índices dos folds (versionado)
├── src/
│   ├── data/                  # inspect_raw.py, make_sample.py, labels.py
│   ├── features/              # preprocess.py
│   ├── models/                # registry.py, pipelines.py
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

**O que deve constar no artigo:** a proporção 50/50 e a distribuição uniforme entre os tipos de ataque são escolhas de desenho do estudo, e não reproduzem a distribuição original nem a de uma casa real. Por isso, TPR, FPR e recall por tipo são as métricas mais transferíveis, e a precisão não se transfere.

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

### Etapa 4 — Pré-processamento

**Objetivo:** implementar o pré-processamento decidido na EDA sem vazamento.

**Transformações fixas** (decididas na amostra E, sem parâmetros aprendidos, aplicadas a M antes da divisão):

- remover atributos constantes e redundantes listados em `features.yaml`;
- separar `X` das colunas auxiliares (`label34`, `category8`, `source_file`, `row_idx`);
- substituir ±inf por NaN, se houver.

**Transformações ajustadas** (dentro de um `Pipeline` do scikit-learn, ajustadas só no treino de cada fold):

- `SimpleImputer(strategy="median")`, se a EDA mostrar nulos ou infinitos;
- para regressão logística, kNN, MLP e Naive Bayes: `log1p` nos atributos não negativos de cauda pesada (via `FunctionTransformer`) seguido de `StandardScaler`;
- para árvores e ensembles: nenhuma transformação de escala.

Não há reamostragem (SMOTE ou similar): o balanceamento já foi resolvido na amostragem. Também não há seleção automática de atributos além da remoção decidida na EDA, para manter a interpretação limpa.

**Implementação:** `src/models/pipelines.py` com `build_pipeline(model_name, feature_set)`, em que `feature_set` é `"full"` ou `"no_shortcut"` (sem `Duration` e `IAT`), para a ablação da Etapa 9.

**Pronto quando:** `tests/test_no_leakage.py` confirma que os parâmetros ajustados (médias do scaler, medianas do imputer) dependem só das linhas de treino do fold.

**Diário:** `[ENTREGA]` e `[DECISÃO]` para as transformações escolhidas.

### Etapa 5 — Protocolo de validação

**Objetivo:** gerar e congelar todas as divisões antes de treinar qualquer modelo.

**Tarefas (`src/experiments/make_splits.py`):**

1. **Spot-checking (amostra E):** `RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)`, estratificado por `label34`.
2. **Laço externo (amostra M):** `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`, estratificado por `label34`. Isso preserva tanto a proporção binária quanto a de cada tipo de ataque em todos os folds.
3. **Laço interno:** para cada fold externo `k`, `StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)` sobre as linhas de treino de `k`, estratificado por `label34`. Atenção: os índices internos são relativos ao subconjunto de treino do fold externo.
4. Se a Etapa 1 permitir agrupamento, trocar por `StratifiedGroupKFold`.
5. Salvar todos os índices em `data/splits/` com hash.
6. Opcional, se houver tempo de máquina: repetir o laço externo com uma segunda semente (5 × 2).

**Pronto quando:** `tests/test_splits.py` confirma que os folds externos são disjuntos e cobrem M, que nenhuma linha de teste aparece no treino interno e que as proporções por `label34` estão preservadas.

**Diário:** `[ENTREGA]` com o tamanho de cada partição.

### Etapa 6 — Spot-checking (amostra E)

**Objetivo:** escolher, de forma informada, os 3 algoritmos que seguem para a otimização.

**Tarefas (`src/experiments/spotcheck.py`):**

1. Treinar os 7 algoritmos mais o `DummyClassifier` com hiperparâmetros padrão (apenas o necessário para convergir, como `max_iter`, e `random_state` fixo), nos mesmos folds.
2. Registrar por fold: MCC, F1 por classe, acurácia balanceada, PR-AUC, TPR, FPR, tempo de treino e de predição. Salvar em `results/spotcheck/metrics_per_fold.csv`.
3. Gerar a tabela de média ± desvio e o boxplot de MCC por algoritmo.
4. Selecionar os 3 com maior MCC médio. Se as diferenças estiverem dentro de um desvio-padrão (empate no teto), escolher os 3 com vieses indutivos mais diferentes e registrar que o spot-checking não discriminou. Isso também é um resultado.

**Atenção:** o kNN pode ser lento na predição. Se for preciso reduzir dados para ele, registrar.

**Pronto quando:** os 3 finalistas estão escolhidos e justificados.

**Diário:** `[ENTREGA]` com a tabela e `[DECISÃO]` com os finalistas.

### Etapa 7 — Otimização de hiperparâmetros com CV aninhada (amostra M)

**Objetivo:** estimar o desempenho de cada finalista sem viés de seleção.

**Esqueleto:**

```python
mcc_scorer = make_scorer(matthews_corrcoef)

for k, (train_idx, test_idx) in enumerate(outer_folds):
    search = RandomizedSearchCV(
        estimator=build_pipeline(model_name, feature_set="full"),
        param_distributions=search_spaces[model_name],
        n_iter=cfg.n_iter,            # 30
        scoring=mcc_scorer,
        cv=inner_folds[k],            # índices relativos ao treino do fold k
        refit=True,
        random_state=SEED,
        n_jobs=cfg.n_jobs,
    )
    search.fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = search.predict_proba(X.iloc[test_idx])[:, 1]
    save_fold(model_name, k, search.best_params_, search.best_score_, test_idx, proba)
```

- Salvar por fold: melhores hiperparâmetros, MCC interno, métricas externas e as probabilidades de cada linha de teste (predições fora do fold), em `results/nested/<modelo>/fold<k>.parquet`.
- Avaliar também o `DummyClassifier` nos folds externos, como piso.
- Implementar retomada: se o arquivo de um fold já existe, pular. A execução é longa.
- Hiperparâmetros diferentes entre folds são esperados. Registrar e discutir a estabilidade.

**Espaços de busca iniciais** (`configs/search_spaces.yaml`; ajustar se a busca bater nos limites):

| Algoritmo | Hiperparâmetros |
|---|---|
| Regressão logística | `C` ~ log-uniforme(1e-3, 1e2); `solver="lbfgs"`; `max_iter=2000` |
| Gaussian NB | `var_smoothing` ~ log-uniforme(1e-12, 1e-3) |
| kNN | `n_neighbors` ∈ {3, 5, …, 51}; `weights` ∈ {uniform, distance}; `p` ∈ {1, 2} |
| Árvore de decisão | `max_depth` ∈ {3, …, 30, None}; `min_samples_leaf` ∈ [1, 200]; `criterion` ∈ {gini, entropy} |
| MLP | `hidden_layer_sizes` ∈ {(32,), (64,), (64, 32), (128, 64)}; `alpha` ~ log-uniforme(1e-6, 1e-2); `learning_rate_init` ~ log-uniforme(1e-4, 1e-2); `early_stopping=True` |
| Random Forest | `n_estimators` ∈ [100, 500]; `max_depth` ∈ {10, 20, 30, None}; `max_features` ∈ {sqrt, log2, 0.5}; `min_samples_leaf` ∈ [1, 50] |
| LightGBM | `n_estimators` ∈ [100, 1000]; `learning_rate` ~ log-uniforme(0.01, 0.3); `num_leaves` ∈ [15, 255]; `min_child_samples` ∈ [10, 200]; `subsample` ∈ [0.6, 1] com `subsample_freq=1`; `colsample_bytree` ∈ [0.6, 1]; `reg_lambda` ~ log-uniforme(1e-3, 10) |

**Pronto quando:** os 3 finalistas e o Dummy têm resultados nos 5 folds externos.

**Diário:** `[ENTREGA]` com o tempo de execução e a variação dos hiperparâmetros entre folds.

### Etapa 8 — Avaliação e comparação

**Objetivo:** responder a Q1 e a Q2.

**Tarefas (`src/evaluation/`, `src/reporting/`):**

1. **Tabela principal:** média ± desvio nos 5 folds externos de MCC, F1 (ataque), F1 (benigno), acurácia balanceada, PR-AUC, TPR e FPR, para os finalistas e o Dummy.
2. **Gráfico por fold** (boxplot ou pontos ligados) do MCC de cada finalista.
3. **Teste estatístico:** *corrected resampled t-test* entre o melhor modelo e cada um dos outros, com correção de Holm. Com apenas 5 folds o teste tem pouco poder, e isso deve ser dito no artigo.

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
5. **Recall por tipo de ataque** (34 tipos e 7 categorias) e **FPR no benigno**, com intervalo de confiança de Wilson a 95%. Gráfico de barras horizontais ordenado do pior para o melhor tipo.
6. **Curvas PR e ROC** do melhor modelo, a partir das predições fora do fold.

**Pronto quando:** as tabelas e figuras são geradas por `make evaluate` e Q1 e Q2 estão respondidas no diário.

**Diário:** `[ENTREGA]` e `[DESCOBERTA]` para H1 e H2 (confirmadas ou não).

### Etapa 9 — Interpretação e análise de atalhos

**Objetivo:** responder a Q3 e cumprir o eixo de interpretabilidade, com o **melhor modelo apenas**.

**Princípio:** interpretar os modelos treinados em cada fold externo usando os respectivos dados de teste, que eles não viram. Isso também permite medir a estabilidade das explicações entre folds.

**Tarefas:**

1. **Importância por permutação** (métrica MCC, 10 repetições) em cada fold externo de teste. Reportar média ± desvio entre folds e a estabilidade do ranking (correlação de Spearman entre folds e sobreposição do top 10).
2. **SHAP:**
   - Se o melhor modelo for de árvores, usar `TreeExplainer` numa subamostra de cerca de 2.000 linhas de cada fold de teste. Gerar o gráfico de resumo (beeswarm) e gráficos de dependência para os atributos mais importantes, em especial `Duration` e `IAT`.
   - Se não for de árvores, ficar com permutação e gráficos de dependência parcial (PDP), e registrar o motivo.
3. **Explicações locais:** SHAP para 4 a 6 erros (falsos negativos dos tipos de ataque com pior recall e falsos positivos no benigno), discutidos no artigo.
4. **Testes de atalho:**
   - reavaliar os modelos de um atributo só (`Duration`, `IAT`) nos folds externos de M;
   - **ablação:** repetir a Etapa 7 só para o melhor algoritmo com `feature_set="no_shortcut"` (sem `Duration` e `IAT`), nos mesmos folds. Se houver tempo, remover cada um separadamente;
   - comparar MCC (teste corrigido) e recall por tipo de ataque entre `full` e `no_shortcut`. **Essa é a tabela central do artigo.**
5. **Leitura dos resultados:** queda pequena indica que o modelo é robusto a esses atributos; queda grande é evidência de atalho; colapso em tipos específicos de ataque indica quais ataques são reconhecidos pelo ambiente, e não pelo comportamento.

**Pronto quando:** Q3 está respondida no diário com números e figuras.

**Diário:** `[ENTREGA]` e `[DESCOBERTA]` para H3.

### Etapa 10 — Modelo final e reprodutibilidade

**Objetivo:** deixar tudo reproduzível do CSV bruto até as figuras do artigo.

**Tarefas:**

1. **Modelo final para disponibilização:** rodar a busca de hiperparâmetros (3 folds) em toda a amostra M com o melhor algoritmo, retreinar e salvar `models/final_model.joblib` com os hiperparâmetros, as versões, a semente, o hash dos dados e o commit. Esse modelo **não é avaliado de novo**: a estimativa de desempenho dele é a da CV aninhada. Isso deve ser dito no artigo.
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
2. **Slides:** problema; dataset e amostragem; protocolo de validação; resultados; recall por tipo; achado sobre atalhos; reprodutibilidade; limitações.
3. **Ensaio com cronômetro**, respeitando o tempo que a professora definir.
4. **Apresentação em 01/12** e participação ativa nas discussões dos outros grupos (nota individual).

---

## 8. Cronograma

Hoje é **25/10/2026 (domingo)**. Restam quatro semanas até a meta interna de entrega (20/11, sexta-feira), com dois dias de folga antes do prazo oficial (22/11, 23h59).

| Período | Etapa(s) | Marco (pronto quando) |
|---|---|---|
| 25–26/10 | 0 | Repositório, ambiente, `CLAUDE.md`, diário e registro de IA criados; download iniciado |
| 26–28/10 | 1 | Manifesto e contagens prontos; `[verificar]` resolvidos; decisão sobre agrupamento registrada |
| 28–29/10 | 2 | Amostras E e M geradas, com hashes estáveis entre execuções |
| 29–31/10 | 3 | EDA concluída; `features.yaml` preenchido; suspeitas de atalho confirmadas ou descartadas |
| 01–02/11 | 4 e 5 | Pipelines e folds congelados; testes de vazamento e de divisão passando |
| 03–04/11 | 6 | Spot-checking concluído; 3 finalistas escolhidos |
| 04–08/11 | 7 | CV aninhada concluída para os 3 finalistas e o Dummy |
| 08–10/11 | 8 | Tabelas, figuras e testes estatísticos; Q1 e Q2 respondidas |
| 10–13/11 | 9 | Interpretação e ablação; Q3 respondida |
| 13–15/11 | 10 | Modelo final salvo; `make all` funcionando |
| 14–19/11 | 11 | Artigo completo e revisado; reprodução cruzada concluída |
| **20/11** | Entrega | PDF e código no Moodle |
| 23–30/11 | 12 | Revisões por pares (24–30/11), slides e ensaio |
| **01/12** | 12 | Apresentação oral |

**Em paralelo:**

- **Revisão de literatura (01–07/11),** aproveitando o tempo em que a CV aninhada roda. Ver seção 8.1.
- **Rascunho de Métodos a partir de 02/11,** com base no diário.
- **Consolidação semanal no diário** (seção 6.4) em 31/10, 07/11, 14/11 e 20/11.

### 8.1 Revisão de literatura (o que procurar)

O objetivo é mostrar a lacuna que o trabalho preenche: a maior parte dos trabalhos reporta métricas altíssimas com divisão aleatória e sem investigar atalhos.

- O artigo do CICIoT2023 e os resultados de referência que ele reporta para 2 classes.
- 4 a 6 trabalhos que usam o CICIoT2023 em detecção binária. Para cada um, anotar: protocolo de divisão, métricas, tratamento de desbalanceamento e se investigam vazamento ou atalhos.
- Problemas metodológicos em datasets de detecção de intrusão: Engelen et al. (2021) sobre o CICIDS2017; Arp et al. (2022).
- Aprendizado por atalho: Geirhos et al. (2020).
- Vazamento e reprodutibilidade em ciência baseada em AM: Kapoor e Narayanan (2023).

Registrar cada leitura no diário com uma linha de resumo e a seção do artigo onde será citada.

### 8.2 Plano de contingência

Se o cronograma atrasar, cortar nesta ordem:

1. Ablações separadas de `Duration` e `IAT` (manter só a ablação conjunta).
2. Segunda repetição do laço externo (já é opcional).
3. Iterações da busca: de 30 para 15.
4. Tamanho da amostra M: de 200 mil para 100 mil linhas.
5. Explicações SHAP locais (manter as globais e a permutação).

**Nunca cortar:** a separação entre dados de decisão e de teste; MCC e recall por tipo de ataque; a análise de atalhos; pelo menos 5 algoritmos; manifesto, sementes, README e `make all`; o diário e o registro de IA.

---

## 9. O artigo

### 9.1 Estrutura e orçamento de páginas (limite de 20 páginas, incluindo referências e apêndices)

| Seção | Conteúdo | Páginas |
|---|---|---|
| Abstract | Problema, método, números principais, achado sobre atalhos | — |
| 1. Introduction | Motivação (IoT doméstica vulnerável), resultados quase perfeitos na literatura, Q1–Q3, contribuições | 1,5 |
| 2. Related Work | Trabalhos com o CICIoT2023; problemas metodológicos em datasets de intrusão; aprendizado por atalho | 1,5–2 |
| 3. Data | Dataset, inspeção, amostragem e deduplicação, principais achados da EDA | 2,5 |
| 4. Methods | Pré-processamento, algoritmos, protocolo de validação, métricas, otimização, teste estatístico, interpretação | 2,5 |
| 5. Results | Spot-checking, CV aninhada, testes estatísticos, recall por tipo de ataque | 3 |
| 6. Interpretation and Shortcut Analysis | Permutação, SHAP, modelos de um atributo, ablação | 2,5 |
| 7. Discussion and Limitations | Respostas a Q1–Q3; limitações da seção 3.7; ameaças à validade | 1,5 |
| 8. Conclusion | Síntese e trabalhos futuros | 0,5 |
| Reproducibility and AI Usage Statements | Repositório, tag, hashes, reprodução cruzada; ferramentas de IA usadas e onde | 0,5 |
| References | | 2 |

### 9.2 Figuras e tabelas previstas

**Figuras:**

1. Diagrama do protocolo: amostragem → E e M → spot-checking → CV aninhada → interpretação e ablação.
2. Distribuição de `Duration` e `IAT` por classe (EDA).
3. MCC por algoritmo no spot-checking (boxplot).
4. MCC por fold externo dos finalistas.
5. Recall por tipo de ataque, `full` × `no_shortcut` (figura central).
6. Resumo SHAP do melhor modelo.
7. (Opcional) Curva PR do melhor modelo.

**Tabelas:**

1. Composição das amostras E e M.
2. Atributos removidos e motivo.
3. Resultados da CV aninhada (média ± desvio).
4. Testes estatísticos.
5. Ablação (MCC e recall por categoria, `full` × `no_shortcut`).

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
| *Data snooping* (vazamento) | Amostras E e M disjuntas; pré-processamento ajustado dentro do `Pipeline`; folds congelados; testes automatizados |
| Correlações espúrias | Modelos de um atributo só e ablação de `Duration` e `IAT` (Etapa 9) |
| Seleção enviesada de hiperparâmetros | CV aninhada |
| Linha de base inadequada | `DummyClassifier` e modelos simples (regressão logística, Naive Bayes, árvore) |
| Métrica inadequada | MCC como principal, mais TPR, FPR e recall por tipo; nunca só acurácia |
| Falácia da taxa-base | Discussão explícita de que, numa casa real, o benigno domina e mesmo um FPR baixo gera muitos alarmes falsos |
| Avaliação só em laboratório | Registrada como limitação; generalização fica fora do escopo |

### 10.2 Checklist antes da entrega

- [ ] Problema, objetivo, perguntas de pesquisa e link do dataset descritos com clareza.
- [ ] EDA com gráficos e sumarização estatística.
- [ ] Pré-processamento justificado item a item.
- [ ] Pelo menos 5 algoritmos com vieses diversos, constantes do programa da disciplina.
- [ ] Métrica principal e métricas secundárias justificadas.
- [ ] Estratégia de divisão descrita; mesmas divisões para todos os algoritmos; sementes fixas.
- [ ] Nenhuma decisão usou os folds externos de teste.
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

Todas vieram de memória e devem ser **conferidas antes de citar [verificar]**.

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

**Interpretabilidade**

- Breiman, L. *Random Forests*. Machine Learning, 45, 5–32, 2001 (importância por permutação).
- Lundberg, S. M.; Lee, S.-I. *A Unified Approach to Interpreting Model Predictions*. NeurIPS, 2017.

**Reprodutibilidade**

- Pineau, J. et al. *Improving Reproducibility in Machine Learning Research*. JMLR, 22, 2021.
