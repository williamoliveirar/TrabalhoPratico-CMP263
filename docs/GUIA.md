# Guia prático do projeto (Etapas 0 a 4)

*Situação em 05/11/2026; código da Etapa 4 no commit `e16e460`. Atualize este guia ao fim de cada etapa.*

Este guia explica, do zero, o que já foi construído, como rodar cada parte e como conferir sozinho os resultados e as conclusões. O plano completo está em `docs/PLANO.md`; as decisões e os números, com fonte, em `docs/DIARIO.md`. Aqui o foco é prático: o que é cada coisa, que comando rodar e o que você deve ver. Os termos são os das aulas. O glossário completo está no PLANO, §1.4, e o mapa das etapas formais do treinamento de modelos no PLANO, §1.3.

Sumário:
1. [O projeto em uma página](#1-o-projeto-em-uma-página)
2. [Conceitos e termos que aparecem em todo lugar](#2-conceitos-e-termos-que-aparecem-em-todo-lugar)
3. [O ambiente: onde está cada coisa](#3-o-ambiente-onde-está-cada-coisa)
4. [Mapa do repositório](#4-mapa-do-repositório)
5. [A configuração](#5-a-configuração)
6. [O pipeline, etapa por etapa](#6-o-pipeline-etapa-por-etapa)
7. [Como validar tudo por conta própria](#7-como-validar-tudo-por-conta-própria)
8. [As conclusões até aqui e como chegar a elas](#8-as-conclusões-até-aqui-e-como-chegar-a-elas)
9. [Como trabalhamos com o Git](#9-como-trabalhamos-com-o-git)
10. [O que vem a seguir e o que está com a dupla](#10-o-que-vem-a-seguir-e-o-que-está-com-a-dupla)
11. [Problemas comuns](#11-problemas-comuns)

---

## 1. O projeto em uma página

**Tarefa:** classificação binária de cada instância do CICIoT2023 (uma janela de pacotes) como benigna (classe negativa, `y = 0`) ou ataque (classe positiva, `y = 1`), com uma estratégia de avaliação sem data leakage. O trabalho responde três perguntas:
- **Q1:** qual algoritmo tem o maior MCC, e as diferenças entre os melhores são significativas?
- **Q2:** quais tipos de ataque escapam do melhor modelo (sensibilidade por tipo), e qual a taxa de falsos positivos (TFP, o alarme falso) no tráfego benigno?
- **Q3:** o desempenho depende de atalhos, isto é, de atributos que identificam o laboratório ou o processamento, e não o ataque?

**O que já está pronto:**

| Etapa | Etapa formal (aulas) | O que faz | Situação |
|---|---|---|---|
| 0 | — | Repositório, ambiente, utilitários (semente, registro de execução) | Concluída |
| 1 | Coleta dos dados | Ler os 309 CSVs sem alterar nada, contar, inspecionar, conferir com o Kaggle, ler o artigo do dataset | Concluída |
| 2 | Amostragem; limpeza (duplicatas) | Sortear as amostras E (exploração) e M (modelagem), sem duplicatas e sem instâncias em comum | Concluída |
| 3 | Análise exploratória (EDA) | Só em E; decidir o pré-processamento (`configs/features.yaml`) | Concluída |
| 4 | Pré-processamento | Antes da divisão (regras fixas) e dentro do `Pipeline` (ajustado no treino de cada fold) | Concluída |
| 5 | Divisão em treino, validação e teste | Folds congelados para o spot-checking (E) e para a validação cruzada aninhada (M) | Próxima |

**O caminho dos dados**, do arquivo bruto até a decisão de pré-processamento:

```
309 CSVs oficiais (8,9 GB, fora do repositório)
   │  make ingest   (≈ 4 min)
   ▼
data/interim/ciciot2023_raw.parquet   46.776.697 linhas × 44 colunas, valores intactos
   │  make inspect  (≈ 20 s)  → contagens, esquema, nulos e infinitos (results/inspection/)
   │  make sample   (≈ 10 s)
   ▼
data/processed/sample_E.parquet   48.104 linhas  (exploração e spot-checking)
data/processed/sample_M.parquet  192.550 linhas  (otimização e avaliação; ninguém olha antes da Etapa 7)
   │  make eda      (≈ 10 s, só E)
   ▼
results/eda/  (tabelas e figuras)  →  configs/features.yaml  (D13: o que remover e transformar)
   │  Etapa 4 (código em src/models/, sem alvo próprio; conferido por make test)
   ▼
prepare()          antes da divisão: 27 atributos, Std ausente → 0 (linha a linha)
build_pipeline()   dentro de cada fold: log1p + padronização + one-hot ajustados só no treino, depois o modelo
```

---

## 2. Conceitos e termos que aparecem em todo lugar

**Os dados**
- **Instância = janela de pacotes.** O CICIoT2023 não traz um pacote por linha. Cada linha (instância) resume uma janela de pacotes consecutivos: médias, contagens, frações de protocolo. Segundo o artigo do dataset, a janela tem **10 pacotes** no benigno e em 14 tipos de ataque, e **100 pacotes** nos 19 tipos de DDoS, DoS e Mirai.
- **Atributos preditivos e atributo alvo.** Os atributos preditivos (x) são as 39 colunas estatísticas, que viram 27 depois da limpeza. O atributo alvo (y) é a classe: ataque (positiva) ou benigno (negativa).
- **Captura.** Cada CSV é uma gravação (`.pcap`) de um experimento. Há 309 capturas, uma pasta por rótulo.
- **Rótulo por captura.** Toda janela de uma captura de ataque recebe o rótulo do ataque, inclusive o tráfego de fundo que houver nela. Isso é um possível ruído de rótulo e vai para as limitações.
- **`label34`, `category8`, `y`:** o rótulo original (34 valores: benigno + 33 ataques), a categoria (8: Benign, DDoS, DoS, Mirai, Recon, Web, BruteForce, Spoofing) e a classe binária. As três **nunca** entram como atributo preditivo; servem para estratificar e para medir a sensibilidade por tipo de ataque.
- **`source_file` e `row_idx`:** de qual CSV e de qual linha dele veio cada janela. Permitem voltar ao dado bruto a qualquer momento.

**A divisão dos dados** (Aulas 04 e 08)
- **Amostras E e M.** E (exploração) é onde olhamos os dados e tomamos decisões: EDA e spot-checking. M (modelagem) só é usada para otimizar hiperparâmetros e estimar o desempenho, e os folds de teste de M nunca informam decisões. Se a EDA fosse feita em M, as decisões "veriam" os dados de avaliação, e isso é data leakage.
- **Conjuntos de treinamento, de validação e de teste.**
  - *treinamento:* onde o modelo e o pré-processamento são ajustados;
  - *validação:* onde se comparam configurações de hiperparâmetros (ou algoritmos, no spot-checking);
  - *teste:* onde se estima o desempenho final, uma vez só, com tudo já decidido. É o *holdout* de 3 vias das aulas.
- **Validação cruzada aninhada** (Etapa 7, em M). Troca cada partição do *holdout* de 3 vias por uma validação cruzada:
  - *laço externo* (5 folds): 4 folds de treino e 1 fold de **teste**, para a estimativa de desempenho;
  - *laço interno* (3 folds, dentro do treino externo): 2 folds de treino e 1 de **validação**, para o ajuste de hiperparâmetros.
- **Estratificar por `label34`.** Ao dividir em folds, sortear separadamente dentro de cada um dos 34 rótulos, para que cada fold tenha a mesma proporção de cada tipo de ataque. Isso garante também a proporção de `y`.

**O pré-processamento** (Aulas 02b, 13–14 e 15–16)
- **Antes da divisão:** operações que removem instâncias ou atributos, ou que aplicam uma regra fixa sem parâmetro estimado dos dados. Podem rodar no conjunto inteiro sem data leakage. Aqui:
  - remover duplicatas (D4);
  - remover atributos redundantes e quase constantes (D13);
  - trocar `Std` ausente por 0.
- **Dentro do `Pipeline`:** operações que estimam parâmetros dos dados. São ajustadas (`fit`) no conjunto de treinamento de cada fold e só aplicadas (`transform`) à validação e ao teste. Aqui: padronização (média e desvio) e codificação *one-hot* (categorias), precedidas de `log1p`.
- **Duplicatas (D4).** Instâncias com os mesmos 39 atributos:
  - *exatas* (mesmo `label34`): fica uma cópia;
  - *conflitantes* (`y` diferente): saem todas as cópias;
  - *de tipo ambíguo* (dois tipos de ataque): saem todas as cópias.
- **Data leakage.** Qualquer informação de validação ou de teste que chegue ao treinamento e torne a estimativa de desempenho otimista. O PLANO §10.2 lista os tipos (Kapoor e Narayanan, 2023) e como cada um é tratado.
- **Atalho.** Um atributo que separa as classes por um motivo que não é o comportamento do ataque. Na taxonomia de data leakage, é um "atributo ilegítimo" (L2). Exemplos:
  - o tamanho da janela, que é uma escolha do processamento;
  - o TTL, que reflete o sistema operacional das máquinas do laboratório.

  Um modelo que aprende atalhos parece ótimo aqui e falha fora do laboratório.

**As métricas** (Aula 09)
- **Matriz de confusão:** VP, VN, FP e FN.
  - *sensibilidade* (recall, TVP): a fração dos ataques detectados;
  - *especificidade*: a fração dos benignos aceitos;
  - *TFP* = 1 − especificidade: o alarme falso.
- **MCC (coeficiente de correlação de Matthews).** A métrica principal (D5). Vai de −1 a 1: 0 é acaso, 1 é perfeito. Usa os quatro quadrantes da matriz de confusão e, ao contrário da acurácia, não se deixa enganar por classes desbalanceadas. Não aparece nos slides; a justificativa é Chicco e Jurman (2020).

**A reprodutibilidade**
- **Determinismo.** Rodar o mesmo script duas vezes gera arquivos idênticos byte a byte. É o que permite conferir resultados comparando arquivos (seção 7).
- **`<script>.run.json`.** Cada script grava, junto dos resultados, um registro da execução: horário, commit do Git, se havia código não commitado (`git_dirty`), hash da configuração, versões das bibliotecas, tempo e SHA-256 de cada saída.

---

## 3. O ambiente: onde está cada coisa

| O quê | Onde (nesta máquina) | Observação |
|---|---|---|
| Repositório | `...\Trabalho prático - Aprendizado de Máquina\TrabalhoPratico-CPM263` | GitHub: `williamoliveirar/TrabalhoPratico-CPM263` (privado) |
| Dados brutos (CSVs) | `..\CICIoT2023-Dataset\CICIoT2023-Dataset-CSV\CSV` (ao lado do repositório) | Nunca vão para o Git |
| Espelho do Kaggle | `..\CICIoT2023-Dataset\Cleaned-CICIoT2023-Dataset Parquet ...\` | Só para conferência |
| Caminhos desta máquina | `configs/local.yaml` | Não versionado; modelo em `configs/local.example.yaml` |
| Ambiente Python | `C:\Users\will_\.venvs\cmp263` | Fora do repositório por causa do limite de 260 caracteres de caminho do Windows |
| Onde o Make procura o ambiente | `local.mk` (`VENV := C:/Users/will_/.venvs/cmp263`) | Não versionado |
| Python | 3.12.7 | |
| Bibliotecas | `requirements.txt` (versões fixas, 55 pacotes) | Lista curta em `requirements.in` |
| Make | GNU Make 4.4.1 (winget) | |
| Dados gerados | `data/interim/`, `data/processed/` | Não versionados; recriados pelo pipeline |

**Abrindo o projeto no VS Code:**
1. *File → Open Folder* → pasta `TrabalhoPratico-CPM263`.
2. `Ctrl+Shift+P` → *Python: Select Interpreter* → escolha `C:\Users\will_\.venvs\cmp263\Scripts\python.exe`.
3. *Terminal → New Terminal* (PowerShell). Todos os comandos deste guia rodam a partir da raiz do repositório.

**Recriando ou conferindo o ambiente:**

```powershell
make env
```

Cria o ambiente (se não existir), atualiza o pip e instala as versões de `requirements.txt`. Se o ambiente já existe, só confirma que está tudo instalado ("Requirement already satisfied"). Em outra máquina, crie antes um `local.mk` com o caminho curto do ambiente (ver o README).

**Rodando sem o Make.** Cada alvo do `Makefile` só chama um comando Python, e o `make` imprime esse comando antes de rodá-lo. Sem o Make, rode o mesmo comando direto, por exemplo:

```powershell
& "C:/Users/will_/.venvs/cmp263/Scripts/python.exe" -m src.data.make_sample
```

**Abrindo o Python do projeto (para os testes da seção 7):**

```powershell
& "C:/Users/will_/.venvs/cmp263/Scripts/python.exe"
```

Abre o interpretador interativo (`>>>`). Rode-o na raiz do repositório para que `import src...` funcione. Para sair: `exit()`.

---

## 4. Mapa do repositório

```
TrabalhoPratico-CPM263/
├── CLAUDE.md                 regras permanentes do projeto (lidas pelo Claude Code)
├── README.md                 visão geral em inglês (para quem for reproduzir)
├── Makefile                  um alvo por etapa: make ingest, make sample, make eda, make test...
├── requirements.in / .txt    dependências (lista curta / versões fixas)
├── configs/
│   ├── base.yaml             semente, caminhos, tamanhos das amostras, parâmetros de cada etapa
│   ├── features.yaml         D13: colunas removidas, transformações e conjuntos de atributos
│   ├── search_spaces.yaml    espaços de busca dos hiperparâmetros (Etapa 7)
│   └── local.example.yaml    modelo do local.yaml (caminhos de cada máquina)
├── src/
│   ├── utils/
│   │   ├── config.py         carrega base.yaml + local.yaml; path() resolve caminhos
│   │   ├── seed.py           gerador aleatório único, a partir da semente da config
│   │   ├── run_metadata.py   grava o <script>.run.json de cada execução
│   │   └── io.py             JSON com quebra de linha LF, SHA-256 de arquivos, caminho relativo
│   ├── data/
│   │   ├── labels.py         pasta → label34 → category8 → y; tamanho da janela de cada rótulo
│   │   ├── ingest.py         Etapa 1: CSVs → Parquet sem alterar valores; manifesto com hashes
│   │   ├── inspect_raw.py    Etapa 1: contagens, esquema, nulos, infinitos (estatísticas globais)
│   │   ├── check_kaggle_mirror.py   compara o espelho do Kaggle com os CSVs (amostra de linhas)
│   │   ├── check_ingest.py   compara a nossa ingestão com o Kaggle, linha a linha
│   │   ├── make_sample.py    Etapa 2: amostras E e M
│   │   └── eda.py            Etapa 3: tabelas e figuras da EDA (só E)
│   ├── models/
│   │   ├── registry.py       Etapa 4: os 7 algoritmos de D7 + Dummy, com hiperparâmetros padrão
│   │   └── pipelines.py      Etapa 4: prepare() antes da divisão; build_pipeline() = pré-processamento + modelo
│   ├── reporting/style.py    estilo e cores das figuras (benigno azul, ataque laranja)
│   └── features/, experiments/, evaluation/, interpretation/   (vazios; Etapas 5 a 10)
├── tests/                    test_ingest.py, test_sample.py, test_eda.py, test_no_leakage.py (19 testes)
├── data/
│   ├── raw_manifest.csv      versionado: tamanho, SHA-256 e linhas de cada um dos 309 CSVs
│   ├── sample_index.csv.gz   versionado: quais linhas (source_file, row_idx) estão em E e em M
│   └── interim/, processed/  não versionados: Parquet da ingestão e amostras
├── results/
│   ├── inspection/           saídas da Etapa 1
│   ├── sampling/             saídas da Etapa 2
│   └── eda/                  saídas da Etapa 3 (tabelas CSV, JSON e figuras PNG)
└── docs/
    ├── PLANO.md              o plano do trabalho (pt-BR)
    ├── DIARIO.md             diário de bordo: decisões e números com fonte (pt-BR)
    ├── IA_LOG.md             registro do uso de IA (exigência do enunciado)
    ├── GUIA.md               este guia
    └── artigo/               template SBC do artigo
```

---

## 5. A configuração

Nenhum número "mágico" fica no código: semente, caminhos e tamanhos vêm de `configs/base.yaml`. As seções principais:

| Seção | Para que serve | Valores atuais |
|---|---|---|
| `seed` | semente única de todo sorteio | 42 |
| `paths` | onde ler e gravar cada arquivo | `local.yaml` sobrescreve os caminhos dos dados brutos |
| `columns` | nomes das colunas auxiliares | `label34`, `category8`, `y`, `source_file`, `row_idx` |
| `sampling` (D3) | tamanhos de E e M, folga para duplicatas, regra dos tipos raros | E: 25.000 + 25.000; M: 100.000 + 100.000; folga 1,3; raros 20% E / 80% M |
| `eda` | limiares da EDA | quase constante > 99,9%; correlação > 0,99; assimetria > 2; árvore de profundidade 2; 5 folds |
| `validation` (D8) | número de folds da divisão em treino, validação e teste | spot-checking 5 × 2 (validação cruzada repetida); laço externo 5 (teste); laço interno 3 (validação) |

`configs/features.yaml` (D13, decidido em 26/10) diz o que acontece com cada coluna antes dos modelos:
- `drop`: 12 colunas removidas, cada uma com o motivo e a entrada do diário. Ficam 27 atributos (limpeza, antes da divisão);
- `fill_null`: `Std` ausente vira 0 (regra fixa, antes da divisão);
- `categorical`: `Protocol Type` vira *one-hot* nos algoritmos sensíveis à escala (dentro do `Pipeline`);
- `log1p`: `Min`, `Max`, `Std` e `IAT` (tratamento de outliers de cauda longa, dentro do `Pipeline`);
- `feature_sets`: `full`, `no_window`, `no_ttl` e `no_shortcut`, para a análise de atalhos.

`configs/search_spaces.yaml` guarda, por algoritmo, os parâmetros fixos (`fixed`, usados também no spot-checking) e os espaços de busca dos hiperparâmetros (`params`, Etapa 7).

Se você mudar um valor da configuração, o hash registrado no `run.json` muda. Assim dá para saber com qual configuração cada resultado foi gerado. Toda mudança de configuração vai para o diário.

---

## 6. O pipeline, etapa por etapa

### Etapa 0 — Ambiente e utilitários

O que existe e por quê:
- `make env` instala tudo com versões fixas: qualquer pessoa obtém o mesmo ambiente;
- `seed.py`: todo sorteio usa um gerador criado a partir da semente 42;
- `run_metadata.py`: cada script grava seu `run.json` (ver a seção 2);
- `.gitattributes` força quebra de linha LF nos arquivos de texto, para que os hashes batam entre Windows e Linux.

### Etapa 1 — Ler e inspecionar os dados brutos

**`make ingest`** (≈ 4 min) — `src/data/ingest.py`
- Lê os 309 CSVs em ordem alfabética e calcula o SHA-256 de cada um.
- Grava tudo num único Parquet (formato colunar, compacto, rápido de ler), **sem alterar nenhum valor**: `inf` continua `inf`, célula vazia continua nula.
- Acrescenta `label34`, `category8`, `y` (a partir da pasta), `source_file` e `row_idx`.
- Descarta só linhas com número errado de campos e registra cada uma. Foram 3: a última linha de `DoS-UDP_Flood7`, `8` e `9`, cortadas no meio.
- Saídas: `data/interim/ciciot2023_raw.parquet`, `data/raw_manifest.csv`, `results/inspection/malformed_rows.csv`.
- Por que não usamos o leitor do polars: ele completa linhas curtas com nulos **sem avisar**, e as 3 linhas truncadas entrariam no dataset como lixo.

**`make inspect`** (≈ 20 s) — `src/data/inspect_raw.py`
- Contagens por arquivo e no total (`label_counts_by_file.csv`, `label_counts_total.csv`) e esquema com nulos, infinitos, mínimo e máximo de cada coluna (`schema.json`).
- Só estatísticas **globais**. Estatística de atributo por classe é EDA e fica para E.

**`make check-ingest`** (≈ 1,5 min, opcional) — compara a nossa ingestão com o Parquet do Kaggle, linha a linha. Resultado em `results/inspection/ingest_vs_kaggle.json`: 0 diferenças não explicadas.

**Leitura do artigo do dataset** (sem código): confirmou as ferramentas, os 105 dispositivos, que `Duration` é o TTL e que o rótulo vale para a captura inteira. Revelou também os tamanhos de janela (10 e 100) por grupo de ataque.

### Etapa 2 — Amostras E e M

**`make sample`** (≈ 10 s) — `src/data/make_sample.py`, passo a passo:
1. **Alvo por tipo:** 25.000 ÷ 33 = 757 linhas por tipo de ataque em E, e 100.000 ÷ 33 = 3.030 em M.
2. **Candidatos:** para cada rótulo, sorteia 1,3 × o alvo, para ter folga quando as duplicatas saírem.
3. **Duplicatas (D4):** removidas como na seção 2. Saíram 1.988 cópias exatas, 651 conflitantes e 2.701 de tipo ambíguo.
4. **Corte:** embaralha cada tipo e corta em E e M. Os 3 tipos raros (Backdoor_Malware, Recon-PingSweep, Uploading_Attack) entram com todas as linhas, 20% em E e 80% em M. O benigno é cortado para empatar com o total de ataque (50/50).

Saídas:
- `data/processed/sample_E.parquet` (48.104 linhas) e `sample_M.parquet` (192.550);
- `data/sample_index.csv.gz` (versionado);
- `results/sampling/sampling_report.json` e `sample_counts.csv`, com a conta de cada rótulo.

### Etapa 3 — EDA (só em E)

**`make eda`** (≈ 10 s) — `src/data/eda.py`. Nunca abre M. Tabelas em `results/eda/`:

| Arquivo | O que responde |
|---|---|
| `composition.csv` | quantas linhas de cada classe, categoria e tipo há em E |
| `feature_summary.csv` | por atributo: nulos, infinitos, faixa, zeros, valores distintos, moda (quase constantes), assimetria (cauda pesada), outliers |
| `correlated_pairs.csv` | colunas idênticas e pares com correlação acima de 0,99 |
| `structural_checks.json` | identidades exatas entre colunas (contagem = fração × `Number` etc.) |
| `number_by_label.csv` | `Number` em cada tipo, comparado ao tamanho de janela do artigo |
| `univariate.csv` | quanto cada atributo sozinho separa benigno de ataque (AUC, KS), no geral e por grupo de janela |
| `single_feature_models.csv` | MCC de uma árvore de profundidade 2 com um atributo só (5 folds em E) |
| `by_type_summary.csv` | mediana e quartis de TTL, `IAT` e `Number` por tipo |

Figuras:

| Figura | O que mostra |
|---|---|
| `fig_number_by_type.png` | `Number` separa os dois grupos de janela |
| `fig_single_feature_mcc.png` | ranking dos atributos sozinhos |
| `fig_distributions.png` | distribuições por classe |
| `fig_spearman.png` | correlações entre atributos |
| `fig_by_type.png` | TTL, `IAT` e `Number` por tipo |

### Etapa 4 — Pré-processamento dos dados

Não tem alvo no Makefile: é código usado pelas etapas seguintes (`src/models/`), conferido por `make test`. As decisões vêm do `features.yaml` (D13); a tabela completa, tarefa por tarefa das aulas, está no PLANO, Etapa 4. Em resumo:

| Tarefa das aulas | O que fazemos | Quando |
|---|---|---|
| Limpeza: duplicatas | Regras da D4 (seção 2) | Antes da divisão (Etapa 2) |
| Limpeza: atributos redundantes e quase constantes | 12 colunas removidas, sem usar o atributo alvo | Antes da divisão |
| Limpeza: valores ausentes | `Std` ausente → 0 (janelas de 1 pacote); `Rate` com `inf` sai junto com a coluna | Antes da divisão |
| Tratamento de outliers | Sem remoção (ataque é atípico por natureza); `log1p` nos 4 atributos de cauda longa | Dentro do `Pipeline` |
| Normalização ou padronização | Padronização (`StandardScaler`) em LR, kNN, MLP e Naive Bayes | Dentro do `Pipeline`, ajustada no treino |
| Codificação | *One-hot* de `Protocol Type` nos mesmos 4 algoritmos | Dentro do `Pipeline`, ajustada no treino |
| Discretização | Não aplicada | — |
| Balanceamento de classes | Feito pelo desenho da amostra (D3: 50/50) | Antes da divisão (Etapa 2) |
| Redução de dimensionalidade | Não aplicada (27 atributos; PCA impediria a análise de atalhos) | — |

Como usar (é o que as Etapas 6 a 10 vão fazer):

```python
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
features = feature_names(E.columns, feature_cfg, "full", auxiliary_columns(cfg))   # 27 atributos
X = prepare(E, features, feature_cfg)        # antes da divisão: linha a linha, sem parâmetro estimado
pipe = build_pipeline("logreg", features, feature_cfg, search_cfg, seed, n_jobs)
pipe.fit(X.iloc[treino], y[treino])          # padronização e one-hot ajustados só no treino
pipe.predict_proba(X.iloc[teste])            # aplicados, sem novo ajuste, ao teste
```

Árvores, Random Forest, LightGBM e o Dummy recebem os atributos como estão. Dentro de uma validação cruzada (`cross_validate`, `RandomizedSearchCV`), o `Pipeline` é clonado e reajustado em cada fold. Por isso não há data leakage do pré-processamento.

---

## 7. Como validar tudo por conta própria

São cinco níveis, do mais rápido ao mais completo. Rode no terminal do VS Code, na raiz do repositório.

### Nível 1 — Os testes passam (≈ 10 s)

```powershell
make test
```

Esperado: `19 passed`. Os testes conferem, com dados pequenos inventados:
- que a ingestão preserva valores e descarta só linhas malformadas;
- que o mapeamento de rótulos tem 34 rótulos e 12 ataques DDoS;
- que a deduplicação segue as três regras e que E e M saem disjuntas e 50/50;
- que o sorteio é determinístico e que os grupos de janela batem com o artigo;
- (`test_no_leakage.py`, Etapa 4) que:
  - o pré-processamento antes da divisão é linha a linha e deixa 27 atributos;
  - as médias e os desvios da padronização e as categorias do *one-hot* vêm só das instâncias de treino;
  - mudar as instâncias de teste não muda nada do que foi ajustado;
  - a validação cruzada reajusta o pré-processamento em cada fold;
  - os 8 algoritmos (7 + Dummy) treinam e predizem.

### Nível 2 — Refazer as Etapas 2 e 3 e obter os mesmos arquivos (≈ 30 s)

```powershell
make sample
```

```powershell
make eda
```

```powershell
git status --short
```

Esperado: só aparecem os dois registros de execução (`results/eda/eda.run.json` e `results/sampling/make_sample.run.json`), que mudam porque guardam o horário. Para confirmar que **nenhum resultado** mudou:

```powershell
git diff --stat -- results data ":(exclude)*.run.json"
```

Esperado: nenhuma saída. Isso prova que as amostras, as tabelas e as figuras que você acabou de gerar são idênticas, byte a byte, às versionadas. Depois, descarte os registros de execução da sua rodada:

```powershell
git restore -- "*.run.json"
```

### Nível 3 — Refazer desde os CSVs brutos (≈ 6 min)

```powershell
make ingest
```

```powershell
make inspect
```

Depois, repita o `git status` e o `git diff` do nível 2. Esperado: só os `.run.json` mudam. O `data/raw_manifest.csv` não mudar prova que os seus CSVs são exatamente os mesmos (mesmo SHA-256 nos 309 arquivos). Se quiser, rode também `make sample` e `make eda` em seguida, e `make check-ingest` para a conferência com o Kaggle. O `make ingest` regrava o Parquet de 800 MB em `data/interim/`, o que é normal.

### Nível 4 — Reproduzir as conclusões com as próprias mãos

Abra o Python do projeto (seção 3) e cole os blocos abaixo, um de cada vez. Eles não usam os scripts do pipeline, só as amostras: é uma verificação independente.

**4.1 — As amostras têm o tamanho certo, são 50/50 e não têm linhas em comum:**

```python
import polars as pl
E = pl.read_parquet("data/processed/sample_E.parquet")
M = pl.read_parquet("data/processed/sample_M.parquet")
print(E.shape, M.shape)                       # (48104, 44) (192550, 44)
print(E["y"].value_counts().sort("y"))        # 24052 benign, 24052 attack
print(E.join(M, on=["source_file", "row_idx"], how="inner").height)   # 0
```

**4.2 — `Number` é o tamanho da janela do artigo:**

```python
from src.data.labels import window_of
E = E.with_columns(pl.col("label34").map_elements(window_of, return_dtype=pl.Int64).alias("window"))
print(E.group_by("window", "y").agg(pl.len().alias("rows"),
      (pl.col("Number") == pl.col("window")).mean().alias("share_Number_eq_window")).sort("window", "y"))
```

Esperado: nos três grupos (benigno com janela 10, ataques com janela 10, ataques com janela 100), mais de 99,6% das linhas têm `Number` igual à janela.

**4.3 — Só com `Number`, uma árvore já chega a MCC 0,65:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import matthews_corrcoef
X, y = E.select("Number").to_numpy(), E["y"].to_numpy()
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, E["label34"])
pred = cross_val_predict(DecisionTreeClassifier(max_depth=2, random_state=42), X, y, cv=folds)
print(round(matthews_corrcoef(y, pred), 3))   # 0.653
print(E.with_columns(pl.Series("pred", pred)).group_by("window", "y")
      .agg(pl.col("pred").mean().alias("share_predicted_attack")).sort("window", "y"))
```

Esperado:
- benigno: 0,0 (nenhum falso positivo);
- ataques de janela 10: 0,0 (nenhum detectado);
- ataques de janela 100: 0,9997.

A árvore aprendeu "janela grande = ataque", e só isso.

**4.4 — As colunas removidas são cópias exatas de outras:**

```python
print((E["ack_count"] - E["ack_flag_number"] * E["Number"]).abs().max())   # ~1e-14: count = fraction × Number
print((E["Tot sum"] - E["AVG"] * E["Number"]).abs().max())                 # ~1e-11 (rounding on values up to 236 537)
print((E["Variance"] - E["Std"] ** 2).abs().max())                         # ~1e-8 (rounding on values up to 6.8e7)
print((E["ARP"] + E["IPv"] - 1).abs().max())                               # ~2e-16: ARP = 1 - IPv
print((E["IPv"] == E["LLC"]).all(), (E["Tot size"] == E["AVG"]).all())     # True True
```

Os resíduos são só arredondamento de ponto flutuante: em termos relativos, ficam abaixo de 1e-14.

**4.5 — TTL: fixo nas inundações, sobreposto nos demais:**

```python
print(E.group_by("window", "y").agg(pl.col("Time_To_Live").median().alias("median"),
      pl.col("Time_To_Live").quantile(0.25).alias("q25"),
      pl.col("Time_To_Live").quantile(0.75).alias("q75")).sort("window", "y"))
```

Esperado:

| Grupo | Mediana (quartis) |
|---|---|
| Benigno | 99,0 (73,7 a 147,5) |
| Ataques de janela 10 | 87,6 (64,0 a 116,7) |
| Ataques de janela 100 | 64,0 (64,0 a 64,6) |

**4.6 — Ler as tabelas de resultado:**

```python
print(pl.read_csv("results/eda/single_feature_models.csv").head(8))
print(pl.read_csv("results/inspection/label_counts_total.csv").filter(pl.col("level") == "category8"))
```

Dica: abra os CSVs no VS Code ou com o polars. O Excel em português pode ler `0.653` como 653, porque espera vírgula decimal.

**4.7 — O pré-processamento só aprende com o conjunto de treinamento (Etapa 4):**

```python
import numpy as np
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.utils.config import load_config
cfg, fcfg, scfg = load_config(), load_config("features"), load_config("search_spaces")
E = pl.read_parquet("data/processed/sample_E.parquet")
feats = feature_names(E.columns, fcfg, "full", auxiliary_columns(cfg))
X, y = prepare(E, feats, fcfg), E["y"].to_numpy()
print(len(feats), int(X.isna().sum().sum()))                        # 27 0
ordem = np.random.default_rng(42).permutation(len(X))
treino, teste = ordem[:38000], ordem[38000:]
pipe = build_pipeline("logreg", feats, fcfg, scfg, seed=42, n_jobs=1).fit(X.iloc[treino], y[treino])
padr = pipe.named_steps["preprocess"].named_transformers_["standard"]
cols = list(padr.feature_names_in_)
print(abs(padr.mean_ - X.iloc[treino][cols].mean().to_numpy()).max())   # 0.0: médias do treino
print(abs(padr.mean_ - X[cols].mean().to_numpy()).max() > 0)            # True: não são as de E inteira
print(pipe.named_steps["preprocess"].transform(X.iloc[teste]).shape)    # (10104, 31)
```

Esperado:
- `27 0`: 27 atributos, nenhum valor ausente;
- `0.0`: as médias da padronização são exatamente as do conjunto de treinamento;
- `True`: elas não são as de E inteira;
- `(10104, 31)`: o teste recebe 31 colunas, isto é, 26 numéricas mais 5 colunas *one-hot* de `Protocol Type`.

### Nível 5 — Olhar as figuras e comparar com o diário

Abra as figuras de `results/eda/` (clique no explorador do VS Code, ou `start results\eda\fig_number_by_type.png`). Para cada afirmação da entrada "Etapa 3 — EDA na amostra E" do diário, o arquivo citado deve mostrar o número citado. Por exemplo, a afirmação "MCC 0,653 com `Number`" está na primeira linha de `single_feature_models.csv` e na primeira barra de `fig_single_feature_mcc.png`.

### Como ler um `run.json`

Abra, por exemplo, `results/eda/eda.run.json`:
- `git_commit`: o commit do código que gerou os resultados. `git show <commit> --stat` mostra o que havia nele;
- `git_dirty`: `false` significa que não havia código alterado e não commitado. Resultado com `true` não serve de fonte;
- `config_sha256`: muda se a configuração mudar;
- `outputs`: SHA-256 de cada arquivo gerado. Compare com `Get-FileHash <arquivo>` no PowerShell.

---

## 8. As conclusões até aqui e como chegar a elas

Cada conclusão abaixo tem a evidência, onde ela está e como conferir.

| # | Conclusão | Evidência | Como conferir |
|---|---|---|---|
| 1 | Usamos os CSVs oficiais por captura: 309 arquivos, 39 colunas, sem coluna de rótulo (o rótulo vem da pasta). O plano descrevia outra versão, com 46 atributos (D15) | `results/inspection/schema.json`, `kaggle_vs_official.json` | Nível 3; abra `schema.json` |
| 2 | A ingestão é fiel: mesmas 46.776.697 linhas do Kaggle, e toda diferença está em `inf`/vazio que o Kaggle limpou | `results/inspection/ingest_vs_kaggle.json` | `make check-ingest` |
| 3 | Agrupar os folds por captura é inviável (15 dos 33 tipos têm uma captura só). A divisão é por linha, estratificada por `label34` (D8) | `label_counts_total.csv`, coluna `n_files` | Conte os tipos com `n_files` = 1 |
| 4 | O tamanho da janela é um atalho de construção: `Number` sozinho dá MCC 0,653 e separa os 19 tipos de inundação do benigno sem falso positivo (D12) | `number_by_label.csv`, `single_feature_models.csv`, artigo §4 | Níveis 4.2 e 4.3 |
| 5 | Contagens, `Tot sum`, `Variance`, `Tot size`, `IPv` e `LLC` são cópias exatas de outras colunas; `Rate` quase espelha `IAT`; `Telnet` e `SMTP` são quase constantes. Ficam 27 atributos (D13) | `structural_checks.json`, `correlated_pairs.csv`, `feature_summary.csv` | Nível 4.4 |
| 6 | TTL é atalho só em parte: vale 64 nas inundações, mas se sobrepõe ao benigno nos demais tipos. `IAT` não é marcador temporal. `HTTPS` é forte, mas sem evidência de artefato | `by_type_summary.csv`, `univariate.csv`, `fig_by_type.png` | Nível 4.5 |
| 7 | O pré-processamento não tem data leakage: o que é feito antes da divisão é linha a linha, e o que estima parâmetros é ajustado só no treino de cada fold | `tests/test_no_leakage.py` | Níveis 1 e 4.7 |

**Consequência para o trabalho:** um modelo pode ter MCC alto aprendendo só "janela grande = ataque". Por isso:
- a Etapa 9 compara `full` com `no_window`, `no_ttl` e `no_shortcut`;
- todas as métricas são reportadas separadamente para os ataques de janela 10, em que benigno e ataque têm a mesma janela. Essa é a comparação mais honesta.

---

## 9. Como trabalhamos com o Git

A ordem é sempre esta, para que cada número tenha um commit limpo como origem:
1. **Commit do código** (e da configuração).
2. **Rodar o pipeline.** O `run.json` registra esse commit com `git_dirty: false`.
3. **Commit dos resultados** (`results/`, `data/raw_manifest.csv`, `data/sample_index.csv.gz`).
4. **Commit do diário**, citando o commit dos resultados.

Comandos úteis:

```powershell
git log --oneline -15
```

```powershell
git show c3b2bf2 --stat
```

```powershell
git log --oneline -- results/eda
```

O primeiro lista os últimos commits; o segundo mostra o que entrou num commit (aqui, os resultados da EDA); o terceiro lista os commits que mudaram os resultados da EDA.

Regras que valem sempre (estão no `CLAUDE.md`):
- o diário só recebe entradas novas; só o quadro "Decisões vigentes" é editável;
- resultado registrado não é sobrescrito sem perguntar;
- todo número no diário ou no artigo cita o arquivo e o commit de origem.

---

## 10. O que vem a seguir e o que está com a dupla

**Próximas etapas** (cronograma do PLANO §8; em 05/11 estamos cerca de 3 dias atrasados):

| Etapa | Previsto no plano | O que será feito |
|---|---|---|
| 5 | 01–02/11 | `make splits`: divisão em treino, validação e teste; folds congelados e versionados em `data/splits/` |
| 6 | 03–04/11 | Spot-checking em E: 7 algoritmos + Dummy; 3 finalistas |
| 7 | 04–08/11 | Validação cruzada aninhada em M para os finalistas: ajuste de hiperparâmetros (laço interno) e estimativa de desempenho (laço externo) |

**Pendências da dupla:**
- responder às propostas da entrada "2026-11-05 — Revisão metodológica à luz das aulas" do diário (D16 sobre a distribuição dos folds de teste, SVM na D7 e as demais);
- preencher, no `docs/IA_LOG.md`, os campos "O que foi aproveitado" e "O que foi rejeitado" de cada entrada, e registrar as conversas de planejamento no claude.ai;
- baixar a pasta *Supplementary material* do dataset (o formulário pede nome, e-mail e instituição) para confirmar se as linhas estão em ordem cronológica. Isso só pesa na Etapa 5, para a análise complementar de blocos contíguos;
- escolher a licença do repositório e pôr o nome do(a) colega no README.

---

## 11. Problemas comuns

| Sintoma | Causa e solução |
|---|---|
| `make: command not found` / `make : O termo 'make' não é reconhecido` | O terminal foi aberto antes da instalação do Make. Feche e abra outro terminal |
| `pip install` falha com caminho longo | Limite de 260 caracteres do Windows. Use um ambiente num caminho curto (`local.mk`) |
| `... not found; run make ingest first` | O Parquet da ingestão não existe nesta máquina (não é versionado). Rode `make ingest` |
| `ModuleNotFoundError: No module named 'src'` | O Python foi aberto fora da raiz do repositório. Use `cd` até `TrabalhoPratico-CPM263` |
| `git status` mostra resultados modificados além dos `.run.json` | Algo mudou (dados, código, configuração ou versão de biblioteca). **Não commite.** Veja o que mudou com `git diff` e registre no diário antes de qualquer coisa |
| Números estranhos ao abrir CSV no Excel | Ponto decimal lido como separador de milhar. Abra no VS Code ou no polars |
| O Kaggle não está nesta máquina | `make check-mirror` e `make check-ingest` são opcionais; o resto do pipeline não depende deles |
