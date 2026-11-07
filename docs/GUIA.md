# Guia prático do projeto (Etapas 0 a 7)

*Situação em 07/11/2026; código da Etapa 7 nos commits `ae91fea` e `b583e03` e resultados da validação cruzada aninhada no commit `6da18f6`. Atualize este guia ao fim de cada etapa.*

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
| 5 | Divisão em treino, validação e teste | Folds congelados para o spot-checking (E) e para a validação cruzada aninhada (M) | Concluída |
| 6 | Treinamento e spot-checking | 8 algoritmos + Dummy com hiperparâmetros padrão em E; 3 finalistas | Concluída |
| 7 | Otimização de hiperparâmetros e seleção de modelos | Validação cruzada aninhada dos finalistas em M | Próxima |

**O caminho dos dados**, do arquivo bruto até a escolha dos finalistas:

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
   │  make splits   (≈ 15 s, E e M)
   ▼
data/splits/   folds de treino, validação e teste, gerados uma vez e versionados (Etapa 5)
   │  make spotcheck (≈ 15 min, só E)
   ▼
results/spotcheck/   ranking dos 8 algoritmos  →  finalistas (configs/base.yaml, search.finalist_models)
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
- **Folds congelados.** Todas as divisões são geradas uma vez (`make splits`, Etapa 5), antes de qualquer treino, e versionadas em `data/splits/`. Assim, todos os algoritmos usam exatamente os mesmos conjuntos de treinamento, de validação e de teste em cada iteração, como pede o enunciado.
- ***Leave-capture-out*** (análise complementar, proposta na Etapa 5). Divide por captura, e não por instância, os 16 tipos de ataque com pelo menos 5 capturas. Assim, o modelo é testado em capturas que não viu no treino, e a comparação com a divisão principal mede o efeito de janelas quase duplicadas entre treino e teste.

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
- **AUC-ROC e AUPRC.** Medem se o modelo ordena os ataques acima dos benignos, sem depender do limiar. A AUC-ROC vale 0,5 no acaso e 1 na ordenação perfeita, e não depende da prevalência. O escore é a probabilidade de ataque ou, no SVM, a distância à margem (`decision_function`).
- **Overfitting e underfitting** (Aula 04; P4). Compare o MCC no treino com o da validação. Treino muito acima da validação é overfitting (o modelo memorizou). Os dois baixos e próximos é underfitting (o modelo é simples demais).
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
│   │   ├── registry.py       Etapas 4 e 6: os 8 algoritmos de D7 (com o SVM) + Dummy, com hiperparâmetros padrão
│   │   └── pipelines.py      Etapa 4: prepare() antes da divisão; build_pipeline() = pré-processamento + modelo
│   ├── reporting/style.py    estilo e cores das figuras (benigno azul, ataque laranja)
│   ├── experiments/
│   │   ├── make_splits.py    Etapa 5: folds congelados; funções que os entregam ao scikit-learn (cv=)
│   │   ├── spotcheck.py      Etapa 6: spot-checking em E; ranking e grupo de empate
│   │   └── nested_cv.py      Etapa 7: validação cruzada aninhada em M (busca no laço interno, teste no externo)
│   ├── evaluation/metrics.py métricas de D5 e D6 (MCC, sensibilidade, especificidade, AUC-ROC...)
│   └── features/, interpretation/   (vazios; Etapas 8 a 10)
├── tests/                    test_ingest.py, test_sample.py, test_eda.py, test_no_leakage.py, test_splits.py, test_spotcheck.py, test_nested_cv.py (37 testes); synthetic.py (dados inventados)
├── data/
│   ├── raw_manifest.csv      versionado: tamanho, SHA-256 e linhas de cada um dos 309 CSVs
│   ├── sample_index.csv.gz   versionado: quais linhas (source_file, row_idx) estão em E e em M
│   ├── splits/               versionado: folds de treino, validação e teste de E e de M (Etapa 5)
│   └── interim/, processed/  não versionados: Parquet da ingestão e amostras
├── results/
│   ├── inspection/           saídas da Etapa 1
│   ├── sampling/             saídas da Etapa 2
│   ├── eda/                  saídas da Etapa 3 (tabelas CSV, JSON e figuras PNG)
│   ├── splits/               saídas da Etapa 5 (tamanho de cada partição, conferências, hashes)
│   ├── spotcheck/            saídas da Etapa 6 (métricas por fold, resumo, ranking, figura)
│   └── nested/               saídas da Etapa 7 (por modelo e fold: configuração, busca, predições; folds.csv)
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
| `validation` (D8) | número de folds da divisão em treino, validação e teste | spot-checking 5 × 2 (validação cruzada repetida); laço externo 5 (teste); laço interno 3 (validação); a divisão por captura usa os 5 folds do laço externo |
| `spotcheck` | conjunto de atributos do spot-checking | `full` (27 atributos) |
| `search` (D9) | busca de hiperparâmetros e finalistas | 30 configurações; finalistas: LightGBM, Random Forest e MLP |

`configs/features.yaml` (D13, decidido em 26/10) diz o que acontece com cada coluna antes dos modelos:
- `drop`: 12 colunas removidas, cada uma com o motivo e a entrada do diário. Ficam 27 atributos (limpeza, antes da divisão);
- `fill_null`: `Std` ausente vira 0 (regra fixa, antes da divisão);
- `categorical`: `Protocol Type` vira *one-hot* em todos os pipelines (P6; dentro do `Pipeline`);
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
| Normalização ou padronização | Padronização (`StandardScaler`) em LR, Naive Bayes, kNN, SVM e MLP | Dentro do `Pipeline`, ajustada no treino |
| Codificação | *One-hot* de `Protocol Type` em todos os pipelines (P6) | Dentro do `Pipeline`, ajustada no treino |
| Discretização | Não aplicada | — |
| Balanceamento de classes | Feito pelo desenho da amostra (D3: 50/50) | Antes da divisão (Etapa 2) |
| Redução de dimensionalidade | Não aplicada (27 atributos; PCA impediria a análise de atalhos) | — |

Como usar (é o que as Etapas 6 a 10 vão fazer com o pré-processamento):

```python
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
features = feature_names(E.columns, feature_cfg, "full", auxiliary_columns(cfg))   # 27 atributos
X = prepare(E, features, feature_cfg)        # antes da divisão: linha a linha, sem parâmetro estimado
pipe = build_pipeline("logreg", features, feature_cfg, search_cfg, seed, n_jobs)
pipe.fit(X.iloc[treino], y[treino])          # padronização e one-hot ajustados só no treino
pipe.predict_proba(X.iloc[teste])            # aplicados, sem novo ajuste, ao teste
```

Árvores, Random Forest, LightGBM e o Dummy recebem os atributos numéricos como estão, além do *one-hot* de `Protocol Type`. Dentro de uma validação cruzada (`cross_validate`, `RandomizedSearchCV`), o `Pipeline` é clonado e reajustado em cada fold. Por isso não há data leakage do pré-processamento.

### Etapa 5 — Divisão dos dados: treino, validação e teste

```powershell
make splits
```

Gera todas as divisões uma vez, antes de qualquer treino (≈ 15 s). Cada arquivo de `data/splits/` tem uma linha por instância da amostra, na mesma ordem dela, e diz em qual fold a instância fica de fora:

| Arquivo | Divisão | Colunas | Papel dos folds |
|---|---|---|---|
| `spotcheck_E.csv.gz` | Validação cruzada repetida em E (5 folds × 2) | `repeat0`, `repeat1` | Treino × **validação**: escolhe os finalistas (Etapa 6) |
| `nested_M.csv.gz` | Validação cruzada aninhada em M | `outer` (laço externo); `inner0` a `inner4` (laço interno de cada fold externo; −1 = instância do teste daquele fold) | Laço externo: treino × **teste** (estimativa de desempenho). Laço interno: treino × **validação** (ajuste de hiperparâmetros) |
| `capture_M.csv.gz` | Divisão por captura em M (complementar, proposta) | `capture_outer`, `grouped` | Treino × teste, com capturas inéditas no teste para os 16 tipos agrupados (Etapa 9) |

Tamanhos (`results/splits/partition_sizes.csv`):
- laço externo: 154.040 instâncias de treino e 38.510 de teste;
- laço interno: cerca de 102.693 de treino e 51.347 de validação;
- spot-checking: cerca de 38.483 de treino e 9.621 de validação.

Nas divisões estratificadas, cada um dos 34 rótulos mantém a proporção: a contagem de um rótulo difere em no máximo uma instância entre dois folds.

Como usar (é o que as Etapas 6 a 9 vão fazer):

```python
from src.experiments.make_splits import NESTED, SPOTCHECK, inner_folds, load_assignment, outer_folds, spotcheck_folds
cv_E = spotcheck_folds(load_assignment(cfg, SPOTCHECK, E))   # 10 pares (treino, validação), para cross_validate
assign = load_assignment(cfg, NESTED, M)                     # falha se M não estiver na ordem dos folds
for k, (treino, teste) in enumerate(outer_folds(assign)):    # laço externo
    cv_interno = inner_folds(assign, k)                      # 3 pares, posições relativas a X.iloc[treino]
```

### Etapa 6 — Spot-checking (só em E)

```powershell
make spotcheck
```

Treina os 8 algoritmos e o Dummy com os hiperparâmetros padrão (≈ 15 min, quase todo o tempo é do SVM). Cada um roda nos 10 folds congelados de E (Etapa 5), com o pipeline da Etapa 4. O fold avaliado faz o papel de conjunto de **validação**: o resultado serve para escolher os finalistas, e não para estimar o desempenho final.

| Arquivo (`results/spotcheck/`) | O que tem |
|---|---|
| `metrics_per_fold.csv` | Por algoritmo e fold: MCC, F1 das duas classes, acurácia balanceada, sensibilidade, especificidade, TFP, AUPRC, AUC-ROC e matriz de confusão (`tp`, `fn`, `tn`, `fp`), na validação; MCC no treino (`mcc_train`, P4); avisos de convergência |
| `summary.csv` | Média e desvio padrão de cada métrica por algoritmo, do maior para o menor MCC médio |
| `selection.json` | Ranking, os 3 maiores MCC médios e o grupo de empate (a menos de um desvio padrão do melhor) |
| `fig_spotcheck_mcc.png` | MCC de cada fold (pontos), média (traço) e MCC médio no treino (losango) |
| `timings_per_fold.csv` | Tempo de treino e de predição; é o único arquivo que muda de uma rodada para outra |

Resultado: LightGBM (0,807 ± 0,006) e Random Forest (0,803 ± 0,006) empatam no topo, seguidos do MLP (0,776). Os três são os finalistas propostos para a Etapa 7 (`search.finalist_models` em `configs/base.yaml`). A árvore de decisão e a Random Forest chegam a MCC 1,000 no treino: sem limite de profundidade, elas memorizam o treino (overfitting). A leitura completa está no diário (entrada "2026-11-06 — Etapa 6").

Para conferir que uma nova rodada dá os mesmos números, rode `make spotcheck` e depois:

```powershell
git diff --stat -- results/spotcheck ":(exclude)*.run.json" ":(exclude)*timings*"
```

Esperado: nenhuma saída.

### Etapa 7 — Validação cruzada aninhada (em M)

```powershell
make tune
```

Para cada modelo (LightGBM, Random Forest e MLP, os finalistas; a árvore de decisão, para a H1; e o Dummy, como piso) e cada um dos 5 folds externos de M:
- **laço interno (treino × validação):** a busca aleatória sorteia 30 configurações de `configs/search_spaces.yaml` e escolhe a de maior MCC médio nos 3 folds de **validação** tirados do treino externo. A melhor é retreinada no treino externo inteiro;
- **laço externo (treino × teste):** só então o fold de **teste** é usado, uma única vez, para a estimativa de desempenho.

Leva cerca de 5,4 h com 6 núcleos (Random Forest ≈ 2,7 h, LightGBM ≈ 1,5 h, MLP ≈ 1,0 h). Cada fold grava os seus arquivos ao terminar, e uma execução interrompida recomeça do fold que falta. Para rodar só uma parte: `python -m src.experiments.nested_cv --models decision_tree --folds 0`. **Deixe a tampa do notebook aberta:** fechá-la suspende a máquina mesmo durante a execução (ver a seção 11).

| Arquivo (`results/nested/`) | O que tem |
|---|---|
| `<modelo>/fold<k>.json` | Melhor configuração; MCC na validação (média e desvio nos 3 folds internos) e no treino (P4); MCC no treino externo; métricas no teste e matriz de confusão |
| `<modelo>/fold<k>_search.csv` | As 30 configurações do laço interno, com o MCC médio na validação e no treino e o ranking. As 30 são as mesmas em todos os folds |
| `<modelo>/fold<k>_predictions.csv.gz` | Predições fora do fold: posição em M, classe verdadeira, classe prevista e escore de cada instância de teste (entrada da Etapa 8) |
| `<modelo>/fold<k>_timing.json` | Tempos; é o único arquivo que muda de uma rodada para outra (e inclui o tempo em que a máquina ficou suspensa) |
| `folds.csv` | Uma linha por modelo e fold, com tudo acima menos as predições |
| `nested_cv*.run.json` | Um registro por chamada (`nested_cv_fold0`, `nested_cv_fold1_fold2_fold3_fold4` e a consolidação, `nested_cv`) |

Resultado (MCC no teste, média ± desvio nos 5 folds externos): LightGBM 0,832 ± 0,002, Random Forest 0,824 ± 0,002, MLP 0,801 ± 0,003, árvore de decisão 0,793 ± 0,003 e Dummy 0. A ordem é a mesma em todos os folds. A configuração escolhida se repete entre folds (Random Forest em 5 de 5, LightGBM em 4 de 5). Onde muda, as vencedoras empatam. O MLP e a Random Forest escolheram a borda do espaço de busca, o que gerou a proposta P8. A leitura completa está no diário (entrada "2026-11-07 — Etapa 7").

Para conferir que um fold se reproduz, use o trecho 4.10 (refaz a busca da árvore no fold 0 em cerca de 40 s). Rodar `make tune` de novo não refaz nada, porque todos os folds já existem.

---

## 7. Como validar tudo por conta própria

São cinco níveis, do mais rápido ao mais completo. Rode no terminal do VS Code, na raiz do repositório.

### Nível 1 — Os testes passam (≈ 15 s)

```powershell
make test
```

Esperado: `37 passed`. Os testes conferem, com dados pequenos inventados:
- que a ingestão preserva valores e descarta só linhas malformadas;
- que o mapeamento de rótulos tem 34 rótulos e 12 ataques DDoS;
- que a deduplicação segue as três regras e que E e M saem disjuntas e 50/50;
- que o sorteio é determinístico e que os grupos de janela batem com o artigo;
- (`test_no_leakage.py`, Etapa 4) que:
  - o pré-processamento antes da divisão é linha a linha e deixa 27 atributos;
  - as médias e os desvios da padronização e as categorias do *one-hot* vêm só das instâncias de treino;
  - mudar as instâncias de teste não muda nada do que foi ajustado;
  - a validação cruzada reajusta o pré-processamento em cada fold;
  - os 9 pipelines (8 algoritmos + Dummy) treinam e predizem;
- (`test_splits.py`, Etapa 5) que:
  - os folds lidos dos arquivos são os mesmos do scikit-learn com a mesma semente;
  - os folds externos são disjuntos e cobrem a amostra, e nenhuma instância de teste entra no laço interno;
  - cada rótulo mantém a sua proporção em todos os folds;
  - na divisão por captura, cada captura dos tipos agrupados fica inteira num fold;
  - os arquivos versionados de `data/splits/` cumprem os critérios de pronto da Etapa 5;
- (`test_spotcheck.py`, Etapa 6) que:
  - as métricas batem com as definições, numa matriz de confusão feita à mão;
  - a predição usada nas métricas é idêntica ao `predict` de cada pipeline, e o escore do SVM é a `decision_function`;
  - o MCC no treino denuncia uma árvore que memoriza dados aleatórios;
  - a regra de seleção reconhece um empate no topo;
- (`test_nested_cv.py`, Etapa 7) que:
  - todos os espaços de busca de `search_spaces.yaml` geram configurações válidas, dentro dos limites, que chegam ao modelo;
  - durante a busca, cada ajuste recebe exatamente o treino do seu fold interno, o retreino recebe o treino externo inteiro e nenhuma instância de teste aparece em nenhum deles;
  - o Dummy é avaliado sem busca;
  - a tabela consolidada só é gravada quando todos os modelos têm todos os folds.

### Nível 2 — Refazer as Etapas 2, 3 e 5 e obter os mesmos arquivos (≈ 45 s)

```powershell
make sample
```

```powershell
make eda
```

```powershell
make splits
```

```powershell
git status --short
```

Esperado: só aparecem os três registros de execução (`results/eda/eda.run.json`, `results/sampling/make_sample.run.json` e `results/splits/make_splits.run.json`), que mudam porque guardam o horário. Para confirmar que **nenhum resultado** mudou:

```powershell
git diff --stat -- results data ":(exclude)*.run.json"
```

Esperado: nenhuma saída. Isso prova que as amostras, os folds, as tabelas e as figuras que você acabou de gerar são idênticas, byte a byte, às versionadas. Depois, descarte os registros de execução da sua rodada:

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

**4.8 — Os folds congelados são os do scikit-learn e não misturam teste com treino (Etapa 5):**

```python
import polars as pl
from sklearn.model_selection import StratifiedKFold
M = pl.read_parquet("data/processed/sample_M.parquet", columns=["source_file", "row_idx", "label34"])
f = pl.read_csv("data/splits/nested_M.csv.gz")
print(f.select("source_file", "row_idx").equals(M.select("source_file", "row_idx")))   # True
print(f["outer"].value_counts().sort("outer")["count"].to_list())                      # [38510, 38510, 38510, 38510, 38510]
print(f.group_by("label34", "outer").len().group_by("label34")
      .agg((pl.col("len").max() - pl.col("len").min()).alias("spread"))["spread"].max())   # 1
print(f.filter(pl.col("outer") == 0)["inner0"].unique().to_list())                    # [-1]
y = M["label34"].to_numpy()
sk = [teste for _, teste in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(y, y)]
print(all((f["outer"].to_numpy()[t] == k).all() for k, t in enumerate(sk)))           # True
```

Esperado:
- `True`: o arquivo está na mesma ordem de M, instância por instância;
- `[38510, ...]`: os 5 folds de teste têm o mesmo tamanho e, juntos, cobrem as 192.550 instâncias;
- `1`: a contagem de um rótulo difere em no máximo uma instância entre dois folds (estratificação por `label34`);
- `[-1]`: as instâncias de teste do fold externo 0 não estão em nenhum fold do laço interno dele;
- `True`: o próprio scikit-learn, com a semente 42, gera os mesmos folds de teste.

**4.9 — O spot-checking: o MCC sai da matriz de confusão, e um fold se reproduz (Etapa 6):**

```python
import polars as pl
from sklearn.metrics import matthews_corrcoef
from src.experiments.make_splits import SPOTCHECK, load_assignment, spotcheck_folds
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.utils.config import load_config
d = pl.read_csv("results/spotcheck/metrics_per_fold.csv").filter(pl.col("algorithm") != "dummy")
tp, tn, fp, fn = (pl.col(c) for c in ("tp", "tn", "fp", "fn"))
mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).sqrt()
print(d.select((pl.col("mcc") - mcc).abs().max()).item())                     # 0.0
print(d.group_by("algorithm").agg(pl.col("mcc").mean().round(3).alias("validation"),
                                  pl.col("mcc_train").mean().round(3).alias("training"))
      .sort("validation", descending=True))
cfg, fcfg, scfg = load_config(), load_config("features"), load_config("search_spaces")
E = pl.read_parquet("data/processed/sample_E.parquet")
feats = feature_names(E.columns, fcfg, "full", auxiliary_columns(cfg))
X, y = prepare(E, feats, fcfg), E["y"].to_numpy()
treino, validacao = spotcheck_folds(load_assignment(cfg, SPOTCHECK, E))[0]
pipe = build_pipeline("lightgbm", feats, fcfg, scfg, seed=42, n_jobs=6).fit(X.iloc[treino], y[treino])
print(round(matthews_corrcoef(y[validacao], pipe.predict(X.iloc[validacao])), 6))   # 0.807424
print(round(d.filter((pl.col("algorithm") == "lightgbm") & (pl.col("repeat") == 0)
                     & (pl.col("fold") == 0))["mcc"].item(), 6))                   # 0.807424
```

Esperado:
- `0.0`: o MCC gravado é exatamente o que sai da matriz de confusão de cada fold (VP, VN, FP, FN);
- a tabela: LightGBM 0,807 na validação e 0,841 no treino; Random Forest e árvore de decisão com 1,0 no treino (overfitting); regressão logística, Naive Bayes, SVM e MLP com treino e validação quase iguais;
- `0.807424` duas vezes: retreinar o LightGBM no primeiro fold dá o mesmo MCC que está gravado.

**4.10 — A validação cruzada aninhada: as predições cobrem M uma vez, e um fold se reproduz (Etapa 7, ≈ 40 s):**

```python
import json
import polars as pl
from sklearn.metrics import matthews_corrcoef
from src.experiments.make_splits import NESTED, inner_folds, load_assignment, outer_folds
from src.experiments.nested_cv import param_distributions, tune_and_test
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.utils.config import load_config
t = pl.read_csv("results/nested/folds.csv")
print(t.group_by("model", maintain_order=True).agg(pl.col("test_mcc").mean().round(3).alias("mean"),
                                                   pl.col("test_mcc").std().round(3).alias("std")))
p = pl.concat([pl.read_csv(f"results/nested/lightgbm/fold{k}_predictions.csv.gz") for k in range(5)])
print(p["position"].n_unique(), p.height)                                              # 192550 192550
p0 = pl.read_csv("results/nested/lightgbm/fold0_predictions.csv.gz")
print(round(matthews_corrcoef(p0["y"], p0["pred"]), 6),
      round(t.filter((pl.col("model") == "lightgbm") & (pl.col("outer_fold") == 0))["test_mcc"].item(), 6))
cfg, fcfg, scfg = load_config(), load_config("features"), load_config("search_spaces")
M = pl.read_parquet("data/processed/sample_M.parquet")
feats = feature_names(M.columns, fcfg, "full", auxiliary_columns(cfg))
X, y = prepare(M, feats, fcfg), M["y"].to_numpy()
assign = load_assignment(cfg, NESTED, M)
treino, teste = outer_folds(assign)[0]
pipe = build_pipeline("decision_tree", feats, fcfg, scfg, seed=cfg["seed"], n_jobs=1)
out = tune_and_test(pipe, param_distributions(scfg["decision_tree"]["params"]), X, y, treino, teste,
                    inner_folds(assign, 0), n_iter=cfg["search"]["n_iter"], seed=cfg["seed"], n_jobs=cfg["n_jobs"])
gravado = json.loads(open("results/nested/decision_tree/fold0.json", encoding="utf-8").read())
print(out["best_params"] == gravado["best_params"], round(out["test"]["mcc"], 6), round(gravado["test"]["mcc"], 6))
```

Esperado:
- a tabela: LightGBM 0,832 (desvio 0,002), Random Forest 0,824 (0,002), MLP 0,801 (0,003), árvore de decisão 0,793 (0,003) e Dummy 0,0;
- `192550 192550`: juntas, as predições dos 5 folds de teste cobrem cada instância de M exatamente uma vez;
- `0.833506` duas vezes: o MCC gravado sai das predições gravadas;
- `True 0.793446 0.793446`: refazer a busca da árvore no fold 0 escolhe a mesma configuração e dá o mesmo MCC no teste.

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
| 8 | A divisão dos dados está congelada: todos os algoritmos vão usar os mesmos folds, nenhuma instância de teste entra no laço interno, e cada rótulo mantém a proporção em todos os folds | `data/splits/`, `results/splits/splits_report.json`, `tests/test_splits.py` | Níveis 1 e 4.8 |
| 9 | Com hiperparâmetros padrão, os ensembles de árvores lideram (LightGBM 0,807; Random Forest 0,803) e o MCC em E não passa de cerca de 0,81, bem abaixo do benchmark binário do artigo do dataset. A explicação provável, ainda uma hipótese, é que os ataques de janela 10 se parecem com o benigno | `results/spotcheck/summary.csv`, `selection.json`, `fig_spotcheck_mcc.png` | Nível 4.9 |
| 10 | Árvores sem limite de profundidade memorizam o treino (MCC 1,000 no treino, contra 0,723 e 0,803 na validação); LR e Naive Bayes ficam com treino e validação iguais e baixos (underfitting) | `results/spotcheck/summary.csv` | Nível 4.9 |
| 11 | Com hiperparâmetros ajustados, a estimativa de desempenho no laço externo é LightGBM 0,832 ± 0,002, Random Forest 0,824, MLP 0,801 e árvore de decisão 0,793, na mesma ordem em todos os folds. O teto continua perto de 0,83, e os testes estatísticos ficam para a Etapa 8 | `results/nested/folds.csv` | Nível 4.10 |
| 12 | A escolha dos hiperparâmetros é estável: a mesma configuração na Random Forest em 5 de 5 folds e no LightGBM em 4 de 5; onde muda, as vencedoras ficam a no máximo 0,0017 uma da outra. A validação do laço interno fica um pouco abaixo do teste (treino menor) | `results/nested/<modelo>/fold<k>_search.csv`, `folds.csv` | Compare a coluna `rank` dos `fold<k>_search.csv` de um modelo |
| 13 | O ajuste acabou com a memorização da árvore (diferença treino − validação de 0,277 para 0,026). O MLP escolheu a maior arquitetura oferecida em todos os folds, e a Random Forest, o maior `max_features`: os números deles podem ser um limite inferior (P8) | `folds.csv` (`inner_training_mcc`, `inner_validation_mcc`), `fold<k>.json` | Nível 4.10 e os `fold<k>.json` |

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

**Próximas etapas** (cronograma do PLANO §8; em 07/11, a Etapa 7 terminou dentro da janela prevista):

| Etapa | Previsto no plano | O que será feito |
|---|---|---|
| 8 | 08–10/11 | Estimativa de desempenho e comparação de modelos: tabelas, figuras e testes de hipóteses; Q1 e Q2 respondidas |
| 9 | 10–13/11 | Interpretação do melhor modelo, ablação dos atalhos e *leave-capture-out*; Q3 respondida |
| 10 | 13–15/11 | Modelo final treinado em toda a amostra M; `make all` funcionando |

**Pendências da dupla:**
- decidir a P8: o que fazer com as configurações do MLP e da Random Forest que ficaram na borda do espaço de busca (entrada "2026-11-07 — Etapa 7" do diário);
- decidir P1 (D16, métricas reponderadas para a distribuição original) e P5 (intervalos de confiança por bootstrap) antes da Etapa 8;
- preencher, no `docs/IA_LOG.md`, os campos "O que foi aproveitado" e "O que foi rejeitado" de cada entrada, e registrar as conversas de planejamento no claude.ai;
- baixar a pasta *Supplementary material* do dataset (o formulário pede nome, e-mail e instituição) para confirmar se as linhas estão em ordem cronológica. Isso só importa se a dupla quiser reabrir a análise complementar (b) da D8, de blocos contíguos, que não foi adotada;
- escolher a licença do repositório e pôr o nome do(a) colega no README;
- antes das execuções longas da Etapa 9, deixar o notebook na tomada com a tampa aberta, ou mudar a ação de fechar a tampa para "Não fazer nada" com o carregador ligado (seção 11).

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
| Um `fold<k>_timing.json` ou `elapsed_s` com horas a mais que o normal | O notebook suspendeu durante a execução (tampa fechada ou botão de energia). O cálculo só pausa: os resultados não mudam, só o tempo de relógio. Para ver as suspensões: `Get-WinEvent -FilterHashtable @{LogName='System'; Id=42,107}` no PowerShell. Para evitar: tampa aberta, ou "Ao fechar a tampa: Não fazer nada" com o carregador ligado, nas Opções de Energia do Windows |
