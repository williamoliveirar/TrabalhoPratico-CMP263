# Registro de uso de IA

Exigência do enunciado: as ferramentas de IA generativa usadas devem ser citadas no artigo, e o material suplementar deve trazer os prompts usados e as limitações ou potencialidades observadas (ver `docs/PLANO.md`, seção 6.3).

Regras:
- uma entrada por uso relevante, em ordem cronológica, sempre acrescentada no fim;
- entradas antigas não são editadas;
- os campos de avaliação ("aproveitado", "rejeitado ou corrigido") são preenchidos pela dupla, que é quem decide o que fica.

Modelo:

```markdown
## AAAA-MM-DD — Ferramenta (ex.: Claude no claude.ai / Claude Code) — etapa
**Finalidade:** ...
**Prompt:** texto integral ou resumo fiel (para prompts longos)
**O que foi aproveitado:** ...
**O que foi rejeitado ou corrigido, e por quê:** ...
**Limitações ou potencialidades observadas:** ...
```

## Pendências

- [ ] Acrescentar as conversas de planejamento feitas no claude.ai antes da implementação, que deram origem a `docs/PLANO.md` (e o arquivo de contexto usado nelas).

---

## 2026-10-25 — Claude Code (Claude Opus 5.5, app desktop) — Etapa 0 e antecipação da Etapa 1

**Finalidade:** ler o plano e o enunciado; estruturar o repositório do GitHub; recomendar qual das duas versões baixadas do CICIoT2023 usar (CSVs oficiais ou Parquet limpo do Kaggle).

**Prompt (texto integral, com o caminho local do plano omitido):**

> Leia e compreenda o arquivo `PLANO_TRABALHO_CMP263_CICIoT2023.md`, será o guia para começarmos e concluirmos esse projeto Ele traz:
>
> * o resumo do enunciado (entregáveis, pesos, prazos e regras);
> * o problema binário com as três perguntas de pesquisa (desempenho, onde o modelo falha e atalhos);
> * o dataset, com os pontos ainda a confirmar marcados como [verificar];
> * a tabela de decisões metodológicas e a estrutura do repositório;
> * 13 etapas, cada uma com tarefas, saídas e critério de pronto;
> * o cronograma, a estrutura do artigo com orçamento de páginas e o checklist final.
>
> IMPORTANTE:
> 1) Já tenho um repositório criado no GitHub para o Trabalho Final, popule ele, crie os diretórios necessários e os arquivos .MD essenciais e deixe-o pronto, conforme as instruções do Plano.md: https://github.com/williamoliveirar/TrabalhoPratico-CPM263
> 2) Já fiz o download do Dataset, em duas versões, que estão nesse diretório de trabalho. A primeira (\CICIoT2023-Dataset\CICIoT2023-Dataset-CSV) é a versão completa original, em .CSV, separada pelos tipos de ataque, extraída do site oficial do Dataset (https://www.unb.ca/cic/datasets/iotdataset-2023.html). A segunda (..\Cleaned-CICIoT2023-Dataset Parquet with duplicate observations retained) é a versão já tratada do dataset, que contém apenas um arquivo Parquet tipado e compactado, juntamente com seu relatório de limpeza legível por máquina. Observações duplicadas foram mantidas intencionalmente nesse arquivo. A limpeza corrige valores de taxa não finitos, preenche valores de dispersão unitários definidos matematicamente, omite três registros de origem malformados, remove o tamanho total (Tot size) por ser uma duplicata exata da média (AVG) e adiciona rótulos detalhados, de categoria e binários., disponível no Kaggle (https://www.kaggle.com/datasets/dhoogla/ciciotdataset2023/data). De acordo com o enunciado 'Trabalho Prático - Enunciado.pdf', podemos decidir qual das duas versões utilizaremos. Qual das duas seria melhor para fazermos esse trabalho?

**O que a ferramenta produziu:**
- estrutura do repositório conforme a seção 5.1 do plano, mais `data/external/` (espelho do Kaggle) e `data/interim/` (ingestão própria proposta em D15);
- `CLAUDE.md` (modelo da seção 0.2), `README.md`, `docs/PLANO.md` (cópia do plano), este registro, `docs/DIARIO.md`, `docs/artigo/` com o template LaTeX da SBC;
- `configs/base.yaml`, `configs/search_spaces.yaml` (espaços da Etapa 7), `configs/features.yaml` (vazio até a EDA), `Makefile` (esqueleto), `.gitignore`, `.gitattributes`, `requirements.in` e `requirements.txt`;
- `src/utils/config.py` e `src/data/check_kaggle_mirror.py`, que compara as duas versões e grava `results/inspection/kaggle_vs_official.json`;
- recomendação de usar os CSVs oficiais como fonte e o Parquet do Kaggle só como conferência (D15, ver diário).

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Limitação:* o plano, redigido com ajuda de IA a partir de memória, descreve 46 atributos (versão `MERGED_CSV`), mas a versão baixada (CSVs por captura) tem 39 colunas, com nomes diferentes (por exemplo, `Time_To_Live` em vez de `Duration`). A convenção `[verificar]` do plano e a regra "os dados vencem" evitaram que o erro se propagasse.
- *Limitação:* a primeira tentativa de criar o ambiente falhou pelo limite de 260 caracteres de caminho do Windows. A ferramenta não previu o problema e só o contornou depois do erro (ambiente num caminho curto).
- *Potencialidade:* a comparação automática entre as duas versões (contagens por arquivo e valores) foi rápida e deixou um artefato versionado que serve de fonte para os números do diário.

## 2026-10-25 — Claude Code (Claude Opus 5.5, app desktop) — fechamento da Etapa 0 e Etapa 1

**Finalidade:** criar os utilitários de semente e de metadados de execução (Etapa 0); implementar a ingestão dos CSVs oficiais (D15) e a inspeção dos dados brutos (`make inspect`); conferir a ingestão contra o espelho do Kaggle.

**Prompt (texto integral):**

> Confirmo a D15. Feche a Etapa 0 (seed.py e run_metadata.py) e comece a Etapa 1 com a ingestão e o make inspect.

**O que a ferramenta produziu:**
- `src/utils/seed.py`, `src/utils/run_metadata.py` (registro `<script>.run.json` com commit, hash da configuração, versões e tempo) e `src/utils/io.py`;
- `src/data/labels.py` (pasta → `label34` → `category8` → `y`), `src/data/ingest.py` (CSVs → Parquet com valores inalterados, manifesto com SHA-256, linhas malformadas registradas), `src/data/inspect_raw.py` e `src/data/check_ingest.py` (conferência linha a linha com o Kaggle);
- alvos `ingest` e `check-ingest` no `Makefile`, testes em `tests/test_ingest.py`;
- teste da receita de `make env` num clone limpo (sem o binário do Make) e as entradas do diário desta sessão.

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Potencialidade:* o teste do mapeamento de rótulos, escrito a partir da tabela do plano (§3.3), falhou contra os dados e revelou um erro do próprio plano: a categoria DDoS tem 12 ataques, e não 11 (falta `DDoS-ICMP_Fragmentation`).
- *Potencialidade:* a ferramenta notou que o leitor de CSV do polars completa com nulos, sem aviso, as linhas truncadas, e trocou para o leitor do pyarrow, que reporta essas linhas.
- *Limitação:* a primeira tentativa de testar `make env` numa pasta limpa falhou de novo pelo limite de 260 caracteres de caminho do Windows (o clone foi feito numa pasta temporária profunda), um problema que a ferramenta já tinha encontrado na sessão anterior.

## 2026-10-25 — Claude Code (Claude Opus 5.5, app desktop) — D8, fechamento da Etapa 1 e Etapa 2

**Finalidade:** registrar a confirmação da D8; ler o artigo do dataset para fechar a Etapa 1; implementar e executar a amostragem da Etapa 2.

**Prompts (texto integral):**

> Confirmo a D8: folds por linha estratificados por label34, com leave-capture-out e blocos contíguos como análises complementares a decidir na Etapa 5.

> o que significa exatamente, os folds por linha serem "estratificados por label34"? Me explique melhor.

> Continue. Faça o push do commit da D8, e prossiga para as próximas etapas, me atualizando do cronograma.

**O que a ferramenta produziu:**
- registro da D8 no diário e na configuração;
- explicação, no chat, do que é estratificar os folds por `label34` (sem arquivo);
- leitura do artigo do dataset (texto completo na PubMed Central, porque a MDPI recusou o acesso automatizado) e da página oficial, com as respostas da Etapa 1 no diário e correções pontuais no plano (§3.3, §3.5 e Etapa 2);
- `src/data/make_sample.py`, `tests/test_sample.py`, as amostras E e M, o índice e o relatório da amostragem.

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Potencialidade:* ao ler o artigo, a ferramenta cruzou a lista de tamanhos de janela por ataque (10 ou 100 pacotes) com as contagens já versionadas. Daí saiu a hipótese de um atalho criado pela própria extração (`Number` e os atributos que crescem com a janela), mais forte que a suspeita do plano sobre o TTL. A ferramenta não olhou a relação entre atributo e classe no dataset inteiro e deixou a confirmação para a amostra E.
- *Limitação:* a leitura do artigo passou por um resumo automático da página. As citações usadas no diário foram pedidas literalmente numa segunda leitura, mas convém conferir no PDF antes de citá-las no artigo.
- *Limitação:* o material suplementar (scripts de extração), que responderia se as linhas estão em ordem cronológica, fica atrás de um formulário com dados pessoais; a ferramenta não o preenche, e a pendência ficou com a dupla.
- *Limitação:* a ferramenta tomou uma decisão que o plano não previa (remover vetores compartilhados por dois tipos de ataque) para poder executar a Etapa 2. Ela foi registrada como proposta, mas já está nos arquivos gerados; se a dupla preferir a alternativa, a amostragem precisa ser refeita.

## 2026-10-25 — Claude Code (Claude Opus 5.5, app desktop) — Etapa 3 (EDA na amostra E)

**Finalidade:** registrar as confirmações da D4 e da D12; rodar `make env` com o Make instalado; fazer a EDA da Etapa 3 e propor o pré-processamento (D13).

**Prompts (texto integral):**

> Confirmo a D4 refinada, remover como já está implementado. Faça o Push dos 4 commits, e concordo com a revisão da D12 para incluir Number entre os candidatos a atalho. Prossiga.

> eu já instalei o Make, rode o make env também.

**O que a ferramenta produziu:**
- registro das confirmações no diário; execução de `make env`, `make test` e `make eda`;
- `src/data/eda.py` (tabelas e figuras em `results/eda/`), `src/reporting/style.py` (estilo e paleta das figuras), `window_of` em `src/data/labels.py` e `tests/test_eda.py`;
- a proposta de `configs/features.yaml` (D13) e a entrada da Etapa 3 no diário.

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Potencialidade:* a ferramenta testou relações exatas entre colunas (contagem = fração × `Number`, `Tot sum` = `AVG` × `Number`, `ARP` = 1 − `IPv`, `Variance` = `Std`²) em vez de parar nas correlações. Assim, a remoção dessas colunas tem justificativa de identidade, e não só de correlação alta.
- *Potencialidade:* a ferramenta olhou as figuras geradas antes de commitar e corrigiu dois problemas: a escala log que achatava o `IAT` e a legenda que cobria as barras.
- *Limitação:* a primeira versão da regra de cauda pesada marcava frações em [0, 1] e o código categórico `Protocol Type` como candidatos a `log1p`. Só a revisão das tabelas mostrou o erro, corrigido antes dos resultados finais.
- *Limitação:* a proposta da D13 envolve escolhas de julgamento (por exemplo, qual coluna manter em cada par redundante e remover `Rate` em vez de `IAT`). A ferramenta as justificou, mas a decisão é da dupla.

## 2026-10-26 — Claude Code (Claude Opus 5.5, app desktop) — confirmação da D13 e guia prático

**Finalidade:** registrar a confirmação da D13 e escrever um guia didático do que já foi feito, com instruções para rodar e validar tudo sem ajuda.

**Prompt (texto integral):**

> Faça os pushs dos novos commits e Confirmo a D13. Em paralelo, para eu não me perder, quero que você me explique, passo a passo e de forma bem didática, TUDO o que já foi feito e construído em relação ao projeto, os códigos e o ambiente implementado, e me ensine como rodar os códigos (quais e como) e validar tudo por conta própria, testar e chegar às mesmas conclusões que você já chegou, de forma prática e simples.

**O que a ferramenta produziu:**
- registro da D13 no diário e no `features.yaml`;
- `docs/GUIA.md`: conceitos, ambiente, mapa do repositório, pipeline etapa por etapa, cinco níveis de validação, as conclusões com a evidência de cada uma, o fluxo do Git e problemas comuns.
- Antes de escrever, a ferramenta rodou de novo `make sample` e `make eda` e confirmou que só os `.run.json` mudam. Também executou todos os trechos de Python do guia e conferiu as saídas esperadas.

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Potencialidade:* o guia transforma o determinismo do pipeline num teste simples: refazer as etapas e verificar com `git diff` que nenhum resultado versionado mudou.
- *Limitação:* o guia descreve o estado em 26/10 e precisa ser atualizado a cada etapa. Os níveis 3 e 4 de validação só funcionam numa máquina que tenha os CSVs brutos e já tenha rodado `make ingest` e `make sample`.

