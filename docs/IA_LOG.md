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

## 2026-11-05 — Claude Code (Claude Opus 5.5, app desktop) — Etapa 4 e revisão da terminologia pelas aulas

**Finalidade:** implementar a Etapa 4 (pré-processamento sem data leakage) e reescrever a documentação com a terminologia das aulas, conferindo se todas as etapas formais do treinamento de modelos estão cobertas.

**Prompt (texto integral):**

> Faça Push e inicie a Etapa 4. Também, quero que você analise e reescreva as informações dos arquivos de texto do Github, incluindo os em /docs (README, DIÁRIO, GUIA, etc) utilizando as terminologias que a professora Mariana utilizou ao longo das aulas, por exemplo: Separação dos dados em treino, teste e validação; as etapas específicas que fazem parte do tratamento dos dados, até o pré-treino, e então o treino, etc. Certifique-se de que todas as etapas formais de Treinamento de modelos de Aprendizagem de Máquina estão sendo seguidas, e usando a terminologia adequada das auals. Utilize como fonte os materiais de aula que estão disponíveis no diretório "Mestrado PPGC UFRGS\Disciplinas\Aprendizagem de Máquina". Cito como as melhores fontes, os arquivos em .pdf dos diretórios "Avaliação de Modelos Preditivos" e "Pré-processamento de Dados". Você encontrará todos os arquivos de aula no diretório "\Aprendizagem de Máquina\P1 Estudos".

**O que a ferramenta produziu:**
- extração do texto dos slides (Aulas 01b a 15–16, material extra sobre validação cruzada aninhada e data leakage, provas antigas, cronograma v2) e comparação com o plano;
- `src/models/pipelines.py`, `src/models/registry.py` e `tests/test_no_leakage.py` (Etapa 4);
- PLANO v1.1 com o mapa das etapas formais, o glossário, a Etapa 4 reescrita e o quadro dos tipos de data leakage; README, GUIA e quadro "Decisões vigentes" do diário reescritos com os termos das aulas;
- uma entrada de revisão metodológica no diário, com seis propostas. A mais importante é a P1 (D16): a distribuição de classes dos folds de teste não é a original, o que contraria a regra da aula de desbalanceamento.

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Potencialidade:* a leitura sistemática dos slides achou desvios que não estavam no pedido. Os folds de teste com distribuição 50/50, contra a regra da aula de desbalanceamento de classes, e o SVM, que está no programa e faltava entre os algoritmos.
- *Limitação:* a ferramenta não reescreveu as entradas antigas do diário, apesar do pedido. A regra de somente acréscimo do próprio projeto impede isso. No lugar, ela acrescentou uma tabela de correspondência entre os termos antigos e os das aulas. Se a dupla quiser reescrever as entradas antigas, precisa revogar a regra explicitamente.
- *Limitação:* a extração automática de texto dos PDFs perde figuras e embaralha algumas colunas dos slides. Os diagramas, como o da validação cruzada aninhada, foram interpretados pelo texto ao redor.

## 2026-11-05 — Claude Code (Claude Opus 5.5, app desktop) — push da Etapa 4 e Etapa 5

**Finalidade:** enviar ao GitHub os commits da Etapa 4 e implementar a Etapa 5 (divisão dos dados em treino, validação e teste).

**Prompt (texto integral):** resposta à pergunta da ferramenta "Posso fazer o push dos 2 commits novos?", escolhida entre as opções oferecidas:

> Push e Etapa 5

**O que a ferramenta produziu:**
- `src/experiments/make_splits.py` e `tests/test_splits.py`: folds do spot-checking em E, da validação cruzada aninhada em M e de uma divisão complementar por captura em M, congelados em `data/splits/`;
- entrada da Etapa 5 no diário, com o tamanho de cada partição e duas propostas para as análises complementares da D8: adotar o *leave-capture-out* e não adotar os blocos contíguos de `row_idx`;
- atualização do guia, do README e do plano.

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Potencialidade:* a ferramenta propôs um desenho pareado para o *leave-capture-out*. Só os tipos com capturas suficientes mudam de fold, e o efeito da captura pode ser medido sem mudar o resto da divisão.
- *Limitação:* a ferramenta gerou os folds do *leave-capture-out* antes da confirmação da dupla, para que todas as divisões ficassem congeladas antes de qualquer treino. Se a dupla recusar a proposta, o arquivo precisa ser removido.

## 2026-11-06 — Claude Code (Claude Opus 5.5, app desktop) — decisões P2 a P6 e Etapa 6

**Finalidade:** registrar as decisões da dupla sobre as propostas pendentes e executar a Etapa 6 (spot-checking na amostra E).

**Prompt (texto integral):**

> Iniciar Etapa 6. P2: Sim, incluir. P3: Sim, acrescentar a AUC-ROC. P4: Sim, registrar o MCC também. P6: Sim, aplicar o one-hot encoding. Sobre a D8: Concordo com a sua avaliação, faça o que for melhor para a viabilidade do trabalho e do treinamento. Prossiga.

Antes disso, na mesma sessão, a dupla autorizou o push dos commits da Etapa 5 ("Pode fazer o push").

**O que a ferramenta produziu:**
- medição avulsa do custo do SVM RBF num fold de E, antes de incluí-lo;
- SVM no `registry.py`, *one-hot* em todos os pipelines, `src/evaluation/metrics.py` (com AUC-ROC, especificidade e matriz de confusão) e `src/experiments/spotcheck.py`, com 5 testes novos;
- os resultados do spot-checking, a figura e a conferência de que o fold 0 se reproduz exatamente;
- duas entradas no diário: a confirmação das decisões e a Etapa 6, com a proposta de finalistas (LightGBM, Random Forest e MLP) e uma proposta nova, a P7, que inclui a árvore de decisão na Etapa 7 para testar a H1;
- atualização do README, do PLANO e do guia.

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Potencialidade:* a ferramenta mediu o custo do SVM antes de rodar tudo. Com isso, estimou que ele não caberia na Etapa 7 em M, um critério de viabilidade que o plano não previa.
- *Potencialidade:* o diagnóstico de overfitting (P4) mostrou que a comparação da H1 com a árvore sem ajuste seria injusta, e daí veio a P7.
- *Limitação:* a explicação do teto de MCC em torno de 0,81 (os ataques de janela 10 se parecem com o benigno) é uma hipótese apoiada numa conta por diferença. Ela só é confirmada com a sensibilidade por tipo, na Etapa 8.
- *Limitação:* um teste escrito pela ferramenta falhou na primeira execução, porque os dados inventados empatavam exatamente na média. O erro era do teste, não do código, e foi corrigido.

## 2026-11-06 — Claude Code (Claude Opus 5.5, app desktop) — P7 e Etapa 7

**Finalidade:** registrar a decisão da dupla sobre a P7 e executar a Etapa 7 (validação cruzada aninhada na amostra M). A execução terminou em 07/11.

**Prompt (texto integral):** resposta à pergunta da ferramenta "Incluo também a árvore de decisão na Etapa 7, para testar a H1 (P7)?", escolhida entre as opções oferecidas:

> Sim, incluir a árvore

**O que a ferramenta produziu:**
- `src/experiments/nested_cv.py` e `tests/test_nested_cv.py` (6 testes): busca aleatória no laço interno, um único uso do fold de teste no laço externo, retomada por fold e a tabela consolidada `results/nested/folds.csv`;
- a execução dos 5 folds externos para LightGBM, Random Forest, MLP, árvore de decisão e Dummy (cerca de 5,4 h de cálculo), com a conferência de que um fold refeito sai idêntico;
- o diagnóstico dos tempos de relógio inflados: o notebook suspendeu quatro vezes durante a execução (log do Windows);
- duas entradas no diário: a confirmação da P7 e a Etapa 7, com a estabilidade dos hiperparâmetros entre folds e a proposta P8 sobre as bordas do espaço de busca;
- atualização do guia, do README e do PLANO.

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Potencialidade:* o teste dos espaços de busca encontrou um defeito que derrubaria a busca da regressão logística e do SVM: o PyYAML lê `1.0e2` como texto.
- *Potencialidade:* como as 30 configurações candidatas são as mesmas em todos os folds, a ferramenta pôde comparar as escolhas entre folds diretamente e separar empates de instabilidade.
- *Limitação:* a estimativa de tempo da ferramenta (2,5 a 3 h) ficou em cerca de metade do real (5,4 h). Ela se baseou em treinos isolados e não considerou a lentidão de 6 treinos simultâneos.
- *Limitação:* depois do fold externo 0, a ferramenta decidiu sozinha manter os espaços de busca, mesmo vendo o MLP na borda, sem consultar a dupla e contra a instrução do PLANO ("ajustar se a busca bater nos limites"). Ela mesma apontou o problema na entrada da Etapa 7, que traz a decisão para a dupla como P8.
- *Limitação:* o pedido para o Windows não suspender, feito pela ferramenta, não impediu as suspensões por tampa fechada. Isso não afetou os resultados, mas inflou os tempos gravados.

## 2026-11-07 — Claude Code (Claude Opus 5.5, app desktop) — push, P1 e P5 e passagem da Etapa 8

**Finalidade:** enviar ao GitHub os commits das Etapas 6 e 7, explicar as propostas pendentes P1 e P5, registrar a decisão sobre elas e preparar a passagem da Etapa 8 para a colega da dupla.

**Prompts (texto integral):**

> Faça o Push. Me explique melhor a P1 e a P5 pendentes?

> Adora as duas, conforme a sua recomendação. A "Etapa 8 — Estimativa de desempenho e comparação de modelos" será feita pela minha colega, a minha dupla. Falta algo antes de iniciá-la? E quero que você prepare um arquivo .md completo para recomendações, direcionamentos e ponderações ao início dessa Etapa, que serão utilizadas por ela na IA de apoio dela, como o Claude Code, (não precisa atrelar ao repositório do git, envie só para mim).

**O que a ferramenta produziu:**
- explicação da P1 e da P5, com um exemplo numérico da precisão projetada (0,14 com 1% de ataques, no LightGBM) e uma ressalva sobre o que o bootstrap sobre instâncias mede;
- conferência de que a Etapa 8 pode ser feita sem os dados brutos;
- entrada no diário com a decisão e a preparação da Etapa 8;
- um documento de passagem da Etapa 8 para a IA de apoio da colega, entregue a William e não versionado.

**O que foi aproveitado:** [a preencher pela dupla]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher pela dupla]

**Limitações ou potencialidades observadas:**
- *Limitação:* o documento de passagem foi escrito pela ferramenta e vai orientar outra IA. Ele não está no repositório; para o material suplementar do artigo, a dupla deve guardar uma cópia, e a colega deve registrar no IA_LOG os pedidos que fizer à IA dela.

## 2026-11-12 — Claude Code (Claude Opus 5.5, app desktop) — trabalho individual e Etapa 8

**Finalidade:** registrar que o trabalho passou a ser individual e executar a Etapa 8.

**Prompt (texto integral; o nome da colega foi omitido nesta transcrição, por privacidade):**

> a minha colega [nome omitido] (retcelvic) desistiu de fazer a disciplina e não fará mais o trabalho comigo, portanto, o terminarei sozinho e assumirei todas as próximas etapas. Revogue os acessos dela ao repositório e tire as menções ao Claude.md de aprovação de acesso para ela. Já retirei-a como colaboradora. E vamos seguir com as próximas etapas desde já.

**O que a ferramenta produziu:**
- conferência, pela API do GitHub, de que a conta da colega não tem mais acesso e de que não há convite pendente nem branch ou PR dela;
- registro no diário, no PLANO e no guia, com as consequências para o enunciado e para a reprodução cruzada;
- a Etapa 8 (detalhada nas entradas do diário de 12/11).

**O que foi aproveitado:** [a preencher por William]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher por William]

**Limitações ou potencialidades observadas:**
- *Potencialidade:* a ferramenta apontou que o enunciado pede trabalho em dupla, o que exige confirmação com a professora.

## 2026-11-12 — Claude Code (Claude Opus 5.5, app desktop) — push da Etapa 8 e Etapa 9

**Finalidade:** enviar os commits da Etapa 8 e executar a Etapa 9 (interpretação do modelo e análise de atalhos). A sessão foi de 12/11 à madrugada de 13/11.

**Prompts (texto integral), em sequência:**

> Pode fazer o push

> Começar a Etapa 9

**O que a ferramenta produziu:**
- o envio dos commits da Etapa 8, depois da conferência de autoria feita em passo separado;
- o pré-registro da Etapa 9 no diário e em `configs/base.yaml`, commitado antes de qualquer cálculo;
- o código da etapa:
  - ablação sobre um executor reaproveitável da validação cruzada aninhada;
  - retreino e conferência dos modelos de cada fold;
  - importância por permutação, SHAP e explicações locais;
  - síntese da Q3 e da H3;
  - modelo final e *leave-capture-out*;
  - 8 testes novos;
- as execuções (cerca de 9 h de cálculo, em parte em paralelo) e o commit de cada resultado;
- a correção de uma figura com texto sobreposto, redesenhada sem recalcular;
- a entrada do diário com a leitura dos resultados e as atualizações do guia, do PLANO e do README.

**O que foi aproveitado:** [a preencher por William]

**O que foi rejeitado ou corrigido, e por quê:** [a preencher por William]

**Limitações ou potencialidades observadas:**
- *Limitação:* a ferramenta fez commits enquanto a ablação rodava. Por isso os `run.json` dos três conjuntos registram commits diferentes, e foi preciso conferir pelo `git diff` que nada do que a ablação importa tinha mudado.
- *Limitação:* o pré-registro escrito com a ferramenta deixou "sensibilidade alta" sem número. O limiar (0,90) foi fixado no código antes da execução, mas depois do pré-registro.
- *Potencialidade:* a ferramenta apontou, antes de ver os resultados, que importância não é dependência e que só a ablação responde a Q3. Depois, apontou a confusão total entre janela de 100 pacotes e ataques de DDoS, DoS e Mirai, e o `HTTPS` como possível atalho fora da lista pré-registrada.
