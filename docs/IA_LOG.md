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
