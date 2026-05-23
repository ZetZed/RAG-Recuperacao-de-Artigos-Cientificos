# Trabalho Prático --- Inteligência Artificial (FACOM/UFMS, 2026/1)

**Aluno:** _Diego Rodrigues de Souza Arimura_
**Matrícula:** 202601525
**Nível:** _Mestrado_
**Tema da coleção:** _Ex.: Detecção de fraudes em transações financeiras com deep learning_

## Estrutura do repositório

```
.
├── README.md                <- este arquivo
├── requirements.txt         <- dependências Python
├── data/                    <- coleção bruta e processada (não versionar arquivos grandes)
│   ├── arxiv_raw.jsonl
│   └── corpus.jsonl
├── notebooks/
│   ├── 01_coleta_arxiv.ipynb
│   ├── 02_baseline_bm25.ipynb
│   ├── 03_retrieval_knn.ipynb
│   ├── 04_modulo_aprofundamento.ipynb
│   └── runs/                <- arquivos .trec gerados pelos modelos
├── src/                     <- código reutilizável
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── retrievers.py
│   └── utils.py
├── eval/
│   ├── queries.tsv
│   ├── qrels.tsv
│   └── evaluate.py
└── relatorio/
    ├── relatorio.tex
    └── relatorio.pdf
```

## Reprodução

```bash
# 1. Criar ambiente e instalar dependências
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Coletar dados (ajuste as palavras-chave no notebook 01)
jupyter notebook notebooks/01_coleta_arxiv.ipynb

# 3. Rodar o baseline BM25
jupyter notebook notebooks/02_baseline_bm25.ipynb

# 4. Rodar avaliação
python eval/evaluate.py \
    --qrels eval/qrels.tsv \
    --runs notebooks/runs/bm25.trec notebooks/runs/knn.trec \
    --k 10
```

## Decisões de projeto

_(documente aqui de forma sucinta as principais decisões; o detalhamento vai no relatório)_

- **Tema/escopo da coleção:** ...
- **Categorias do ArXiv consideradas:** ...
- **Janela temporal:** ...
- **Tamanho final da coleção:** ...
- **Pré-processamento:** ...
- **Modelos implementados:** BM25, KNN-densa, ...
- **Módulo(s) de aprofundamento:** ...

## Uso de assistentes de IA generativa

_(declare aqui, conforme exigido pelo enunciado, em que pontos usou GenAI --- e.g., apoio à escrita, geração de trechos de código, sugestão de hiperparâmetros)_

## Vídeo de apresentação

URL: _https://..._
