# Trabalho Prático --- Inteligência Artificial (FACOM/UFMS, 2026/1)

**Aluno:** _Diego Rodrigues de Souza Arimura_

**Matrícula:** 202601525

**Nível:** _Mestrado_

**Tema da coleção:** _Engenharia de Software para Aplicativos Móveis Cross-Platform: Análise de Qualidade de Código, Testes Automatizados e Desempenho em Flutter, React Native, Android e iOS_

## Estrutura do repositório

```
.
├── README.md                <- este arquivo
├── requirements.txt         <- dependências Python (incluindo Flask)
├── demo.py                  <- script de demonstração interativa no terminal (CLI)
├── gui.py                   <- servidor web Flask para demonstração visual (GUI)
├── templates/
│   └── index.html           <- interface frontend da GUI web
├── data/                    <- coleção bruta, processada e cache de embeddings
│   ├── arxiv_raw.jsonl
│   ├── corpus.jsonl
│   └── doc_embeddings_all-MiniLM-L6-v2.npy
├── notebooks/
│   ├── 01_coleta_arxiv.ipynb
│   ├── 02_baseline_bm25.ipynb
│   ├── 03_retrieval_knn.ipynb
│   ├── 04_modulo_aprofundamento.ipynb
│   └── runs/                <- arquivos .trec gerados pelos modelos
├── src/                     <- código modular e scripts de apoio
│   ├── preprocessing.py     <- pipeline de normalização e tokenização de texto
│   ├── retrievers.py        <- implementação dos modelos BM25, KNN e RRF Híbrido
│   ├── collect.py           <- script de coleta paginada da API do ArXiv
│   ├── generate_pooling.py  <- geração de pool de candidatos para anotação
│   └── annotate_qrels.py    <- consolidação das anotações em formato TREC
├── eval/
│   ├── queries.tsv          <- as 15 queries de avaliação do sistema
│   ├── qrels.tsv            <- gabarito de julgamentos de relevância em formato TREC
│   └── evaluate.py          <- script oficial de cálculo de métricas
└── relatorio/
    ├── relatorio.tex        <- código-fonte LaTeX em formato SBC
    └── relatorio.pdf        <- relatório compilado em PDF
```

## Reprodução e Uso

### 1. Preparação do Ambiente e Dependências
No Windows (PowerShell ou CMD), execute:
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Execução da Busca Demonstrativa (Terminal CLI)
Permite executar buscas interativamente ou passando a consulta via parâmetro:
```bash
# Execução interativa (perguntará a query no console)
.venv\Scripts\python demo.py

# Ou buscando uma query específica por parâmetro:
.venv\Scripts\python demo.py --query "automated testing in flutter" --k 5
```

### 3. Execução da Busca com Interface Gráfica (GUI Web)
Uma interface local moderna e responsiva em três colunas para comparar o ranking de cada modelo lado a lado em tempo real.
```bash
# Iniciar o servidor local
.venv\Scripts\python gui.py
```
* Abra o seu navegador e acesse: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
* Digite termos de busca livres ou clique nos botões de sugestões rápidas fornecidos.

### 4. Execução da Avaliação Experimental
Para rodar a avaliação oficial com base nas queries de teste e no arquivo de julgamento `qrels.tsv`:
```bash
.venv\Scripts\python eval/evaluate.py \
    --qrels eval/qrels.tsv \
    --runs notebooks/runs/bm25.trec notebooks/runs/knn.trec notebooks/runs/hybrid.trec \
    --k 10
```

## Decisões de projeto

- **Tema/escopo da coleção:** Engenharia de Software para Aplicativos Móveis Cross-Platform: Análise de Qualidade de Código, Testes Automatizados e Desempenho em Flutter, React Native, Android e iOS.
- **Categorias do ArXiv consideradas:** cs.SE, cs.HC, cs.PL, cs.CY, cs.CR.
- **Janela temporal:** 2015 a 2026.
- **Tamanho final da coleção:** 2.699 artigos deduplicados.
- **Pré-processamento:** Remoção de pontuações, conversão para minúsculas, tokenização por expressão regular (preservando termos técnicos com hífen, ex: "cross-platform") e remoção de stopwords (NLTK). Sem stemming para evitar colisões semânticas em termos cruciais (como "Dart").
- **Modelos implementados:** 
  - BM25 (esparso - léxico, com $k_1 = 1.5$ e $b = 0.75$).
  - KNN (denso - semântico, usando SentenceTransformers `all-MiniLM-L6-v2` com normalização L2 e similaridade de cosseno).
  - Híbrido RRF (fusão de ranking com constante $c = 60$).
- **Módulo de aprofundamento (Mestrado):** M5 - Ranking Híbrido (BM25 + KNN com Reciprocal Rank Fusion - RRF).

## Uso de assistentes de IA generativa

Em conformidade com as diretrizes do enunciado do trabalho prático, foi utilizado o assistente inteligente **Antigravity (Google DeepMind)** como ferramenta de apoio em par programado:
1. **Programação e Codificação:** Apoio na estruturação matemática do cálculo de similaridade vetorial da busca semântica, desenvolvimento do servidor de integração REST (`gui.py`) e estilo da interface front-end (`templates/index.html`).
2. **Escrita e Formatação:** Apoio na elaboração e revisão gramatical do relatório acadêmico escrito em formato LaTeX, e na estruturação do roteiro de apresentação em vídeo.

## Vídeo de apresentação

URL: https://youtu.be/WXqr7ucUfEw
