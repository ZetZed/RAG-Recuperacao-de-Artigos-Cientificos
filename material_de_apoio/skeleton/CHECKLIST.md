# Checklist de submissão --- Trabalho Prático de IA (2026/1)

Use esta lista antes de submeter o seu `.zip` no AVA. Cada item evita uma penalidade comum na correção.

## Coleção
- [ ] Coleção tem entre 1.000 e 5.000 artigos.
- [ ] Tema da coleção está alinhado ao seu projeto de pesquisa.
- [ ] Critérios de coleta (palavras-chave, categorias, janela temporal) estão documentados.
- [ ] Arquivos `arxiv_raw.jsonl` e `corpus.jsonl` (ou link externo) estão na entrega.

## Pré-processamento
- [ ] Decisões de tokenização / *stopwords* / *stemming* estão justificadas no relatório.
- [ ] Mesmo pré-processamento é aplicado a consultas e documentos.

## Modelos
- [ ] **BM25 implementado**, hiperparâmetros documentados, conexão com paradigma probabilístico discutida.
- [ ] **KNN/recuperador denso implementado**, escolha da representação justificada.
- [ ] Pelo menos 1 (Mestrado) ou 2 (Doutorado) módulos de aprofundamento implementados.
- [ ] Doutorado: módulo M1 ou M5 está entre os escolhidos.

## Avaliação
- [ ] 10 a 20 *queries* criadas, refletindo o tema da coleção.
- [ ] Arquivo `qrels.tsv` com anotações manuais de relevância sobre o *top-k* dos modelos (*pooling*).
- [ ] Métricas P@k, R@k e MAP (ou nDCG) reportadas, com análise.
- [ ] Análise qualitativa de pelo menos 2 *queries* (acertos e falhas).
- [ ] Doutorado: teste estatístico de significância na comparação principal.

## Relatório
- [ ] Formato SBC, até 10 páginas.
- [ ] Seções: Resumo, Introdução, Trabalhos Relacionados, Metodologia, Avaliação, Resultados, Discussão, Conclusão, Referências.
- [ ] Subseção dedicada à relação entre coleção e tema de pesquisa do aluno.
- [ ] Conexões explícitas entre componentes implementados e tópicos da disciplina.
- [ ] Declaração curta de uso de IA generativa (se houve).
- [ ] Link do vídeo em nota de rodapé na conclusão.

## Reprodutibilidade
- [ ] `README.md` com instruções claras de como reproduzir.
- [ ] `requirements.txt` declarando dependências.
- [ ] Notebooks / scripts executam sem ajustes manuais excessivos.
- [ ] Sementes aleatórias fixadas onde relevante.

## Vídeo
- [ ] Duração ≤ 8 minutos.
- [ ] Cobre motivação, decisões de projeto, modelos, metodologia, resultados.
- [ ] Hospedado em plataforma de acesso público (YouTube ou Google Drive).
- [ ] Link incluído no relatório e em `LINKS.txt`.

## Empacotamento
- [ ] `.zip` único contendo: relatório PDF, código-fonte, eval/, `README.md`, `requirements.txt`, `LINKS.txt`.
- [ ] Submetido até **23:59 de 17/06/2026** no AVA.
