# 📚 Sistema RAG para Análise do livro "Os Sertões"

> Sistema de Recuperação e Geração Aumentada (RAG) para análise inteligente da obra clássica "Os Sertões" de Euclides da Cunha, implementando três abordagens progressivas de complexidade.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-orange.svg)](https://openai.com/)

## 📋 Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura dos RAGs](#arquitetura-dos-rags)
  - [1. Naive RAG (Básico)](#1-naive-rag-básico)
  - [2. Parent Document RAG](#2-parent-document-rag)
  - [3. Rerank RAG (Compressor)](#3-rerank-rag-compressor)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Comparação entre Abordagens](#comparação-entre-abordagens)
- [Perguntas de Teste](#perguntas-de-teste)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

## 🎯 Sobre o Projeto

Este projeto implementa três estratégias diferentes de RAG (Retrieval-Augmented Generation) para responder perguntas sobre a obra literária "Os Sertões" de Euclides da Cunha. O sistema é capaz de:

- ✅ Processar e indexar conteúdo de documentos PDF
- ✅ Realizar buscas semânticas avançadas
- ✅ Gerar respostas contextualizadas usando LLMs
- ✅ Comparar diferentes estratégias de recuperação de informação

### 🎯 Objetivos

- Demonstrar a evolução das técnicas de RAG
- Comparar performance entre diferentes abordagens
- Fornecer respostas precisas baseadas exclusivamente no contexto do livro
- Servir como referência para implementação de sistemas RAG

## 🏗️ Arquitetura dos RAGs

### 1. Naive RAG (Básico)

**Notebook:** `naiveRag_challenge.ipynb`

#### 📖 Conceito

A implementação mais simples e direta de RAG. Divide o documento em chunks fixos, cria embeddings e realiza busca por similaridade direta.

#### 🔧 Funcionamento

```
PDF → Chunks (4000 chars) → Embeddings → Vector Store → Similarity Search → LLM → Resposta
```

#### ✨ Características

- **Chunk Size:** 4000 caracteres
- **Chunk Overlap:** 20 caracteres
- **Retrieval:** Top-4 documentos por similaridade
- **Embedding:** OpenAI Embeddings
- **LLM:** OpenAI ou Ollama (llama3) para um llm local
- **Vector Store:** ChromaDB

#### 👍 Vantagens

- Simples de implementar e entender
- Rápido para prototipar
- Baixo custo computacional

#### 👎 Limitações

- Perda de contexto ao quebrar documentos
- Pode retornar chunks incompletos
- Sem hierarquia de informação

### 2. Parent Document RAG

**Notebooks:** `parentRag_challenge.ipynb`

#### 📖 Conceito

Abordagem hierárquica que mantém documentos "pais" completos enquanto indexa chunks "filhos" menores. Ao encontrar um chunk relevante, retorna o documento pai completo.

#### 🔧 Funcionamento

```
PDF → Parent Chunks (4000 chars) + Child Chunks (200 chars)
    → Child Embeddings → Vector Store
    → Similarity Search (child) → Return Parent → LLM → Resposta
```

#### ✨ Características

- **Parent Chunk Size:** 4000 caracteres (overlap: 200)
- **Child Chunk Size:** 200 caracteres
- **Storage:** InMemoryStore (docstore) + ChromaDB (vectorstore)
- **Embedding:** OpenAI Embeddings ou HuggingFace (BAAI/bge-m3)
- **LLM:** OpenAI ou Ollama (llama3) para um llm local

#### 🎯 Estratégia

1. Indexa chunks pequenos (200 chars) para busca precisa
2. Mantém documentos maiores (4000 chars) como contexto
3. Retorna contexto completo ao encontrar match

#### 👍 Vantagens

- Preserva contexto completo do documento
- Busca precisa com chunks pequenos
- Melhor compreensão de informações relacionadas

#### 👎 Limitações

- Maior consumo de memória (duas stores)
- Mais complexo de implementar
- Processamento mais lento

### 3. Rerank RAG (Compressor)

**Notebook:** `rerankRag_challenge.ipynb`

#### 📖 Conceito

A abordagem mais sofisticada. Primeiro recupera um grande conjunto de documentos (k=10), depois usa um modelo de reranking (Cohere) para selecionar apenas os mais relevantes (top-3).

#### 🔧 Funcionamento

```
PDF → Chunks (4000 chars) → Embeddings → Vector Store
    → Retrieve Top-10 → Cohere Rerank → Top-3 Best Matches → LLM → Resposta
```

#### ✨ Características

- **Chunk Size:** 4000 caracteres (overlap: 20)
- **Initial Retrieval:** Top-10 documentos
- **Reranking Model:** Cohere rerank-v3.5
- **Final Selection:** Top-3 documentos rerankeados
- **Embedding:** OpenAI Embeddings
- **LLM:** OpenAI ou Ollama (llama3) para um llm local

#### 🎯 Estratégia

1. Fase 1: Retrieval amplo (10 documentos)
2. Fase 2: Reranking inteligente com modelo especializado
3. Fase 3: Seleção dos 3 melhores documentos
4. Fase 4: Geração da resposta

#### 👍 Vantagens

- Maior precisão na seleção de contexto
- Reduz ruído de documentos irrelevantes
- Melhor qualidade de resposta
- Usa modelo especializado em relevância

#### 👎 Limitações

- Requer API adicional (Cohere)
- Maior custo (duas chamadas de API)
- Processamento mais lento
- Mais complexo de configurar

## 🛠️ Tecnologias Utilizadas

### Core Framework
- **LangChain** - Framework para aplicações com LLMs
- **LangChain Classic** - Componentes legados (ParentDocumentRetriever)
- **LangChain Community** - Integrações comunitárias

### LLMs e Embeddings
- **OpenAI** (GPT-3.5-turbo, OpenAI Embeddings)
- **Ollama** (llama3 - alternativa local)
- **HuggingFace** (BAAI/bge-m3 - alternativa open-source para embeddings)
- **Cohere** (rerank-v3.5 - modelo de reranking)

### Vector Stores
- **ChromaDB** - Banco de dados vetorial - local

### Processamento de Documentos
- **PyPDF** - Extração de texto de PDFs
- **RecursiveCharacterTextSplitter** - Divisão inteligente de texto

### Utilitários
- **python-dotenv** - Gerenciamento de variáveis de ambiente
- **PyTorch** - Backend para modelos HuggingFace
- **Transformers** - Modelos de embeddings alternativos

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Conta OpenAI com API Key
- Conta Cohere com API Key (para Rerank RAG)
- PDF do livro "Os Sertões" (`os-sertoes.pdf`)

## 🚀 Instalação

### 1. Clone o Repositório

```bash
git clone https://github.com/alexsandro-oliveira/RAG_-_LLM.git
cd rag_solution
```

### 2. Crie um Ambiente Virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

## ⚙️ Configuração

### 1. Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sua_chave_openai_aqui
COHERE_API_KEY=sua_chave_cohere_aqui
```

### 2. Documento PDF

Coloque o arquivo `os-sertoes.pdf` na raiz do projeto.

### 3. Estrutura de Diretórios

O sistema criará automaticamente os seguintes diretórios para os bancos de dados vetoriais:

```
db/
├── naiveChallenge_db/    # Naive RAG
├── parentChallenge_db/   # Parent RAG
└── rerankDb/             # Rerank RAG
```

## 💻 Uso

### Executando os Notebooks

#### 1. Naive RAG (Básico)

```bash
jupyter notebook naiveRag_challenge.ipynb
```

Execute todas as células sequencialmente. O notebook irá:
1. Carregar e processar o PDF
2. Criar chunks e embeddings
3. Armazenar no ChromaDB
4. Executar 5 perguntas de teste

#### 2. Parent Document RAG

```bash
jupyter notebook parentRag_challenge.ipynb
```

Execute todas as células. O processo inclui:
1. Configuração de embeddings (OpenAI ou local)
2. Criação de chunks pai e filho
3. Indexação hierárquica
4. Execução das perguntas

#### 3. Rerank RAG

```bash
jupyter notebook rerankRag_challenge.ipynb
```

Execute as células para:
1. Configurar retriever base
2. Adicionar camada de reranking (Cohere)
3. Executar queries com reranking

### Exemplo de Uso Programático

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# Carregar vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="naiveChallenge_db",
    embedding_function=embeddings
)

# Criar retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Fazer pergunta
question = "Quem foi Antônio Conselheiro?"
docs = retriever.get_relevant_documents(question)
```

## 📁 Estrutura do Projeto

```
rag_solution/
│
├── 📓 naiveRag_challenge.ipynb          # Implementação Naive RAG
├── 📓 parentRag_challenge.ipynb         # Parent RAG com OpenAI
├── 📓 desafio_parentRag.ipynb          # Parent RAG com HF local
├── 📓 rerankRag_challenge.ipynb        # Rerank RAG com Cohere
│
├── 📄 os-sertoes.pdf                   # Documento fonte
├── 📄 requirements.txt                 # Dependências Python
├── 📄 .env                             # Variáveis de ambiente (criar)
├── 📄 README.md                        # Este arquivo
│
└── 📂 db/                              # Bancos de dados vetoriais
    ├── naiveChallenge_db/
    ├── parentChallenge_db/
    └── rerankDb/
```

## 🔬 Comparação entre Abordagens

| Característica | Naive RAG | Parent RAG | Rerank RAG |
|---------------|-----------|------------|------------|
| **Complexidade** | ⭐ Baixa | ⭐⭐ Média | ⭐⭐⭐ Alta |
| **Precisão** | ⭐⭐ Moderada | ⭐⭐⭐ Boa | ⭐⭐⭐⭐ Excelente |
| **Velocidade** | ⚡⚡⚡ Rápida | ⚡⚡ Moderada | ⚡ Lenta |
| **Memória** | 💾 Baixa | 💾💾 Alta | 💾 Média |
| **Custo API** | 💰 Baixo | 💰 Baixo | 💰💰 Alto |
| **Contexto** | ❌ Limitado | ✅ Completo | ✅ Otimizado |
| **Setup** | ✅ Simples | ⚠️ Moderado | ⚠️ Complexo |

### Quando Usar Cada Abordagem?

#### ✅ Use Naive RAG quando:
- Precisar de prototipagem rápida
- Trabalhar com documentos curtos
- Tiver restrições de memória
- Custo for prioridade

#### ✅ Use Parent RAG quando:
- Contexto completo for crucial
- Trabalhar com documentos estruturados
- Precisar manter hierarquia de informação
- Qualidade > velocidade

#### ✅ Use Rerank RAG quando:
- Máxima precisão for necessária
- Trabalhar com grandes volumes de documentos
- Puder investir em infraestrutura
- Qualidade > custo

## ❓ Perguntas de Teste

Todas as implementações respondem às mesmas 5 perguntas sobre "Os Sertões":

1. **Ambiente Natural**
   > "Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?"

2. **População Sertaneja**
   > "Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?"

3. **Contexto Histórico**
   > "Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?"

4. **Antônio Conselheiro**
   > "Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?"

5. **Crítica Social**
   > "Quais são os principais aspectos da crítica social e política presentes em 'Os Sertões'? Como esses aspectos refletem a visão do autor sobre o Brasil da época?"

## 🎓 Aprendizados e Insights

### Lições Práticas

1. **Tamanho de Chunk é Crítico:** Chunks muito pequenos perdem contexto, muito grandes perdem precisão.

2. **Reranking Vale a Pena:** Para aplicações de produção, o custo adicional do reranking compensa pela qualidade.

3. **Embeddings Locais:** HuggingFace oferece alternativa viável, mas requer mais recursos.

4. **Prompt Engineering:** O template de prompt é crucial - instruções claras melhoram respostas.

5. **Chunk Overlap:** Pequeno overlap (20-200 chars) ajuda a preservar contexto entre chunks.

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Ideias para Contribuição

- [ ] Implementar outros modelos de embeddings
- [ ] Adicionar interface web (Streamlit/Gradio)
- [ ] Criar testes automatizados
- [ ] Adicionar métricas de avaliação (RAGAS)
- [ ] Implementar cache de resultados
- [ ] Adicionar suporte a outros formatos (EPUB, TXT)

## 📊 Roadmap

- [ ] Implementar avaliação quantitativa (precision, recall, F1)
- [ ] Adicionar visualização de embeddings (UMAP/t-SNE)
- [ ] Criar dashboard de comparação entre RAGs
- [ ] Implementar RAG híbrido (combinando estratégias)
- [ ] Adicionar suporte a multi-query
- [ ] Implementar cache Redis para embeddings

## 🐛 Troubleshooting

### Problemas Comuns

**Erro: "OpenAI API Key not found"**
```bash
# Verifique se o .env está configurado corretamente
cat .env
source .env  # Linux/Mac
```

**Erro: "Out of Memory"**
```python
# Reduza o batch_size no Parent RAG
embeddings_model = HFEmbeddings("BAAI/bge-m3", batch_size=2)
```

**ChromaDB Conflicts**
```bash
# Limpe os bancos existentes
rm -rf db/naiveChallenge_db/
rm -rf db/parentChallenge_db/
rm -rf db/rerankDb/
```

**Cohere API Errors**
```python
# Verifique a cota e modelo disponível
rerank = CohereRerank(top_n=3, model='rerank-v3.5')
```

## 📚 Referências

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Cohere Rerank](https://docs.cohere.com/docs/rerank)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAG Papers and Resources](https://github.com/langchain-ai/rag-from-scratch)

## 👨‍💻 Autor

**Alexsandro Oliveira**

- GitHub: [@alexsandro-oliveira](https://github.com/alexsandro-oliveira)
- LinkedIn: [Alexsandro Oliveira](https://www.linkedin.com/in/alexs-oliveirasantos/)

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!

