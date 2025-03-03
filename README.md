# PDF RAG Chatbot

## Objective

Develop a chatbot that allows users to upload a PDF and query its content using a Retrieval-Augmented Generation (RAG) approach.

## Requirements

- **Text Extraction & Indexing:** Implement functionality to extract and index text from PDF files.
- **Contextual Question Answering:** Enable the chatbot to answer questions by referring back to the relevant sections of the uploaded document.
- **Scalability & Efficiency:** Ensure the solution handles document data efficiently and scales for real-world applications.
- **Documentation:** Provide comprehensive instructions on how to test and deploy the solution.

---

## Project Overview

This project enables users to upload a PDF and interact with its content through a RAG-based chatbot. The system is designed for high retrieval accuracy and contextual relevance while emphasizing scalability and efficiency.

---

## Project Evolution

### Simpler RAG Prototype

I began with a Simpler RAG model to establish the core pipeline:

- **Extraction:** Used PyMuPDF for fallback raw text extraction and simple directory-based loading.
- **Chunking:** Implemented naive but reliable chunking methods to break down the document.
- **Indexing & Retrieval:** Built a vector index using FAISS, enabling robust and fast query retrieval.
- **Response Generation:** Integrated OpenAI’s API to generate answers based on retrieved context.

This Simpler prototype has proven extremely robust and reliable, serving as a solid foundation for understanding the fundamental RAG mechanism.

### Advanced Model

Building on the insights from the simpler prototype, I developed an advanced version with additional features:

- **Adaptive Chunking:** Introduced dynamic chunking based on sentence length and linguistic cues.
- **Structured Extraction & Tagging:** Leveraged the Unstructured library to partition PDFs and tag sections (e.g., Abstract, Introduction, Conclusion), enhancing retrieval granularity.
- **Hybrid Retrieval:** Combined FAISS vector search with BM25 re-ranking for improved context relevance.
- **Enhanced Prompt Engineering:** Fine-tuned prompts for more precise and context-aware response generation.

Although the advanced model offers greater customizability and fine-tuning, its performance varies slightly under certain conditions. The simpler model, however, remains highly robust and reliable. This iterative development approach not only deepened my understanding of RAG systems but also provides a clear path for future enhancements.

---

## Key Features

- **Advanced PDF Processing:**\
  Uses PyMuPDF (fitz) for raw text extraction and Unstructured’s partition\_pdf for structured content extraction, ensuring tables, headers, and lists are preserved.

- **Intelligent NLP Processing:**\
  Utilizes spaCy for tokenization, sentence segmentation, and regex-based section tagging to enhance document structuring.

- **Adaptive Dynamic Chunking:**\
  Implements dynamic chunking based on average sentence length and transition-word cues to produce semantically coherent text chunks.

- **Hybrid Retrieval (Vector + Keyword):**\
  Combines FAISS for vector-based search with BM25 filtering for lexical relevance, improving the precision of context retrieval.

- **Embedding & Response Generation:**\
  Employs SentenceTransformer for generating embeddings and OpenAI’s GPT-4o-mini for context-aware answer generation.

- **Backend & Logging:**\
  Integrates comprehensive logging for monitoring extraction performance and query latency, supporting ongoing optimization.

---

## Repository Details

### Structure

The repository maintains a clear separation between the **Simpler RAG model** and the **advanced model**:

- `simpler_rag_model.ipynb` – Robust and reliable core RAG implementation.
- `advanced_rag_model.ipynb` – Extended model with enhanced features.
- `simpler_rag_requirements.txt` – Dependencies for the simpler model.
- `advanced_rag_requirements.txt` – Dependencies for the advanced model.
- Two **virtual environments** to avoid dependency conflicts between models.
- Data directory for storing uploaded PDFs.
- Logs directory for tracking application performance.

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <repository-url>
cd <repository-folder>
```

### 2. Set Up a Virtual Environment

Since the simpler and advanced models have overlapping but different dependencies, separate virtual environments are required.

#### For the Simpler Model:

```sh
python -m venv simpler_rag_venv
source simpler_rag_venv/bin/activate  # On macOS/Linux
simpler_rag_venv\Scripts\activate    # On Windows
```

#### For the Advanced Model:

```sh
python -m venv advanced_rag_venv
source advanced_rag_venv/bin/activate  # On macOS/Linux
advanced_rag_venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies

The dependencies will be installed automatically when you run the respective Jupyter notebooks, as the first cell in each notebook runs:

```sh
pip install -r simpler_rag_requirements.txt  # For the simpler model
pip install -r advanced_rag_requirements.txt  # For the advanced model
```

Alternatively, install manually:

```sh
pip install -r simpler_rag_requirements.txt
```

*or*

```sh
pip install -r advanced_rag_requirements.txt
```

### 4. Run the Jupyter Notebook

```sh
jupyter notebook
```

Open `simpler_rag_model.ipynb` or `advanced_rag_model.ipynb` and execute the cells.

---

## Challenges Faced & Solutions

### **1. Keyword Detection Issues**

- Initial retrieval was not accurately detecting key phrases and was leading to very irrelevant responses.
- **Solution:** Implemented **Hybrid Retrieval (Vector + Keyword)** to combine FAISS vector search with BM25 filtering for improved precision.

### **2. Cut-off Sentences Leading to Incorrect or Diluted Answers**

- Sentence chunks were split in ways that led to incomplete context that led to LLM hallucinating and generating answers by itself.
- **Solution:** Used **Adaptive Dynamic Chunking**, which analyzes sentence structures to ensure meaningful text segmentation.

### **3. Failure to Detect Structured Sections (Tables, Headers, Lists)**

- Raw text extraction methods did not preserve document structures.
- **Solution:** Implemented **Advanced PDF Processing** using Unstructured’s `partition_pdf`, ensuring better section preservation.

### **4. Dependency Conflicts Between Simpler and Advanced Models**

- Overlapping dependencies caused package mismatches.
- **Solution:** Created separate virtual environments for the simpler and advanced models.

### **5. Latency in Query Processing**

- Response time was slow due to large document sizes.
- **Solution:** Optimized FAISS indexing and added caching mechanisms for faster queries.

---

## Scalability & Efficiency

- **Optimized Query Processing:**\
  The retrieval pipeline leverages FAISS for fast vector search and BM25 for effective keyword filtering, ensuring responsive performance even with large PDFs.

- **Flexible Embedding Options:**\
  The system supports both cloud-based and local embedding models, offering a balance between accuracy and cost-performance.

- **Low-Latency Architecture:**\
  Efficient indexing and caching strategies maintain low response times under high concurrency.

---

## Future Enhancements

Multi-Document Querying → Expanding capabilities to allow simultaneous search across multiple PDFs.

Fine-Tuning Retrieval Models → Exploring domain-specific embedding optimizations for improved accuracy.

Contextual Retrieval → Exploring Combinational Embedding and Contextual BM25 as it has shown to reduce the top-20-chunk retrieval failure rate by 49%.according to  Anthropic.

---

## Conclusion

This project demonstrates a comprehensive approach to building a scalable and efficient PDF RAG chatbot. The simpler prototype provided a robust foundation, while the advanced model offers additional customizability and fine-tuning capabilities. Both versions illustrate my understanding of RAG systems and my commitment to continuous improvement.
