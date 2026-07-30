*This project has been created as part of the 42 curriculum by yanlu.*

# RAG against the machine

## Description
This project builds a **Retrieval-Augmented Generation (RAG)** system that answers questions about the codebase of vLLM (v0.10.1) by retrieving relevant information and generating evidence-based responses, implementing intelligent chunking, efficient retrieval.

Instead of retraining a model, RAG augments a model with an external source at inference time.
In practice, RAG has four stages:

- **Indexing**: organise the data so it can be searched.
- **Retrieving**: match a question against the index and pull the most relevant snippets.
- **Augmenting**: filter those snippets and place them in the model’s context window.
- **Generating**: read that context and produce the answer.

### System architecture
Describe your RAG pipeline components and how they
interact

### Chunking strategy
Split the files into chunks and store them as index that can be queried.

Two distinct chunking strategies are implemented for the two types of files:
- Python code chunking
- Markdown/text chunking

Chunk size is configurable through a CLI argument (`--max_chunk_size`), with a default of 2000 characters.

The index is stored under `data/processed/`.

### Retrieval method: BM25
Search the index and returns the top-k most relevant chunks to the prompt. Each result is a source location (a file path).

The result is stored as a JSON file under ``.

### Performance analysis
Discuss recall@k scores and system performance

### Design decisions
Explain key implementation choices
#### BM25 vs. TF-IDF

### Challenges faced
Document difficulties encountered and solutions

## Instructions
This project uses `uv` for dependency management and a `Makefile` to automate common tasks.

### Prerequisites
- Python 3.10+
- uv 0.12.0+

### Install project dependencies

    make install

### Run with default values

    make
    # or
    make run    

### CLI arguments
|Commands|Description|
|--------|-----------|
|index| Index the codebase|
|search| Search for relevant chunks|
|answer| Anwer the prompt based on retrieved chunks|
|evaluate| 

|Flag|Description|Default value|Valid range|
|----|-----------|-------------|-----------|
|--max_chunk_size|Maximum size of chunking.|2000|100-2000|
|--k|The top-k most relevant sources to retrieve.|--|>0|


### Example usage
#### Index the codebase

    # define a maximun size of chunks
    uv run python -m src index --max_chunk_size 2000

#### Search the codebase

    # single-query
    # define the the top-k most relevant sources
    uv run python -m src search "How to configure the OpenAI server?" --k 5

#### Generate answer

    # single-query
    # define the the top-k most relevant sources
    uv run python -m src answer "How to configure the OpenAI server?" --k 5

### Output format

## Resources
