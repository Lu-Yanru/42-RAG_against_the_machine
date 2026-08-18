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
Files are split into chunks and stored as index that can be queried.

The index is stored under `data/processed`.

Max chunk size is configurable through a CLI argument (`--max_chunk_size`), with a default of 2000 characters.

Two distinct chunking strategies are implemented for the two types of files:

#### Text chunking
Plain text files are first splitted into paragraphs (separated by an empty newline). If a paragraph is shorter than `max_chunk_size`, check if one can combine the next paragraph without exceeding `max_chunk_size` before starting a new chunk. If a paragraph is longer than `max_chunk_size`, split the paragraph by line and apply the same greedy packing. If a line is longer than `max_chunk_size`, fall back to a raw sliding window and log a warning suggesting the `max_chunk_size` might be too small. 

For Markdown files, split the file first by sections marked by `#` section headings and apply greedy packing. If a section is longer than `max_chunk_size`, chunk it further as plain text.

#### Python code chunking
Python code is chunked based on the Abstract Syntax Tree (AST). Chunks are split at class `ClassDef` or function (`FunctionDef`/`AsyncFunctionDef`) level.

If one of the class or function exceed `max_chunk_size`, then spilt it further as plain text. 

### Retrieval method
This project implements lexical, semantic and hybrid retrieval.

#### Lexical (sparse) retrieval
The **Best Match 25 (BM25)** algorithm is used to search the index and returns the top-k most relevant chunks to the prompt. Each result is a source location (a file path). The result is stored as a JSON file under `data/output/search_results`.

BM25 is a statistical ranking function based on term frequency (TF) and inverse document frequency (IDF).

#### Semantic (dense) retrieval
This project also uses the sentence-transformers model `all-MiniLM-L6-v2` to index and retrieve semantically relevant chunks to the prompt. The sentence-transformers model maps sentences & paragraphs to a 384 dimensional dense vector space and finds the chunk with the most similar vectors to the prompt.

#### Hybrid
**Reciprocal Rank Fusion (RRF)** is a rank aggregation methods that combines rankings from multiple sources into a single, unified ranking. It combines rankings, not scores, so it needs no score normalization.

### Performance analysis
The retrieval quality is measured with the **recall@k** metric.
For each question, recall@k is the share of its correct sources that you retrieve in your
top-k results. A correct source counts as found when one of your results is in the same
file and overlaps its character range (IoU >= 0.05).

#### Lexical retrieval
|Metric | Target | Status |
|-------|--------|--------|
|Indexing time | <= 5 min |✅ < 1 min|
|Retrieval throughput| <= 90s for 200 questions | ✅ < 2s|
|Recall@5 for docs | >= 80% | ✅ 84.0%|
|Recall@5 for code | >= 50% | ✅ 50.5%|
 
#### Semantic retrieval
|Metric | Status |
|-------|--------|
|Indexing time | < 2 min|
|Retrieval throughput| < 8s for 200 questions|
|Recall@5 for docs | 56.0%|
|Recall@5 for code | 35.4%|

#### Hybrid retrieval
|Metric | Status |
|-------|--------|
|Indexing time | < 2 min|
|Retrieval throughput| < 8s for 200 questions|
|Recall@5 for docs | 81.0%|
|Recall@5 for code | 51.5%|


### Design decisions

#### BM25 vs. TF-IDF
The **BM25** algorithm is chosen over the **Term Frequency–Inverse Document Frequency (TF-IDF)** approach, because it is an improved version of TF-IDF.

- It measures term frequency and document relevance more accurately.
- It accounts for document length normalization, giving fair weight to all documents.
- It helps to deliver more relevant search results based on keyword matching and context.

#### Incremental index
This project implements **incremental chunking** and **incremental indexing** for semantic indexing.
When a file changes, re-chunk and re-index only that file instead of rebuilding the whole index.

Incremental indexing is not implemented for lexical indexing, because the `bm25s` library does not support incremental indexing, but instead achieve its speed by precomputing all possible term-document BM25 scores at index time. Based on my performance test, indexing the whole corpus with `bm25s` only takes less than 1 min, which is well less than the target 5 min. Therefore, I don't see the need to switch to another library jsut to implement incremental lexical indexing.

#### Caching
The results of both indexing and retrieval are cached in JSON files stored in `data/processed`. The former is stored as a list of file paths and hashes of their content, the latter is stored as a list of query hashes and a list of the retrieved entries. On the next run of `index` or `search` command, the program checks whether any of the hashes changed, load the results from the unchanges entries first, and only run indexing or search for the changed entries. This saves time for rerunning the commands.

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
|Commands|Description|Flags| Flag description|Flag default value|Flag valid range|
|--------|-----------|-----|-----------------|------------------|----------------|
|index| Index the codebase|--max_chunk_size|Maximum size of chunking|2000|100-2000|
|||--method| The method of indexing| lexical| lexical, semantic, hybrid|
|search| Search for relevant chunks for a single prompt|--k|The top-k most relevant sources to retrieve.|--|positive integer|
|||--method| | | |
|search_dataset| Search for relevant chunks for a whole dataset|--dataset_path | The path of the dataset|data/datasets/UnansweredQuestions/dataset_docs_public.json|Valid JSON file|
|||--k||||
|||--save_directory|The path to save the search result|data/output/search_results/UnansweredQuestions|--|
|||--method| | | |
|answer| Anwer a single prompt prompt based on retrieved chunks|--k||||
|||--method| | | |
|answer_dataset| Generate anwers for a whole dataset|--student_search_results_path|The path where the search results are stored|data/output/search_results|--|
|||--save_directory|The path to save the answers|data/output/seach_results_and_answer|--|
|evaluate| Evaluate the retrival quality based on recall@k metrics|--student_search_results_path||||
|||--dataset_path||||

### Example usage
#### Index the codebase

    # define a maximun size of chunks
    uv run python -m src index --max_chunk_size 2000

#### Search the codebase

    # single-query
    # define the the top-k most relevant sources
    uv run python -m src search "How to configure the OpenAI server?" --k 5

    # batch query
    uv run python -m src search_dataset \
        --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
        --k 5 \
        --save_directory data/output/search_results/UnansweredQuestions

#### Generate answer

    # single-query
    # define the the top-k most relevant sources
    uv run python -m src answer "How to configure the OpenAI server?" --k 5

    # batch query
    uv run python -m src answer_daraset \
        --student_search_results_path  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
        --save_directory data/output/search_results_and_answer/UnansweredQuestions

#### Evaluate

    uv run python -m src evaluate \
        --student_search_results_path data/output/search_results \
        --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json

    # Score with moulinette
    ./moulinette evaluate_student_search_results \
        data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
        data/datasets/AnsweredQuestions/dataset_docs_public.json \
        --k 10 --max_context_length 2000

### Output format
The results are saved as JSON files in the following format:

#### Example search result

    "search_results": [
        {
            "question_id": "q1",
            "question": "How to configure OpenAI server?",
            "retrieved_sources": [
                {
                    "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
                    "first_character_index": 9867,
                    "last_character_index": 10100
                },
                {
                    "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
                    "first_character_index": 267,
                    "last_character_index": 400
                }
            ]
        }
    ],
    "k": 10

#### Example answer output

    "search_results": [
            {
                "question_id": "q1",
                "question": "How to configure OpenAI server?",
                "retrieved_sources": [
                {
                    "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
                    "first_character_index": 9867,
                    "last_character_index": 10100
                },
                {
                    "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
                    "first_character_index": 267,
                    "last_character_index": 400
                    }
                ],
                "answer": "To configure the OpenAI compatible server in vLLM..."
            }
        ],
    "k": 10

## Resources
- [Python Fire library documentation](https://github.com/google/python-fire)
- [PyTorch library documentation](https://docs.pytorch.org/docs/2.13/index.html)
- [Transformers library documentation](https://huggingface.co/docs/transformers/index)
- [GeeksforGeeks What is BM25 algorithm?](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/)
- [BM25S: Accelerated Sparse BM25 Retrieval](https://www.emergentmind.com/topics/bm25s)
- [Intersection over union (IoU) calculation for evaluating an image segmentation model](https://medium.com/data-science/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686)
- [Evaluation Metrics for Search and Recommendation Systems](https://weaviate.io/blog/retrieval-evaluation-metrics)
- [Documentation all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Sparse vs Dense vs Hybrid Retrieval: BM25, BERT, and Reranking Compared](https://www.abhik.ai/concepts/embeddings/sparse-vs-dense)
- [Reciprocal Rank Fusion (RRF) explained in 4 mins — How to score results form multiple retrieval methods in RAG](https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a)

AI is used to help create tests and debug the code, explain RAG concepts in simple language with examples and explain documentations of the `bm25s`, `PyTorch` and `transformers` libraries. 
