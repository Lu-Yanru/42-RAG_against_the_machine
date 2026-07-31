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

The index is stored under `data/processed`.

### Retrieval method: BM25
Search the index and returns the top-k most relevant chunks to the prompt. Each result is a source location (a file path).

The result is stored as a JSON file under `data/output/search_results`.

### Performance analysis
The retrieval quality is measured with the **recall@k** metric.
For each question, recall@k is the share of its correct sources that you retrieve in your
top-k results. A correct source counts as found when one of your results is in the same
file and overlaps its character range.

|Metric | Target | Status |
|-------|--------|--------|
|Indexing time | <= 5 min |✅ |
|Retrieval throughput| <= 90s for 200 questions | ✅ |
|Recall@5 for docs | >= 80% | ✅ |
|Recall@5 for code | >= 50% | ✅ |
 

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
|Commands|Description|Flags| Flag description|Flag default value|Flag valid range|
|--------|-----------|-----|-----------------|------------------|----------------|
|index| Index the codebase|--max_chunk_size|Maximum size of chunking|2000|100-2000|
|search| Search for relevant chunks for a single prompt|--k|The top-k most relevant sources to retrieve.|--|positive integer|
|search_dataset| Search for relevant chunks for a whole dataset|--dataset_path | The path of the dataset|data/datasets/UnansweredQuestions/dataset_docs_public.json|Valid JSON file|
|||--k||||
|||--save_directory|The path to save the search result|data/output/search_results/UnansweredQuestions|--|
|answer| Anwer a single prompt prompt based on retrieved chunks|--k||||
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
