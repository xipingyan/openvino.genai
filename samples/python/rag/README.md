# Retrieval Augmented Generation Sample

This example showcases inference of Text Embedding and Text Rerank Models. The application has limited configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.TextEmbeddingPipeline` and `openvino_genai.TextRerankPipeline` and uses text as an input source.

## Download and Convert the Model and Tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --trust-remote-code --model BAAI/bge-small-en-v1.5 BAAI/bge-small-en-v1.5
```

Alternatively, do it in Python code:

```python
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer

output_dir = "embedding_model"

model = OVModelForFeatureExtraction.from_pretrained("BAAI/bge-small-en-v1.5", export=True, trust_remote_code=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
export_tokenizer(tokenizer, output_dir)
```

## Run

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then, run a sample:

### 1. Text Embedding Sample (`text_embeddings.py`)
- **Description:**
  Demonstrates inference of text embedding models using OpenVINO GenAI. Converts input text into vector embeddings for downstream tasks such as retrieval or semantic search.
- **Run Command:**
  ```sh
  python text_embeddings.py <MODEL_DIR> "Document 1" "Document 2"
  ```
Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#text-embeddings-models) for more details.

### 2. Text Rerank Sample (`text_rerank.py`)
- **Description:**
  Demonstrates inference of text rerank models using OpenVINO GenAI. Reranks a list of candidate documents based on their relevance to a query using a cross-encoder or reranker model.
- **Run Command:**
  ```sh
  python text_rerank.py <MODEL_DIR> "<QUERY>" "<TEXT 1>" ["<TEXT 2>" ...]
  ```


# Text Embedding Pipeline Usage

```python
import openvino_genai

pipeline = openvino_genai.TextEmbeddingPipeline(model_dir, "CPU")

embeddings = pipeline.embed_documents(["document1", "document2"])
```

# Text Rerank Pipeline Usage

```python
import openvino_genai

pipeline = openvino_genai.TextRerankPipeline(model_dir, "CPU")

rerank_result = pipeline.rerank(query, documents)
```
