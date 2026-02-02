# Image Search Engine
An image search engine built on Elasticsearch using HNSW for fast vector similarity search. Images from an Instagram-like dataset are embedded with [google/siglip-base-patch16-512](https://huggingface.co/google/siglip-base-patch16-512) and indexed as dense vectors, enabling efficient k-NN search for visually similar images.

## Performance Results

| Metric | Value |
|--------|-------|
| Mean Recall@10 | 99.70% |
| Mean ES(KNN+HNSW) Latency | 51.53 ms |
| Mean KNN Latency | 84.29 ms |
| Total TEST Queries | 3,019 |

*Results from evaluation comparing Elasticsearch (with HNSW) vs classic KNN on 3,019 test queries against 27,172 indexed images.*

<details>
<summary>Scope & Assumptions</summary>

### Scope Clarification

This prototype focuses exclusively on **image-based similarity search**.

The initial system design considered a multimodal setup (image + text) using models such as **Qwen3-VL**. However, the implementation was intentionally constrained to image-only retrieval in order to reduce system complexity within a time-limited prototype.

Crucially, image embeddings use the same indexing, storage, and retrieval infrastructure that would be required for a multimodal system. Therefore, restricting the prototype to images validates the **core retrieval architecture** while demonstrating that additional modalities can be integrated later without structural changes.

---

### Assumptions

The following assumptions define the boundaries under which the system is designed and evaluated. They remove ambiguity regarding how similarity is defined, how retrieval quality is measured, how results are returned, and which operational constraints apply.

* **Assumption 1 — Embedding validity**: Distances in the learned embedding space correspond to the client’s notion of similarity. Formally, given a query image $X$ and a candidate set of images $N$, ranking images in $N$ by embedding-space similarity to $X$ corresponds to the client’s perceived similarity ranking.
* **Assumption 2 — Ground truth for evaluation**: Labeled datasets are often unavailable at cold start. Therefore, explicit relevance labels cannot be used for evaluation, and results from exact k-nearest neighbor (KNN) search over the full dataset are treated as a ground truth.
* **Assumption 3 — Fixed-size retrieval**: For each query image, the system returns a fixed number $n_{\text{similar}}$ of most similar items.
* **Assumption 4 — Data confidentiality**: Input data is assumed to be confidential. Input images and their embeddings are not persistently stored or indexed beyond what is strictly necessary for immediate computation. Any temporary storage used for processing is ephemeral, and no long-term saving of raw images or embeddings occurs. This ensures compliance with privacy constraints and allows future extension if selective storage is introduced under controlled conditions.

</details>

<details>
<summary>System Architecture (Embedder, DB, Storage)</summary>

### Embedder: Why SigLIP was used
The model **`siglip-base-patch16-512`** was selected due to its strong performance on public benchmarks (e.g., [MTEB](https://huggingface.co/spaces/mteb/leaderboard)) combined with a relatively small RAM and VRAM footprint. This enables stable, reproducible inference on limited hardware.

The `SigLIPEmbedder` is implemented using **dependency injection**, allowing the embedding component to be replaced with alternative models without requiring changes to the indexing or retrieval pipeline.

### Vector Database: Why Elasticsearch
Elasticsearch is used as the retrieval engine because it supports **Approximate Nearest Neighbor (ANN)** search via **HNSW**, providing sublinear query complexity $O(\log N)$ in practice. Elasticsearch supports keyword search and metadata filtering, and its architecture can be easily extended to multimodal retrieval, enabling combination of image, text, and other feature embeddings within the same search framework.

### Why ANN instead of exact KNN
Exact KNN requires computing distances between the query embedding and all indexed items, resulting in $O(N)$ complexity per query. ANN methods such as HNSW trade a small amount of accuracy for orders-of-magnitude faster retrieval.

### Image Storage: Why S3
Amazon S3 is used to store raw image assets because it provides scalable, durable, and cost-efficient object storage. Storing images outside the search engine keeps the indexing layer lightweight.

</details>

<details>
<summary>Evaluation Metric: Recall@K</summary>

**Recall@K** measures the overlap between the results of an Approximate Nearest Neighbor (ANN) search and the exact Nearest Neighbors (KNN).

### Definition

For a given query image, let:
- $$GT_{[:n_{similar}]}$$ be the top $$n_{similar}$$ neighbors from exact KNN.
- $$Pred_{[:n_{similar}]}$$ be the top $$n_{similar}$$ neighbors returned by the ANN index.
- $$|GT|$$ be the total number of ground truth items.

$$Recall@n_{similar} = \frac{|GT_{[:n_{similar}]} \cap Pred_{[:n_{similar}]}|}{\min(n_{similar}, |{GT}|)}$$
</details>
<details>
<summary>Setup</summary>

**Note: tested against Python 3.12**
1. Cloning repository:
    ```bash 
    git clone https://github.com/EvelinaAleksiutenko/strv-test
    ```

2. Creating venv:
    
    a) Using Pipenv (Recommended)

    ```bash
    pip install pipenv
    pipenv install
    pipenv shell
    pipenv run pip install -r requirements.txt 
    ```
    
    b) Using requirements.txt

    ```bash
    pip install -r requirements.txt
    ```
3.  Pulling ES docker image:
    ```bash
    docker network create elastic
    docker pull docker.elastic.co/elasticsearch/elasticsearch:9.2.4
    docker run --name es01 --net elastic -p 9200:9200 -it -m 1GB docker.elastic.co/elasticsearch/elasticsearch:9.2.4
    ```
4. Installing the dataset:

    Note: project_dir - root directory of the project
    ```bash
    git lfs install
    mkdir project_dir / "datasets/instagram"
    cd project_dir / "datasets/instagram"
    git clone https://huggingface.co/datasets/mrSoul7766/instagram_post_captions
    ```

5. Configuration:

   Create a `.env` file:

    ```env
    ELASTICSEARCH_HOST=localhost
    ELASTICSEARCH_PORT=9200
    ELASTICSEARCH_PASSWORD=your_password
    ELASTICSEARCH_INDEX=image_embeddings
    S3_BUCKET_NAME=your-bucket
    AWS_REGION_NAME=your-region
    API_BASE_URL=your-api-base-url
## Running
### Preparation of the dataset
After placing the dataset in the correct location(project_dir / "datasets/instagram/instagram_post_captions), run the dataset preparation notebook for shuffling and splitting the data:

```bash
jupyter notebook notebooks/prepare_dataset.ipynb
```
This notebook will process the raw dataset and create the necessary train/test splits with UUIDs assigned to each image.

The expected structure should be:

```
datasets/
└── instagram/
    ├── images/
    ├── instagram_post_captions/
    │   └── data/
    │       ├── train/
    │       └── test/
```
**Note**: The term "train data" in this pipeline refers to the non-test portion of the dataset - we are not training the SigLIP model (using pre-trained embeddings), but we are building the search index with this data for KNN similarity search.
### Pipeline

Process all training data, save locally, upload to S3, generate SigLIP embeddings, and index to Elasticsearch:
![IMG_2289](https://github.com/user-attachments/assets/c5830e9c-e24d-4f1e-9a34-93938b714a50)

```bash
python setup.py
```

### FastAPI
![IMG_2287](https://github.com/user-attachments/assets/32447d24-6b33-4aef-bbb7-913eb9296e7b)
```bash
uvicorn app:app --reload
``` 
Result available at: API_BASE_URL

### Streamlit
![IMG_2286](https://github.com/user-attachments/assets/f5206f2c-fb44-477a-b971-f729dbf66003)

```bash
streamlit run demo.py
```
Results available at: http://localhost:8501

### Evaluation

Run evaluation with comparison between classic KNN and KNN+HNSW:

```bash
python eval.py
```

**Note**: The specific model used can be found in `MODEL_NAME` in [src/image/embedder/siglip_embedder.py](src/image/embedder/siglip_embedder.py).
