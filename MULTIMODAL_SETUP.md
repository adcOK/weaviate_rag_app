# マルチモーダルベクトルデータベース構築ガイド

このドキュメントでは、Weaviateを使用して画像とテキストを同一ベクトル空間で扱えるマルチモーダルなベクトルストアを構築する方法を説明します。

## 概要

このシステムでは、以下の要件を満たすマルチモーダルベクトルデータベースを構築します：

- ローカルでDBが動作すること
- 画像とテキストをベクトル化（Embedding）して、ベクトルストアに格納できること
- 画像とテキストは同一のベクトル空間に配置され、マルチモーダルなEmbeddingモデルを使用すること
- RAG（Retrieval-Augmented Generation）として、マルチモーダルな類似検索が可能であること
- ベクトル化のモデルを複数試せるように、DBのスキーマを構成すること
- OpenAIやGemini APIなどを使用せずに、RAGを含めた回答生成が可能であること

## 技術スタック

- **Weaviate**: ベクトルデータベース
- **HuggingFace Transformers**: マルチモーダルEmbeddingモデル（Qwen-VLなど）
- **CLIP**: マルチモーダルEmbeddingモデル（デフォルト）
- **Cohere Vision**: マルチモーダルEmbeddingモデル（外部API）
- **Ollama**: ローカルLLM（回答生成用）

## セットアップ手順

### 1. 前提条件

- DockerとDocker Composeがインストールされていること
- NVIDIA GPU（オプション、GPU使用時）
- Python 3.8以上

### 2. Docker Composeの起動

```bash
docker-compose up -d
```

これにより、以下のサービスが起動します：

- `weaviate`: メインのWeaviateサービス（ポート8080）
- `multi2vec-clip`: CLIPモデル用サービス
- `multi2vec-transformers-qwen`: HuggingFace Transformers用サービス（Qwen-VLなど）
- `text2vec-transformers`: テキスト用Transformerサービス
- `reranker-transformers`: 検索結果の再ランキングサービス
- `ollama`: Ollama LLMサービス（ポート11434）

### 3. Python環境のセットアップ

```bash
# 仮想環境の作成（既に存在する場合はスキップ）
python -m venv venv

# 仮想環境の有効化
# Windows PowerShellの場合
.\venv\Scripts\Activate.ps1

# 必要なパッケージのインストール
pip install weaviate-client
```

### 4. コレクション（スキーマ）の作成

```bash
python sample_code/create_multimodal_schema.py
```

このスクリプトは以下のコレクションを作成します：

- `MultimodalData_CLIP_ViT_B_32`: CLIP ViT-B-32モデルを使用
- `MultimodalData_Qwen_VL`: Qwen-VLモデルを使用（HuggingFace Transformers）
- `MultimodalData_Cohere_Vision`: Cohere Vision（embed-v4.0）を使用

**注意**: Cohere Visionを使用する場合は、`docker-compose.yml`の`COHERE_APIKEY`環境変数を設定してください。

## 使用方法

### 1. データのインポート

```bash
python sample_code/import_multimodal_data.py
```

**注意**: スクリプト内の`create_sample_data()`関数を編集して、実際の画像ファイルのパスを指定してください。

```python
sample_data[0]["image_path"] = "path/to/sunset.jpg"
sample_data[1]["image_path"] = "path/to/city.jpg"
sample_data[2]["image_path"] = "path/to/cat.jpg"
```

### 2. マルチモーダル検索

```bash
python sample_code/multimodal_search.py
```

このスクリプトでは以下の検索が可能です：

- **テキストクエリによる画像検索**: `search_by_text()`関数を使用
- **画像クエリによるテキスト検索**: `search_by_image()`関数を使用
- **複数モデルの比較**: `compare_models()`関数を使用

### 3. マルチモーダルRAG

```bash
python sample_code/multimodal_rag.py
```

このスクリプトでは、検索結果を基にOllamaを使用して回答を生成します。

**注意**: Ollamaモデル（例：`llama3.2`）がインストールされていることを確認してください。

```bash
# Ollamaモデルのインストール例
ollama pull llama3.2
```

## コレクション構成の詳細

### CLIP ViT-B-32コレクション

- **ベクトライザー**: `multi2vec-clip`（デフォルトサービス）
- **特徴**: ローカル完結、高速
- **使用例**: 一般的なマルチモーダル検索

### Qwen-VLコレクション

- **ベクトライザー**: `multi2vec-clip`（カスタムtransformers-inferenceサービス）
- **特徴**: HuggingFace Transformersを使用、ローカル完結
- **使用例**: 日本語対応や特定ドメインに特化した検索

### Cohere Visionコレクション

- **ベクトライザー**: `text2vec-cohere`（embed-v4.0）
- **特徴**: 外部API使用、高性能
- **使用例**: 高精度が求められる検索

## HuggingFace Transformersモデルの追加方法

新しいHuggingFaceモデルを試す場合は、以下の手順を実行します：

1. `docker-compose.yml`に新しい`transformers-inference`サービスを追加：

```yaml
multi2vec-transformers-new-model:
  image: cr.weaviate.io/semitechnologies/transformers-inference:YourModel/ModelName
  ports:
    - 8082:8080  # 既存のサービスと競合しないポート番号
  environment:
    ENABLE_CUDA: '1'
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

2. `create_multimodal_schema.py`に新しいコレクション作成コードを追加：

```python
collection_name_new = "MultimodalData_NewModel"
if not client.collections.exists(collection_name_new):
    client.collections.create(
        name=collection_name_new,
        description="新しいモデルを使用したマルチモーダルデータ",
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="image", data_type=DataType.BLOB),
            Property(name="metadata", data_type=DataType.TEXT),
        ],
        vector_config=Configure.Vectors.multi2vec_clip(
            image_fields=["image"],
            text_fields=["text"],
            inference_url="http://multi2vec-transformers-new-model:8080",
        ),
    )
```

## トラブルシューティング

### GPUが認識されない場合

1. NVIDIAドライバーがインストールされているか確認：
   ```bash
   nvidia-smi
   ```

2. NVIDIA Container Toolkitがインストールされているか確認：
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### コレクションが作成できない場合

- Weaviateサービスが起動しているか確認：
  ```bash
  docker-compose ps
  ```

- ログを確認：
  ```bash
  docker-compose logs weaviate
  ```

### Ollamaモデルが見つからない場合

- モデルがインストールされているか確認：
  ```bash
  ollama list
  ```

- モデルをインストール：
  ```bash
  ollama pull llama3.2
  ```

## 参考リンク

- [Weaviate公式ドキュメント](https://docs.weaviate.io/)
- [Weaviate Python Client](https://github.com/weaviate/weaviate-python-client)
- [Weaviate Transformers統合](https://docs.weaviate.io/weaviate/model-providers/transformers)
- [Weaviate マルチモーダルEmbedding](https://docs.weaviate.io/weaviate/model-providers/transformers/embeddings-multimodal)
- [Ollama公式サイト](https://ollama.ai/)

