# docker-compose.yml 説明書

## 概要

このdocker-compose.ymlファイルは、Weaviateベクトルデータベースとその関連サービスを起動するための設定ファイルです。Weaviateは、機械学習モデルを使用したベクトル検索とセマンティック検索を提供するデータベースです。

この構成では、以下の5つのサービスが定義されています：
- **weaviate**: メインのWeaviateサービス
- **text2vec-transformers**: Transformerモデルを使用したテキストベクトル化サービス（GPU対応）
- **reranker-transformers**: 検索結果の再ランキングサービス（GPU対応）
- **multi2vec-clip**: CLIPモデルを使用したマルチモーダルベクトル化サービス（GPU対応）
- **ollama**: Ollama LLMサービス

## サービス構成の詳細

### weaviate（メインサービス）

Weaviateのコアサービスです。

- **イメージ**: `cr.weaviate.io/semitechnologies/weaviate:1.34.0`
- **ポート**:
  - `8080`: HTTP API
  - `50051`: gRPC API
- **コマンド**: HTTPスキームで0.0.0.0:8080でリッスン
- **環境変数**:
  - `QUERY_DEFAULTS_LIMIT`: デフォルトのクエリ結果数（25件）
  - `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED`: 匿名アクセスを有効化
  - `PERSISTENCE_DATA_PATH`: データ永続化パス
  - `ENABLE_MODULES`: 有効化するモジュール（text2vec-transformers, multi2vec-clip, text2vec-ollama, reranker-transformers）
  - `TRANSFORMERS_INFERENCE_API`: text2vec-transformersサービスのエンドポイント
  - `CLIP_INFERENCE_API`: multi2vec-clipサービスのエンドポイント
  - `RERANKER_INFERENCE_API`: reranker-transformersサービスのエンドポイント
  - `OLLAMA_API_ENDPOINT`: Ollamaサービスのエンドポイント
- **ボリューム**: `weaviate_data`を`/var/lib/weaviate`にマウント
- **依存関係**: ollamaサービスに依存

### text2vec-transformers

Transformerモデルを使用したテキストベクトル化サービスです。GPUアクセラレーションに対応しています。

- **イメージ**: `cr.weaviate.io/semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1`
- **環境変数**:
  - `ENABLE_CUDA: '1'`: CUDA（GPU）を有効化
- **GPU設定**: NVIDIA GPUリソースを予約

### reranker-transformers

検索結果の再ランキングを行うサービスです。GPUアクセラレーションに対応しています。

- **イメージ**: `cr.weaviate.io/semitechnologies/reranker-transformers:cross-encoder-ms-marco-MiniLM-L-6-v2`
- **環境変数**:
  - `ENABLE_CUDA: '1'`: CUDA（GPU）を有効化
- **GPU設定**: NVIDIA GPUリソースを予約

### multi2vec-clip

CLIPモデルを使用したマルチモーダル（テキストと画像）ベクトル化サービスです。GPUアクセラレーションに対応しています。

- **イメージ**: `cr.weaviate.io/semitechnologies/multi2vec-clip:sentence-transformers-clip-ViT-B-32-multilingual-v1`
- **環境変数**:
  - `ENABLE_CUDA: '1'`: CUDA（GPU）を有効化
- **GPU設定**: NVIDIA GPUリソースを予約

### ollama

Ollama LLMサービスです。

- **イメージ**: `ollama/ollama:latest`
- **ポート**: `11434`
- **ボリューム**: `ollama_data`を`/root/.ollama`にマウント

## GPU設定の説明

このdocker-compose.ymlでは、以下の3つのサービスがGPUを使用するように設定されています：

1. `text2vec-transformers`
2. `reranker-transformers`
3. `multi2vec-clip`

### GPU設定の構成要素

各GPU対応サービスには以下の設定が含まれています：

```yaml
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

- **`ENABLE_CUDA: '1'`**: 環境変数でCUDAを有効化します。これにより、サービスがGPUを使用して推論を実行します。
- **`deploy.resources.reservations.devices`**: Docker Composeのリソース予約セクションです。
  - **`driver: nvidia`**: NVIDIA GPUドライバーを使用することを指定します。
  - **`count: all`**: 利用可能なすべてのGPUを使用します。
  - **`capabilities: [gpu]`**: GPU機能を要求します。

## 起動方法

### 前提条件

#### GPUを使用する場合

GPUを使用する場合は、以下の前提条件が必要です：

1. **NVIDIA GPUドライバー**: NVIDIA GPUがインストールされ、適切なドライバーがインストールされている必要があります。
2. **NVIDIA Container Toolkit**: DockerがGPUにアクセスできるようにするためのツールキットです。

##### NVIDIA Container Toolkitのインストール（Linux）

```bash
# リポジトリの追加
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# インストール
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Dockerデーモンの再起動
sudo systemctl restart docker
```

##### NVIDIA Container Toolkitのインストール（Windows）

Windowsでは、Docker Desktop for Windowsを使用している場合、WSL2経由でGPUアクセスが可能です。

1. WSL2にNVIDIAドライバーをインストール
2. Docker Desktopの設定でWSL2バックエンドを有効化

##### GPUの動作確認

起動前に、GPUがDockerから認識できることを確認します：

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

このコマンドでGPU情報が表示されれば、GPUサポートが正しく設定されています。

### 通常の起動方法（GPUなし）

GPUを使用しない場合でも、サービスは起動しますが、GPU対応サービスはCPUモードで動作します（パフォーマンスが低下します）。

```bash
docker-compose up -d
```

### GPU有効での起動方法

GPUサポートが正しく設定されていれば、そのまま起動できます：

```bash
docker-compose up -d
```

GPUが有効になっているか確認するには、各GPU対応サービスのログを確認します：

```bash
# text2vec-transformersのログを確認
docker-compose logs text2vec-transformers

# reranker-transformersのログを確認
docker-compose logs reranker-transformers

# multi2vec-clipのログを確認
docker-compose logs multi2vec-clip
```

ログにCUDA関連のメッセージが表示されていれば、GPUが正しく使用されています。

### サービスの停止

```bash
docker-compose down
```

データを保持したまま停止する場合は上記のコマンドで、データも削除する場合は：

```bash
docker-compose down -v
```

## 環境変数と設定

### 主要な環境変数

#### weaviateサービス

| 環境変数 | 説明 | デフォルト値 |
|---------|------|------------|
| `QUERY_DEFAULTS_LIMIT` | デフォルトのクエリ結果数 | 25 |
| `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED` | 匿名アクセスを有効化 | true |
| `PERSISTENCE_DATA_PATH` | データ永続化パス | /var/lib/weaviate |
| `ENABLE_MODULES` | 有効化するモジュール | text2vec-transformers,multi2vec-clip,text2vec-ollama,reranker-transformers |

#### GPU対応サービス

| 環境変数 | 説明 |
|---------|------|
| `ENABLE_CUDA` | CUDA（GPU）を有効化（'1'で有効） |

### ボリュームマウント

- **`weaviate_data`**: Weaviateのデータを永続化するためのボリューム（`/var/lib/weaviate`にマウント）
- **`ollama_data`**: Ollamaのモデルとデータを保存するためのボリューム（`/root/.ollama`にマウント）

## トラブルシューティング

### GPUが認識されない場合

1. **NVIDIAドライバーの確認**
   ```bash
   nvidia-smi
   ```
   このコマンドでGPU情報が表示されない場合は、ドライバーが正しくインストールされていません。

2. **NVIDIA Container Toolkitの確認**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```
   このコマンドでGPU情報が表示されない場合は、NVIDIA Container Toolkitが正しく設定されていません。

3. **Docker Composeのバージョン確認**
   Docker Compose v2を使用していることを確認します：
   ```bash
   docker compose version
   ```
   v1を使用している場合は、`deploy.resources`セクションがサポートされていない可能性があります。

4. **サービスのログ確認**
   ```bash
   docker-compose logs text2vec-transformers
   ```
   エラーメッセージを確認し、CUDA関連のエラーがないかチェックします。

### よくある問題と解決方法

#### 問題: GPUサービスが起動しない

**解決方法**:
- GPUドライバーとNVIDIA Container Toolkitが正しくインストールされているか確認
- Dockerデーモンを再起動: `sudo systemctl restart docker`（Linux）

#### 問題: ポートが既に使用されている

**解決方法**:
- 使用中のポートを確認: `netstat -tulpn | grep <ポート番号>`
- docker-compose.ymlのポート番号を変更するか、競合しているプロセスを停止

#### 問題: ボリュームの権限エラー

**解決方法**:
- ボリュームの所有者を確認し、必要に応じて権限を変更
- または、ボリュームを削除して再作成: `docker-compose down -v`

#### 問題: サービス間の通信エラー

**解決方法**:
- すべてのサービスが起動しているか確認: `docker-compose ps`
- ネットワーク設定を確認: `docker network ls`
- サービス名が正しく解決されているか確認（Docker Composeの内部DNSを使用）

## 参考リンク

- [Weaviate公式ドキュメント](https://weaviate.io/developers/weaviate)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Docker Compose公式ドキュメント](https://docs.docker.com/compose/)

