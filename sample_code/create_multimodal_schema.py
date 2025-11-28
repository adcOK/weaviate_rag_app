"""
マルチモーダルベクトルストアのスキーマ作成スクリプト

複数のEmbeddingモデル（CLIP、Qwen-VL、Cohere Vision）用のコレクションを作成します。
各モデルごとに別々のコレクションを作成することで、同じデータで異なるモデルの性能を比較できます。
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType


def create_multimodal_collections():
    """複数のマルチモーダルコレクションを作成"""
    
    with weaviate.connect_to_local() as client:
        
        # 1. CLIP ViT-B-32用コレクション（デフォルトのmulti2vec-clipサービス使用）
        collection_name_clip = "MultimodalData_CLIP_ViT_B_32"
        if not client.collections.exists(collection_name_clip):
            client.collections.create(
                name=collection_name_clip,
                description="CLIP ViT-B-32モデルを使用したマルチモーダルデータ",
                properties=[
                    Property(name="text", data_type=DataType.TEXT, description="テキストデータ"),
                    Property(name="image", data_type=DataType.BLOB, description="画像データ（base64エンコード）"),
                    Property(name="metadata", data_type=DataType.TEXT, description="追加メタデータ（JSON文字列）"),
                ],
                vector_config=Configure.Vectors.multi2vec_clip(
                    image_fields=["image"],
                    text_fields=["text"],
                    # inference_urlを指定しない場合はデフォルトのCLIPサービスを使用
                ),
            )
            print(f"✓ {collection_name_clip} コレクションを作成しました")
        else:
            print(f"⚠ {collection_name_clip} コレクションは既に存在します")
        
        # 2. Qwen-VL用コレクション（カスタムtransformers-inferenceサービス使用）
        collection_name_qwen = "MultimodalData_Qwen_VL"
        if not client.collections.exists(collection_name_qwen):
            client.collections.create(
                name=collection_name_qwen,
                description="Qwen-VLモデルを使用したマルチモーダルデータ（HuggingFace Transformers）",
                properties=[
                    Property(name="text", data_type=DataType.TEXT, description="テキストデータ"),
                    Property(name="image", data_type=DataType.BLOB, description="画像データ（base64エンコード）"),
                    Property(name="metadata", data_type=DataType.TEXT, description="追加メタデータ（JSON文字列）"),
                ],
                vector_config=Configure.Vectors.multi2vec_clip(
                    image_fields=["image"],
                    text_fields=["text"],
                    # カスタムtransformers-inferenceサービスを指定
                    # Docker Compose内ではサービス名でアクセス可能
                    inference_url="http://multi2vec-transformers-qwen:8080",
                ),
            )
            print(f"✓ {collection_name_qwen} コレクションを作成しました")
        else:
            print(f"⚠ {collection_name_qwen} コレクションは既に存在します")
        
        # 3. Cohere Vision用コレクション（外部API使用）
        collection_name_cohere = "MultimodalData_Cohere_Vision"
        if not client.collections.exists(collection_name_cohere):
            try:
                client.collections.create(
                    name=collection_name_cohere,
                    description="Cohere Vision (embed-v4.0) を使用したマルチモーダルデータ",
                    properties=[
                        Property(name="text", data_type=DataType.TEXT, description="テキストデータ"),
                        Property(name="image", data_type=DataType.BLOB, description="画像データ（base64エンコード）"),
                        Property(name="metadata", data_type=DataType.TEXT, description="追加メタデータ（JSON文字列）"),
                    ],
                    vector_config=Configure.Vectors.text2vec_cohere(
                        model="embed-v4.0",  # マルチモーダル対応モデル
                    ),
                )
                print(f"✓ {collection_name_cohere} コレクションを作成しました")
            except Exception as e:
                print(f"✗ {collection_name_cohere} コレクションの作成に失敗しました: {e}")
                print("  注意: Cohere APIキーが設定されていることを確認してください")
        else:
            print(f"⚠ {collection_name_cohere} コレクションは既に存在します")
        
        print("\nすべてのコレクションの作成が完了しました！")
        print("\n作成されたコレクション:")
        collections = client.collections.list_all()
        for collection_name in collections:
            print(f"  - {collection_name}")


if __name__ == "__main__":
    print("マルチモーダルベクトルストアのスキーマを作成します...\n")
    create_multimodal_collections()

