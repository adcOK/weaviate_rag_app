"""
CLIP用のスキーマ作成スクリプト

CLIP ViT-B-32モデルを使用したマルチモーダルコレクションを作成します。
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType


def create_clip_collection():
    """CLIP用のコレクションを作成"""
    
    with weaviate.connect_to_local() as client:
        
        # CLIP ViT-B-32用コレクション（デフォルトのmulti2vec-clipサービス使用）
        collection_name_clip = "MultimodalData_CLIP_ViT_B_32"
        
        if client.collections.exists(collection_name_clip):
            print(f"⚠ {collection_name_clip} コレクションは既に存在します")
            return
        
        print(f"CLIP用コレクション '{collection_name_clip}' を作成しています...")
        
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
        print("\nコレクションの詳細:")
        collection = client.collections.get(collection_name_clip)
        print(f"  名前: {collection_name_clip}")
        print(f"  説明: CLIP ViT-B-32モデルを使用したマルチモーダルデータ")
        print(f"  プロパティ:")
        print(f"    - text (TEXT)")
        print(f"    - image (BLOB)")
        print(f"    - metadata (TEXT)")
        print(f"  ベクトライザー: multi2vec-clip (デフォルトサービス)")


if __name__ == "__main__":
    print("="*60)
    print("CLIP用のスキーマを作成します")
    print("="*60)
    print()
    
    try:
        create_clip_collection()
        print("\n" + "="*60)
        print("スキーマ作成が完了しました！")
        print("="*60)
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        print("\nトラブルシューティング:")
        print("1. Weaviateサービスが起動しているか確認してください")
        print("   docker-compose ps")
        print("2. Weaviateのログを確認してください")
        print("   docker-compose logs weaviate")

