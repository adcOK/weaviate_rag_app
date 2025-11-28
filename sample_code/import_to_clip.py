"""
CLIPコレクションへのデータインポートスクリプト

CLIPコレクション（MultimodalData_CLIP_ViT_B_32）にデータをインポートします。
画像ファイルはオプションです。テキストのみでもインポート可能です。
"""

import weaviate
import base64
import json
import os
from typing import List, Dict, Optional


def encode_image_to_base64(image_path: str) -> str:
    """画像ファイルをbase64エンコードする"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def import_to_clip(
    data_items: List[Dict],
    batch_size: int = 10
):
    """
    CLIPコレクションにデータをインポート
    
    Args:
        data_items: インポートするデータのリスト
                   各アイテムは {"text": str, "image_path": Optional[str], "metadata": Optional[dict]} の形式
        batch_size: バッチサイズ
    """
    collection_name = "MultimodalData_CLIP_ViT_B_32"
    
    with weaviate.connect_to_local() as client:
        if not client.collections.exists(collection_name):
            print(f"✗ コレクション '{collection_name}' が存在しません。先にスキーマを作成してください。")
            return
        
        collection = client.collections.get(collection_name)
        
        imported_count = 0
        skipped_count = 0
        
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for item in data_items:
                # 画像ファイルをbase64エンコード
                image_base64 = None
                if "image_path" in item and item["image_path"]:
                    image_path = item["image_path"]
                    if os.path.exists(image_path):
                        try:
                            image_base64 = encode_image_to_base64(image_path)
                        except Exception as e:
                            print(f"⚠ 画像ファイルの読み込みに失敗しました: {image_path} - {e}")
                    else:
                        print(f"⚠ 画像ファイルが見つかりません: {image_path}")
                
                # メタデータをJSON文字列に変換
                metadata_str = None
                if "metadata" in item and item["metadata"]:
                    try:
                        metadata_str = json.dumps(item["metadata"], ensure_ascii=False)
                    except Exception as e:
                        print(f"⚠ メタデータの変換に失敗しました: {e}")
                
                # プロパティを準備
                properties = {
                    "text": item.get("text", ""),
                }
                
                if image_base64:
                    properties["image"] = image_base64
                
                if metadata_str:
                    properties["metadata"] = metadata_str
                
                # オブジェクトを追加
                try:
                    batch.add_object(properties=properties)
                    imported_count += 1
                except Exception as e:
                    print(f"⚠ データの追加に失敗しました: {e}")
                    skipped_count += 1
        
        print(f"\n✓ {collection_name}: {imported_count}件のデータをインポートしました")
        if skipped_count > 0:
            print(f"⚠ {skipped_count}件のデータがスキップされました")


def create_sample_data() -> List[Dict]:
    """
    サンプルデータを作成
    
    注意: 画像ファイルを使用する場合は、image_pathを実際のファイルパスに変更してください
    """
    sample_data = [
        {
            "text": "A beautiful sunset over the ocean with orange and pink colors",
            "image_path": None,  # 画像ファイルのパスを指定する場合はここに記入
            "metadata": {"category": "nature", "tags": ["sunset", "ocean"]}
        },
        {
            "text": "A modern city skyline at night with many illuminated buildings",
            "image_path": None,  # 画像ファイルのパスを指定する場合はここに記入
            "metadata": {"category": "urban", "tags": ["city", "night"]}
        },
        {
            "text": "A cute cat sitting on a windowsill looking outside",
            "image_path": None,  # 画像ファイルのパスを指定する場合はここに記入
            "metadata": {"category": "animals", "tags": ["cat", "pet"]}
        },
        {
            "text": "A serene mountain landscape with snow-capped peaks",
            "image_path": None,
            "metadata": {"category": "nature", "tags": ["mountain", "snow"]}
        },
        {
            "text": "A delicious plate of sushi with fresh fish and rice",
            "image_path": None,
            "metadata": {"category": "food", "tags": ["sushi", "japanese"]}
        },
    ]
    return sample_data


if __name__ == "__main__":
    print("="*60)
    print("CLIPコレクションにデータをインポートします")
    print("="*60)
    print()
    
    # サンプルデータを作成
    sample_data = create_sample_data()
    
    # 画像ファイルのパスを指定する場合は、以下のコメントを解除して編集してください
    # sample_data[0]["image_path"] = "path/to/sunset.jpg"
    # sample_data[1]["image_path"] = "path/to/city.jpg"
    # sample_data[2]["image_path"] = "path/to/cat.jpg"
    
    print(f"インポートするデータ数: {len(sample_data)}件")
    print("（画像ファイルはオプションです。テキストのみでもインポート可能です）")
    print()
    
    try:
        import_to_clip(sample_data)
        print("\n" + "="*60)
        print("インポートが完了しました！")
        print("="*60)
        print("\n次のステップ:")
        print("1. マルチモーダル検索を実行: python sample_code/multimodal_search.py")
        print("2. RAGを実行: python sample_code/multimodal_rag.py")
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        print("\nトラブルシューティング:")
        print("1. Weaviateサービスが起動しているか確認してください")
        print("   docker-compose ps")
        print("2. コレクションが作成されているか確認してください")
        print("   python sample_code/create_clip_schema.py")

