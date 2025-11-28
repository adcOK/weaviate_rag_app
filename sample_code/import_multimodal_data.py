"""
マルチモーダルデータのインポートスクリプト

画像ファイルとテキストデータを複数のコレクションにインポートします。
画像はbase64エンコードしてblob型として格納します。
"""

import weaviate
import base64
import json
import os
from pathlib import Path
from typing import List, Dict, Optional


def encode_image_to_base64(image_path: str) -> str:
    """画像ファイルをbase64エンコードする"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def import_multimodal_data(
    collection_name: str,
    data_items: List[Dict],
    batch_size: int = 10
):
    """
    マルチモーダルデータを指定されたコレクションにインポート
    
    Args:
        collection_name: コレクション名
        data_items: インポートするデータのリスト
                   各アイテムは {"text": str, "image_path": str, "metadata": Optional[dict]} の形式
        batch_size: バッチサイズ
    """
    with weaviate.connect_to_local() as client:
        if not client.collections.exists(collection_name):
            print(f"✗ コレクション '{collection_name}' が存在しません。先にスキーマを作成してください。")
            return
        
        collection = client.collections.get(collection_name)
        
        imported_count = 0
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for item in data_items:
                # 画像ファイルをbase64エンコード
                image_base64 = None
                if "image_path" in item and item["image_path"]:
                    image_path = item["image_path"]
                    if os.path.exists(image_path):
                        image_base64 = encode_image_to_base64(image_path)
                    else:
                        print(f"⚠ 画像ファイルが見つかりません: {image_path}")
                
                # メタデータをJSON文字列に変換
                metadata_str = None
                if "metadata" in item and item["metadata"]:
                    metadata_str = json.dumps(item["metadata"], ensure_ascii=False)
                
                # プロパティを準備
                properties = {
                    "text": item.get("text", ""),
                }
                
                if image_base64:
                    properties["image"] = image_base64
                
                if metadata_str:
                    properties["metadata"] = metadata_str
                
                # オブジェクトを追加
                batch.add_object(properties=properties)
                imported_count += 1
        
        print(f"✓ {collection_name}: {imported_count}件のデータをインポートしました")


def create_sample_data() -> List[Dict]:
    """
    サンプルデータを作成（実際の使用時は適宜変更してください）
    
    注意: 画像ファイルのパスは実際のファイルパスに変更してください
    """
    sample_data = [
        {
            "text": "A beautiful sunset over the ocean with orange and pink colors",
            "image_path": None,  # 実際の画像ファイルパスに変更
            "metadata": {"category": "nature", "tags": ["sunset", "ocean"]}
        },
        {
            "text": "A modern city skyline at night with many illuminated buildings",
            "image_path": None,  # 実際の画像ファイルパスに変更
            "metadata": {"category": "urban", "tags": ["city", "night"]}
        },
        {
            "text": "A cute cat sitting on a windowsill looking outside",
            "image_path": None,  # 実際の画像ファイルパスに変更
            "metadata": {"category": "animals", "tags": ["cat", "pet"]}
        },
    ]
    return sample_data


def import_to_all_collections(data_items: List[Dict]):
    """すべてのコレクションに同じデータをインポート"""
    collections = [
        "MultimodalData_CLIP_ViT_B_32",
        "MultimodalData_Qwen_VL",
        "MultimodalData_Cohere_Vision",
    ]
    
    print("マルチモーダルデータをすべてのコレクションにインポートします...\n")
    
    for collection_name in collections:
        try:
            import_multimodal_data(collection_name, data_items)
        except Exception as e:
            print(f"✗ {collection_name}へのインポート中にエラーが発生しました: {e}")
    
    print("\nインポートが完了しました！")


if __name__ == "__main__":
    # サンプルデータを作成（実際の使用時は適宜変更）
    sample_data = create_sample_data()
    
    # 画像ファイルのパスを指定する例
    # sample_data[0]["image_path"] = "path/to/sunset.jpg"
    # sample_data[1]["image_path"] = "path/to/city.jpg"
    # sample_data[2]["image_path"] = "path/to/cat.jpg"
    
    # すべてのコレクションにインポート
    import_to_all_collections(sample_data)
    
    print("\n使用例:")
    print("1. 画像ファイルのパスを指定してデータを準備")
    print("2. import_multimodal_data()関数を使用して特定のコレクションにインポート")
    print("3. またはimport_to_all_collections()関数を使用してすべてのコレクションにインポート")

