"""
マルチモーダル検索スクリプト

テキストクエリや画像クエリを使用して、マルチモーダルな類似検索を実行します。
- テキストクエリによる画像検索
- 画像クエリによるテキスト検索
- テキストクエリによるテキスト検索
- 画像クエリによる画像検索
"""

import weaviate
import base64
import json
from typing import List, Optional


def encode_image_to_base64(image_path: str) -> str:
    """画像ファイルをbase64エンコードする"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def search_by_text(
    collection_name: str,
    query_text: str,
    limit: int = 5,
    return_properties: Optional[List[str]] = None
):
    """
    テキストクエリによる検索
    
    Args:
        collection_name: コレクション名
        query_text: 検索クエリテキスト
        limit: 返却する結果の数
        return_properties: 返却するプロパティのリスト（Noneの場合はすべて）
    
    Returns:
        検索結果のリスト
    """
    with weaviate.connect_to_local() as client:
        if not client.collections.exists(collection_name):
            print(f"✗ コレクション '{collection_name}' が存在しません。")
            return []
        
        collection = client.collections.get(collection_name)
        
        if return_properties is None:
            return_properties = ["text", "image", "metadata"]
        
        response = collection.query.near_text(
            query=query_text,
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            return_properties=return_properties,
        )
        
        results = []
        for obj in response.objects:
            result = {
                "uuid": str(obj.uuid),
                "distance": obj.metadata.distance if obj.metadata else None,
                "properties": obj.properties,
            }
            results.append(result)
        
        return results


def search_by_image(
    collection_name: str,
    image_path: str,
    limit: int = 5,
    return_properties: Optional[List[str]] = None
):
    """
    画像クエリによる検索
    
    Args:
        collection_name: コレクション名
        image_path: 検索に使用する画像ファイルのパス
        limit: 返却する結果の数
        return_properties: 返却するプロパティのリスト（Noneの場合はすべて）
    
    Returns:
        検索結果のリスト
    """
    with weaviate.connect_to_local() as client:
        if not client.collections.exists(collection_name):
            print(f"✗ コレクション '{collection_name}' が存在しません。")
            return []
        
        collection = client.collections.get(collection_name)
        
        # 画像をbase64エンコード
        image_base64 = encode_image_to_base64(image_path)
        
        if return_properties is None:
            return_properties = ["text", "image", "metadata"]
        
        response = collection.query.near_image(
            near_image=image_base64,
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            return_properties=return_properties,
        )
        
        results = []
        for obj in response.objects:
            result = {
                "uuid": str(obj.uuid),
                "distance": obj.metadata.distance if obj.metadata else None,
                "properties": obj.properties,
            }
            results.append(result)
        
        return results


def print_search_results(results: List[dict], query_type: str):
    """検索結果を整形して表示"""
    print(f"\n{'='*60}")
    print(f"{query_type} の検索結果 ({len(results)}件)")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n結果 {i}:")
        print(f"  UUID: {result['uuid']}")
        print(f"  距離: {result['distance']:.4f}" if result['distance'] is not None else "  距離: N/A")
        
        props = result.get('properties', {})
        if 'text' in props:
            text = props['text']
            # 長いテキストは切り詰め
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"  テキスト: {text}")
        
        if 'metadata' in props and props['metadata']:
            try:
                metadata = json.loads(props['metadata'])
                print(f"  メタデータ: {json.dumps(metadata, ensure_ascii=False, indent=4)}")
            except:
                print(f"  メタデータ: {props['metadata']}")
        
        if 'image' in props:
            image_len = len(props['image']) if props['image'] else 0
            print(f"  画像: base64エンコード済み ({image_len}文字)")


def demo_multimodal_search():
    """マルチモーダル検索のデモ"""
    collection_name = "MultimodalData_CLIP_ViT_B_32"  # 使用するコレクション名
    
    print("マルチモーダル検索のデモを実行します...")
    print(f"使用コレクション: {collection_name}\n")
    
    # 1. テキストクエリによる検索
    print("\n1. テキストクエリによる検索")
    print("   クエリ: 'beautiful sunset'")
    text_results = search_by_text(
        collection_name=collection_name,
        query_text="beautiful sunset",
        limit=3
    )
    print_search_results(text_results, "テキストクエリ")
    
    # 2. 画像クエリによる検索（画像ファイルのパスが必要）
    # print("\n2. 画像クエリによる検索")
    # print("   クエリ画像: 'path/to/query_image.jpg'")
    # image_results = search_by_image(
    #     collection_name=collection_name,
    #     image_path="path/to/query_image.jpg",
    #     limit=3
    # )
    # print_search_results(image_results, "画像クエリ")
    
    print("\n" + "="*60)
    print("検索完了！")
    print("="*60)


def compare_models(query_text: str, image_path: Optional[str] = None):
    """
    複数のモデルで同じクエリを実行して結果を比較
    
    Args:
        query_text: テキストクエリ
        image_path: 画像クエリのパス（オプション）
    """
    collections = [
        "MultimodalData_CLIP_ViT_B_32",
        "MultimodalData_Qwen_VL",
        "MultimodalData_Cohere_Vision",
    ]
    
    print(f"\n{'='*60}")
    print("複数モデルの比較検索")
    print(f"{'='*60}")
    print(f"クエリ: {query_text}")
    if image_path:
        print(f"画像: {image_path}")
    print()
    
    for collection_name in collections:
        print(f"\n--- {collection_name} ---")
        
        if image_path:
            results = search_by_image(
                collection_name=collection_name,
                image_path=image_path,
                limit=3
            )
        else:
            results = search_by_text(
                collection_name=collection_name,
                query_text=query_text,
                limit=3
            )
        
        if results:
            print(f"  検索結果: {len(results)}件")
            for i, result in enumerate(results[:2], 1):  # 上位2件を表示
                props = result.get('properties', {})
                text = props.get('text', 'N/A')
                if len(text) > 50:
                    text = text[:50] + "..."
                print(f"    {i}. {text} (距離: {result['distance']:.4f})")
        else:
            print("  検索結果なし")


if __name__ == "__main__":
    # デモ実行
    demo_multimodal_search()
    
    # モデル比較の例（コメントアウト）
    # compare_models("beautiful sunset")
    # compare_models("", image_path="path/to/query_image.jpg")

