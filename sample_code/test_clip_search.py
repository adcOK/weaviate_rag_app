"""
CLIPコレクションの検索テストスクリプト

インポートしたデータが正しく検索できるか確認します。
"""

import weaviate


def test_clip_search():
    """CLIPコレクションで検索をテスト"""
    collection_name = "MultimodalData_CLIP_ViT_B_32"
    
    with weaviate.connect_to_local() as client:
        if not client.collections.exists(collection_name):
            print(f"✗ コレクション '{collection_name}' が存在しません。")
            return
        
        collection = client.collections.get(collection_name)
        
        # コレクション内のデータ数を確認
        total_count = collection.aggregate.over_all(total_count=True).total_count
        print(f"コレクション内のデータ数: {total_count}件\n")
        
        if total_count == 0:
            print("⚠ データがインポートされていません。")
            return
        
        # テキストクエリによる検索
        print("="*60)
        print("テキストクエリによる検索テスト")
        print("="*60)
        print("クエリ: 'beautiful sunset'")
        print()
        
        response = collection.query.near_text(
            query="beautiful sunset",
            limit=3,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            return_properties=["text", "metadata"],
        )
        
        if response.objects:
            print(f"検索結果: {len(response.objects)}件\n")
            for i, obj in enumerate(response.objects, 1):
                print(f"結果 {i}:")
                print(f"  UUID: {obj.uuid}")
                if obj.metadata:
                    print(f"  距離: {obj.metadata.distance:.4f}")
                
                props = obj.properties
                if 'text' in props:
                    text = props['text']
                    if len(text) > 80:
                        text = text[:80] + "..."
                    print(f"  テキスト: {text}")
                
                if 'metadata' in props and props['metadata']:
                    try:
                        import json
                        metadata = json.loads(props['metadata'])
                        print(f"  メタデータ: {metadata}")
                    except:
                        print(f"  メタデータ: {props['metadata']}")
                print()
        else:
            print("検索結果が見つかりませんでした。")
        
        print("="*60)
        print("検索テスト完了")
        print("="*60)


if __name__ == "__main__":
    try:
        test_clip_search()
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

