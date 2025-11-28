"""
マルチモーダルRAG（Retrieval-Augmented Generation）スクリプト

マルチモーダル検索結果を取得し、Ollamaを使用して回答を生成します。
OpenAIやGemini APIを使用せず、すべてローカルで動作します。
"""

import weaviate
from weaviate.classes.generate import GenerativeConfig
import base64
from typing import Optional


def encode_image_to_base64(image_path: str) -> str:
    """画像ファイルをbase64エンコードする"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def rag_with_text_query(
    collection_name: str,
    query_text: str,
    question: str,
    limit: int = 3,
    ollama_model: str = "llama3.2",
    ollama_endpoint: str = "http://localhost:11434"
):
    """
    テキストクエリで検索し、検索結果を基に回答を生成
    
    Args:
        collection_name: コレクション名
        query_text: 検索クエリテキスト
        question: 回答を生成するための質問
        limit: 検索結果の数
        ollama_model: 使用するOllamaモデル名
        ollama_endpoint: Ollama APIエンドポイント
    
    Returns:
        生成された回答テキスト
    """
    with weaviate.connect_to_local() as client:
        if not client.collections.exists(collection_name):
            print(f"✗ コレクション '{collection_name}' が存在しません。")
            return None
        
        collection = client.collections.get(collection_name)
        
        # マルチモーダル検索と回答生成を同時に実行
        response = collection.generate.near_text(
            query=query_text,
            limit=limit,
            grouped_task=question,
            generative_provider=GenerativeConfig.ollama(
                api_endpoint=ollama_endpoint,
                model=ollama_model,
            ),
            return_properties=["text", "image", "metadata"],
        )
        
        return response.generative.text if response.generative else None


def rag_with_image_query(
    collection_name: str,
    image_path: str,
    question: str,
    limit: int = 3,
    ollama_model: str = "llama3.2",
    ollama_endpoint: str = "http://localhost:11434"
):
    """
    画像クエリで検索し、検索結果を基に回答を生成
    
    Args:
        collection_name: コレクション名
        image_path: 検索に使用する画像ファイルのパス
        question: 回答を生成するための質問
        limit: 検索結果の数
        ollama_model: 使用するOllamaモデル名
        ollama_endpoint: Ollama APIエンドポイント
    
    Returns:
        生成された回答テキスト
    """
    with weaviate.connect_to_local() as client:
        if not client.collections.exists(collection_name):
            print(f"✗ コレクション '{collection_name}' が存在しません。")
            return None
        
        collection = client.collections.get(collection_name)
        
        # 画像をbase64エンコード
        image_base64 = encode_image_to_base64(image_path)
        
        # マルチモーダル検索と回答生成を同時に実行
        response = collection.generate.near_image(
            near_image=image_base64,
            limit=limit,
            grouped_task=question,
            generative_provider=GenerativeConfig.ollama(
                api_endpoint=ollama_endpoint,
                model=ollama_model,
            ),
            return_properties=["text", "image", "metadata"],
        )
        
        return response.generative.text if response.generative else None


def demo_multimodal_rag():
    """マルチモーダルRAGのデモ"""
    collection_name = "MultimodalData_CLIP_ViT_B_32"  # 使用するコレクション名
    
    print("マルチモーダルRAGのデモを実行します...")
    print(f"使用コレクション: {collection_name}\n")
    
    # テキストクエリによるRAG
    print("\n1. テキストクエリによるRAG")
    print("   検索クエリ: 'beautiful sunset'")
    print("   質問: 'この画像について説明してください'")
    
    answer = rag_with_text_query(
        collection_name=collection_name,
        query_text="beautiful sunset",
        question="この画像について説明してください",
        limit=3
    )
    
    if answer:
        print(f"\n回答:\n{answer}")
    else:
        print("回答の生成に失敗しました")
    
    # 画像クエリによるRAG（画像ファイルのパスが必要）
    # print("\n2. 画像クエリによるRAG")
    # print("   クエリ画像: 'path/to/query_image.jpg'")
    # print("   質問: 'この画像と類似した画像について説明してください'")
    # 
    # answer = rag_with_image_query(
    #     collection_name=collection_name,
    #     image_path="path/to/query_image.jpg",
    #     question="この画像と類似した画像について説明してください",
    #     limit=3
    # )
    # 
    # if answer:
    #     print(f"\n回答:\n{answer}")
    # else:
    #     print("回答の生成に失敗しました")
    
    print("\n" + "="*60)
    print("RAG完了！")
    print("="*60)


if __name__ == "__main__":
    # デモ実行
    demo_multimodal_rag()
