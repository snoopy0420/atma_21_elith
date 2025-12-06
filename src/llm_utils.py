"""
LLM呼び出しの共通ユーティリティ
"""

import json
from typing import Optional, Dict, Any


class LLMClient:
    """LLM呼び出しの共通クライアント"""
    
    def __init__(self, bedrock_client, default_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Args:
            bedrock_client: boto3のBedrockクライアント
            default_model_id: デフォルトのモデルID
        """
        self.bedrock_client = bedrock_client
        self.default_model_id = default_model_id
    
    def call(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        LLMを呼び出してテキスト生成
        
        Args:
            prompt: 入力プロンプト
            model_id: モデルID（Noneの場合はデフォルトを使用）
            max_tokens: 最大トークン数
            temperature: 温度パラメータ
            system_prompt: システムプロンプト（オプション）
            
        Returns:
            生成されたテキスト、エラー時はNone
        """
        
        if model_id is None:
            model_id = self.default_model_id
        
        # メッセージの構築
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # リクエストボディの構築
        body_dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        # システムプロンプトがあれば追加
        if system_prompt:
            body_dict["system"] = system_prompt
        
        body = json.dumps(body_dict)
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            generated_text = response_body['content'][0]['text']
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"LLM呼び出しエラー (model={model_id}): {e}")
            return None
    
    def call_with_retry(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        リトライ機能付きLLM呼び出し
        
        Args:
            prompt: 入力プロンプト
            model_id: モデルID
            max_tokens: 最大トークン数
            temperature: 温度パラメータ
            system_prompt: システムプロンプト
            max_retries: 最大リトライ回数
            
        Returns:
            生成されたテキスト、エラー時はNone
        """
        
        for attempt in range(max_retries):
            result = self.call(
                prompt=prompt,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
            
            if result is not None:
                return result
            
            if attempt < max_retries - 1:
                print(f"リトライ {attempt + 1}/{max_retries}...")
        
        return None
    
    def check_rejection(self, response: str) -> bool:
        """
        応答が拒否応答かどうかをチェック
        
        Args:
            response: LLMの応答テキスト
            
        Returns:
            True: 拒否応答, False: 通常応答
        """
        
        rejection_phrases = [
            "申し訳ありません",
            "お応えできません",
            "提供できません",
            "お手伝いできません",
            "cannot assist",
            "cannot help",
            "cannot provide",
            "i'm sorry",
            "i apologize"
        ]
        
        response_lower = response.lower()
        
        for phrase in rejection_phrases:
            if phrase in response_lower:
                return True
        
        return False


def create_llm_client(bedrock_client, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0") -> LLMClient:
    """
    LLMクライアントを作成
    
    Args:
        bedrock_client: boto3のBedrockクライアント
        model_id: デフォルトのモデルID
        
    Returns:
        LLMClientインスタンス
    """
    return LLMClient(bedrock_client, model_id)
