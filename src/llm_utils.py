"""
LLM呼び出しの共通ユーティリティ
"""

import json
import time
from typing import Optional, Dict, Any


class LLMClient:
    """LLM呼び出しの共通クライアント（Claude / Mistral対応）"""
    
    def __init__(self, bedrock_client, default_model_id: str = "mistral.mistral-large-2407-v1:0"):
        """
        Args:
            bedrock_client: boto3のBedrockクライアント
            default_model_id: デフォルトのモデルID
        """
        self.bedrock_client = bedrock_client
        self.default_model_id = default_model_id
    
    def _is_mistral(self, model_id: str) -> bool:
        """モデルがMistralかどうかを判定"""
        return "mistral" in model_id.lower()
    
    def call(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        LLMを呼び出してテキスト生成（Claude / Mistral自動判定）
        
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
        
        # モデルに応じてリクエストボディを構築
        if self._is_mistral(model_id):
            body_dict = self._build_mistral_body(prompt, max_tokens, temperature, system_prompt)
        else:
            body_dict = self._build_claude_body(prompt, max_tokens, temperature, system_prompt)
        
        body = json.dumps(body_dict)
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            
            # レスポンスの解析
            if self._is_mistral(model_id):
                generated_text = response_body['outputs'][0]['text']
            else:
                generated_text = response_body['content'][0]['text']
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"LLM呼び出しエラー (model={model_id}): {e}")
            return None
    
    def _build_mistral_body(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Mistral用のリクエストボディを構築"""
        # システムプロンプトをプロンプトに統合
        if system_prompt:
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"
        
        return {
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    
    def _build_claude_body(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Claude用のリクエストボディを構築"""
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        body_dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            body_dict["system"] = system_prompt
        
        return body_dict
    
    def call_with_retry(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        initial_wait: float = 1.0
    ) -> Optional[str]:
        """
        リトライ機能付きLLM呼び出し（エクスポネンシャルバックオフ付き）
        
        Args:
            prompt: 入力プロンプト
            model_id: モデルID
            max_tokens: 最大トークン数
            temperature: 温度パラメータ
            system_prompt: システムプロンプト
            max_retries: 最大リトライ回数
            initial_wait: 初回待機時間（秒）
            
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
                # エクスポネンシャルバックオフ: 1秒 → 2秒 → 4秒
                wait_time = initial_wait * (2 ** attempt)
                print(f"リトライ {attempt + 1}/{max_retries} ({wait_time}秒待機)...")
                time.sleep(wait_time)
        
        print(f"最大リトライ回数 ({max_retries}) に達しました")
        return None
    


def create_llm_client(bedrock_client, model_id: str = "mistral.mistral-large-2407-v1:0") -> LLMClient:
    """
    LLMクライアントを作成
    
    Args:
        bedrock_client: boto3のBedrockクライアント
        model_id: デフォルトのモデルID
        
    Returns:
        LLMClientインスタンス
    """
    return LLMClient(bedrock_client, model_id)
