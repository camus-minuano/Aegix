import requests
import json
from typing import Dict, Optional

import config # Import configurations

class OllamaClient:
    """Ollama API客户端封装类"""
    def __init__(self, base_url: str = config.OLLAMA_BASE_URL, timeout: int = config.OLLAMA_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout

    def generate(self,
                 prompt: str,
                 model: str,
                 stream: bool = False,
                 options: Optional[Dict] = None) -> Dict:
        """调用Ollama生成接口"""
        url = f"{self.base_url}/api/generate"
        
        payload_options = config.OLLAMA_MODEL_OPTIONS.copy()
        if options:
            payload_options.update(options)
            
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": payload_options
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            print(f"连接失败，请确认：\n1. Ollama服务正在运行（终端执行 `ollama serve`）\n2. 模型 {model} 已下载（如 `ollama pull {model}`）")
            return {}
        except requests.exceptions.Timeout:
            print(f"稍等，模型 {model} 正在处理（超时未响应，可适当增大timeout参数）")
            return {}
        except requests.exceptions.HTTPError as e:
            print(f"API调用HTTP错误: {e.response.status_code} - {e.response.text}")
            return {}
        except Exception as e:
            print(f"API调用未知错误: {str(e)}")
            return {}

    def check_ollama_service_availability(self, model_to_test: str = config.SAFETY_MODEL_NAME) -> bool:
        """测试Ollama服务和指定模型是否可用"""
        print(f"正在测试Ollama服务及模型 '{model_to_test}' 的可用性...")
        test_payload = {
            "model": model_to_test,
            "prompt": "Hello",
            "stream": False,
            "options": {"temperature": 0.0}
        }
        url = f"{self.base_url}/api/generate"
        try:
            response = requests.post(url, json=test_payload, timeout=5)
            if response.status_code == 200 and response.json().get("response"):
                print(f"Ollama服务及模型 '{model_to_test}' 可用。")
                return True
            else:
                print(f"Ollama模型 '{model_to_test}' 测试失败。响应: {response.text[:200]}...")
                if "model not found" in response.text.lower():
                    print(f"请确保模型 '{model_to_test}' 已通过 `ollama pull {model_to_test}` 下载。")
                return False
        except requests.exceptions.ConnectionError:
            print(f"无法连接到Ollama服务于 {self.base_url}。请确保Ollama正在运行。")
            return False
        except Exception as e:
            print(f"测试Ollama服务时发生错误: {e}")
            return False
