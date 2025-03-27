from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import Optional, List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
import requests


class DeepSeekChatLLM(BaseLLM):
    """
    自定义适配器，确保输出符合LangChain的ChatGeneration格式
    """
    api_key: str
    model_name: str = "Pro/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    temperature: float = 0.7
    max_tokens: int = 512

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # 实现原始文本调用
        return self._call_api(prompt, is_chat=False)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._call(prompt, stop, run_manager, **kwargs)

    def _generate(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 实现Chat格式生成
        response = self._call_api(messages, is_chat=True)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    def _call_api(self, input_data: Any, is_chat: bool) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **({"messages": input_data} if is_chat else {"prompt": input_data})
        }
        print(payload)

        try:
            response = requests.post(
                "https://api.siliconflow.cn/v1/chat/completions",
                headers=headers,
                json=payload
            )
            print(response.text)
            # response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            raise ValueError(f"API调用失败: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"