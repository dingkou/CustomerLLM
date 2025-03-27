from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


class DeepSeekLLM(LLM):
    """ 自定义DeepSeek API调用类 """
    # model_name: str = "deepseek-chat"  # 根据API文档调整模型名称
    model_name: str = "Pro/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 根据API文档调整模型名称

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "stop": None,
            # "response_format": {"type": "text"},
            # "tools": [
            #     {
            #         "type": "function",
            #         "function": {
            #             "description": "<string>",
            #             "name": "<string>",
            #             "parameters": {},
            #             "strict": False
            #         }
            #     }
            # ]
        }

        # try:
        #     response = requests.post(
        #         "https://api.deepseek.com/v1/chat/completions",  # 根据API文档调整endpoint
        #         headers=headers,
        #         json=payload
        #     )
        #     response.raise_for_status()
        #     return response.json()['choices'][0]['message']['content']
        # except Exception as e:
        #     return f"API调用失败: {str(e)}"

        try:
            response = requests.post(
                "https://api.siliconflow.cn/v1/chat/completions",  # 根据API文档调整endpoint
                headers=headers,
                json=payload
            )
            print(response.text)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"API调用失败: {str(e)}"


# 对话模板（保持工具调用说明）
template = """
你是一个专业的智能客服助手，需要完成以下任务：
1. 回答用户关于产品的咨询
2. 处理订单状态查询（调用 check_order_status 工具）,如果大模型没有获取到订单编号或者订单ID等，就回答让用户输入订单编号
3. 根据用户需求推荐产品（调用 recommend_product 工具）

{chat_history}
用户：{input}
助手：
"""

prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=template
)


# 初始化对话链
def initialize_chain():
    llm = DeepSeekLLM()
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与模板中的 {chat_history} 对应
        human_prefix="用户",
        ai_prefix="助手"
    )
    # memory = ConversationBufferMemory(human_prefix="用户", ai_prefix="助手")

    return {
        "llm": llm,
        "memory": memory,
        "prompt": prompt_template
    }


# 工具函数保持不变
def check_order_status(order_id: str) -> str:
    order_db = {
        "1001": "已发货",
        "1002": "处理中"
    }
    return order_db.get(order_id, "订单不存在")


def recommend_product(category: str) -> str:
    product_db = {
        "electronics": "推荐：智能手表（型号X3，限时折扣）",
        "clothing": "推荐：夏季新款T恤（多种颜色可选）"
    }
    return product_db.get(category, "推荐：畅销款商品")
