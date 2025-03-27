# ---------- utils/agents.py ----------
from typing import List, Dict, Any, Union
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent, Agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from utils.chatllm import DeepSeekChatLLM
from langchain_core.runnables import RunnableLambda, RunnableMap
from config import settings




class AgentManager:
    def __init__(self):
        self.tools = [
            self.process_order,
            self.recommend_product,
            # self.check_inventory,
            # self.get_return_policy,
            # self.calculate_shipping_fee
        ]
        self.memory: List[Dict[str, str]] = []
        self.memory = []
        # self.llm = DeepSeekLLM()
        self.llm = DeepSeekChatLLM(
            api_key=settings.DEEPSEEK_API_KEY,
            # model_name="Pro/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            # temperature=0.1
        )
        self.agent_executor = self._create_agent()


    def _format_intermediate_steps(self, steps):

        return format_to_openai_tool_messages(steps)

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是专业客服，请根据以下信息准确回答问题：\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_tools_agent(self.llm, self.tools, prompt)

        return RunnablePassthrough.assign(
            agent_scratchpad=lambda x: self._format_intermediate_steps(x.get("intermediate_steps", []))
        ) | agent | {
            "output": StrOutputParser()  # 关键修复：强制转换为字符串输出
        }

    @tool
    def process_order(self, order_id: str) -> str:
        """处理订单查询"""
        # 这里可以连接数据库实现真实订单查询
        return f"订单{order_id}状态：已发货"

    @tool
    def recommend_product(self, category: str) -> str:
        """根据用户需求推荐产品"""
        # 这里可以连接产品数据库
        return f"根据您的需求推荐：{category}类畅销产品"

    # @tool
    # def check_inventory(self, product_id: str) -> str:
    #     """查询商品库存"""
    #     return f"商品{product_id}当前库存：235件"
    #
    # @tool
    # def get_return_policy(self) -> str:
    #     """获取退货政策"""
    #     return "7天无理由退货，15天内质量问题免费退换"
    #
    # @tool
    # def calculate_shipping_fee(self, total_amount: float) -> str:
    #     """计算运费"""
    #     return "运费10元，满99元可包邮" if total_amount < 99 else "符合包邮条件"

    def run_agent(self, input_text: str, context: str) -> str:
        try:
            # 准备输入数据
            inputs = {
                "input": input_text,
                "context": context,
                "chat_history": [
                    ("user" if msg["role"] == "user" else "assistant", msg["content"])
                    for msg in self.memory
                ],
                "intermediate_steps": []
            }

            # 执行处理链
            result = self.agent_executor.invoke(inputs)

            # 处理工具调用
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    action, observation = step
                    print(f"工具调用: {action.tool} 输入: {action.tool_input} => 结果: {observation}")

            # 更新记忆
            self._update_memory(input_text, result['output'])

            # # 更新记忆（限制最多保留5轮对话）
            # self.memory.append({"role": "user", "content": input_text})
            # self.memory.append({"role": "assistant", "content": result["output"]})
            # self.memory = self.memory[-10:]  # 保留最近5轮对话
            return result["output"]
        except Exception as e:
            return f"处理请求时出错：{str(e)}"

    def _update_memory(self, user_input: str, response: str):
        self.memory.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response}
        ])
        # 保持最近5轮对话
        self.memory = self.memory[-10:]