import streamlit as st
import sys
import os

# 获取当前文件所在目录（frontend/）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（your_project/）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 Python 路径
sys.path.append(project_root)

from backend.model_handler import initialize_chain, check_order_status, recommend_product
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain

print("Initializing...")
print(f'---st.session_state: {st.session_state}')
# 初始化Session状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    chain_components = initialize_chain()
    st.session_state.chain = LLMChain(
        llm=chain_components["llm"],
        prompt=chain_components["prompt"],
        memory=chain_components["memory"]
    )


# 定义工具（保持不变）
@tool(description="查询订单状态的工具，输入应为订单号（如'1001'）")
def check_order_status_tool(order_id: str) -> str:
    """
    查询订单状态的工具

    Args:
        order_id (str): 用户的订单号，例如 '1001'

    Returns:
        str: 订单状态信息
    """
    return check_order_status(order_id)


@tool
def recommend_product_tool(category: str) -> str:
    """
    根据商品类别推荐产品的工具

    Args:
        category (str): 商品类别，可选值: 'electronics'（电子产品）, 'clothing'（服装）

    Returns:
        str: 推荐结果
    """
    return recommend_product(category)


# 创建Agent
tools = [
    Tool(
        name="CheckOrderStatus",
        func=check_order_status_tool,
        description="用于查询订单状态，输入应为订单号"
    ),
    Tool(
        name="RecommendProduct",
        func=recommend_product_tool,
        description="用于推荐产品，输入应为商品类别（如electronics/clothing）"
    )
]

agent = initialize_agent(
    tools,
    st.session_state.chain.llm,
    agent="conversational-react-description",
    memory=st.session_state.chain.memory,
    verbose=True,
    handle_parsing_errors=True  # 新增错误处理
)

# 界面部分保持不变
st.set_page_config(page_title="DeepSeek智能客服", page_icon="🤖")
st.markdown("""
<style>
.stChatInput textarea {
    background-color: #f0f2f6 !important;
}
</style>
""", unsafe_allow_html=True)

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 使用LangChain Agent处理工具调用
            # response = agent.run(prompt)
            result = agent.invoke({"input": prompt})
            response = result.get("output", "服务暂时不可用")
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"服务异常: {str(e)}")