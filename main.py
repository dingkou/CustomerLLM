# ---------- main.py ----------
from fastapi import FastAPI, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.chains import ConversationChain
from utils.agents import AgentManager
from utils.rag import RAGRetriever

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 初始化组件
rag = RAGRetriever()
agent_manager = AgentManager()

# 示例文档数据（实际项目应从数据库加载）
sample_docs = [
    "我们的退货政策是7天无理由退货",
    "标准运费为10元，满99元包邮",
    "热门产品包括智能手表和无线耳机",
    """
    商品介绍
商品名称：智能健康手环 X10
商品简介：
智能健康手环 X10 是一款集健康管理、运动监测和智能提醒于一体的高科技可穿戴设备。它不仅能够实时监测您的心率、血氧饱和度和睡眠质量，还能记录每日的步数、卡路里消耗和运动距离，帮助您全面了解自己的健康状况。
主要功能：
健康监测：24小时实时心率监测、血氧水平检测、压力指数分析。
运动模式：支持跑步、骑行、游泳等12种专业运动模式，自动记录运动数据。
睡眠追踪：深度与浅度睡眠分析，提供改善睡眠建议。
智能提醒：来电、短信及应用通知提醒，不错过任何重要信息。
防水设计：IP68级防水，适合各种环境使用。
长续航：一次充电可使用长达14天，减少频繁充电烦恼。
 
功能文档
一、产品规格
显示屏：0.96英寸彩色触摸屏
材质：食品级硅胶表带
兼容性：iOS 10.0及以上版本，Android 5.0及以上版本
连接方式：蓝牙5.0
电池容量：120mAh
二、功能详细说明
健康监测
心率监测：采用光学传感器技术，每分钟更新一次数据，确保准确性。
血氧检测：通过红外光测量血氧饱和度（SpO2），帮助了解身体供氧情况。
压力管理：根据心率变异性计算压力水平，并提供放松训练指导。
运动模式
支持多种运动模式，包括但不限于：
跑步
骑行
游泳
瑜伽
每种模式下均可记录时间、距离、卡路里消耗等关键指标。
睡眠追踪
自动识别入睡和醒来时间。
分析深睡、浅睡及快速眼动（REM）阶段。
提供个性化睡眠改进建议。
智能提醒
来电提醒：显示来电号码或联系人姓名。
短信提醒：实时推送短信内容。
应用通知：支持微信、QQ、邮件等多种应用的消息提醒。
其他功能
闹钟：设置多个闹钟，支持静音振动模式。
找手机：一键查找丢失的手机。
天气预报：显示未来几天的天气情况。
三、使用方法
下载APP
在手机应用商店搜索“X10 Health”并下载安装。
绑定设备
打开APP，按照提示将手环与手机蓝牙配对。
开始使用
同步数据后，即可查看各项健康指标和运动记录。
四、注意事项
避免长时间佩戴手环，建议每天取下休息一段时间。
游泳时请勿超过防水深度限制（10米以内）。
定期清理表带，保持卫生。
五、售后服务
提供一年质保服务，非人为损坏免费维修。
如有任何问题，请联系官方客服热线：400-XXXX-XXXX。
    """
]
rag.init_vector_store(sample_docs)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     data = await request.json()
#     user_input = data.get("message")
#     history = data.get("history", [])
#
#     # 处理上下文（列表包含字典格式）
#     context = [{"role": h["role"], "content": h["content"]} for h in history]
#     context.append({"role": "user", "content": user_input})
#
#     # 结合RAG检索
#     retrieved_info = rag.retrieve(user_input)
#
#     # 构建最终提示
#     prompt = f"用户问题：{user_input}\n相关背景：{retrieved_info}"
#
#     # 调用大模型API（示例用，需替换实际API调用）
#     response = "这是AI的示例回复"  # 替换为实际API调用
#
#     context.append({"role": "assistant", "content": response})
#     return {"response": response, "history": context}


@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message")
    # history = data.get("history", [])

    # RAG检索
    retrieved_info = rag.retrieve(user_input)
    rag_context = "\n".join([doc.page_content for doc in retrieved_info])
    # 执行Agent处理
    try:
        response = agent_manager.run_agent(
            input_text=user_input,
            context=rag_context
        )
    except Exception as e:
        response = f"服务暂时不可用，错误信息：{str(e)}"

    return {
        "response": response,
        "history": agent_manager.memory
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=56589)