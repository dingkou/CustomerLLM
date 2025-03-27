# CustomerLLM
基于API大模型的智能客服

~~~
项目名称：基于大模型的智能客服
功能介绍：使用大语言模型（如deepseek）构建一个智能客服系统，能够回答用户问题、处理订单、提供产品推荐等。可以通过LangChain框架实现多轮对话和上下文记忆14。
技术栈：Python、LangChain。
目标：实现高准确率的对话交互和任务完成能力。
~~~

1. **模型调用方式**：
   - 新增 `DeepSeekLLM` 自定义类，继承自 LangChain 的 `LLM` 基类
   - 通过 `requests` 库调用 DeepSeek API
   - 需要设置环境变量 `DEEPSEEK_API_KEY`
2. **上下文管理**：
   - 保留 LangChain 的 `ConversationBufferMemory` 管理对话历史
   - 通过 `LLMChain` 整合 prompt 和 memory
3. **错误处理**：
   - 添加 API 调用异常捕获
   - 前端显示友好错误信息
4. **工具调用**：
   - 保持原有工具函数和 Agent 的集成逻辑
   - 通过 `initialize_agent` 实现工具自动调用

### 运行方式

1. 在DeepSeek平台注册并获取API Key
2. 将API Key填入`.env`文件
3. 运行命令：

```
streamlit run frontend/app.py
```

### 扩展建议

1. **流式响应**：
   修改API调用，使用`stream=True`参数并处理chunk响应，实现打字机效果：
2. **RAG增强**：
   添加向量数据库（如ChromaDB），实现知识库检索增强
3. **对话持久化**：
   将会话历史保存到数据库（如SQLite）：