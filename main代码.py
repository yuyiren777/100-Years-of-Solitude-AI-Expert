import os
import warnings
warnings.filterwarnings("ignore")

from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 加载智谱 API
load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

# 向量库保存路径
DB_PATH = "faiss_db"

# 判断是否已有向量库，有则加载，无则创建
if os.path.exists(DB_PATH):
    print("加载本地向量库")
    embed = ZhipuAIEmbeddings(api_key=api_key, model="embedding-2")
    db = FAISS.load_local(DB_PATH, embed, allow_dangerous_deserialization=True)

else:
    print("首次运行，创建向量库")

    # 1. 加载文档
    loader = TextLoader('./百年孤独.txt', encoding='utf-8')
    documents = loader.load()

    # 2. 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "！", "？", " "]
    )
    texts = text_splitter.split_documents(documents)

    # 3. 智谱 Embedding
    embed = ZhipuAIEmbeddings(api_key=api_key, model="embedding-2")

    # 4. 分批入库
    db = None
    batch_size = 20

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"处理 {i+1} - {i+len(batch)} 条")

        if db is None:
            db = FAISS.from_documents(batch, embed)
        else:
            for doc in batch:
                db.add_documents([doc])

    # 保存向量库到本地
    db.save_local(DB_PATH)
    print("向量库已保存，下次直接加载")

# ==========================
# 检索 n条（全部使用）
# ==========================
retriever = db.as_retriever(search_kwargs={"k": 3})
query = "多年以后，发生了哪些事，分点概括？"
result = retriever.invoke(query)

# 把 n条全部合并成上下文给 AI
context = "\n".join([doc.page_content for doc in result])

print("\n检索到的全部原文：")
print(context)

# ==========================
# GLM-4 总结（基于全部 n条）
# ==========================
from zhipuai import ZhipuAI
client = ZhipuAI(api_key=api_key)

prompt = f"""
你是《百年孤独》问答助手。
请根据以下原文，准确、完整、简洁地回答问题。
不要编造信息。

原文：
{context}

问题：{query}

回答：
"""

response = client.chat.completions.create(
    model="glm-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1
)

print("\nAI 总结回答：")
print(response.choices[0].message.content)
