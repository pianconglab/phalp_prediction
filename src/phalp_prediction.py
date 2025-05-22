import os
import asyncio
import joblib
import pandas
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bio_embeddings.embed import SeqVecEmbedder
from typing import List, Union
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# 请求/响应模型
class SeqRequest(BaseModel):
    sequence: str

class SeqEmbedResponse(BaseModel):
    result: List[float]

class SeqClassifyResponse(BaseModel):
    result: str

class HealthResponse(BaseModel):
    status: str

# Lifespan 事件管理：启动前加载资源，关闭时清理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 读取环境变量 WORKERS（默认 2）
    workers = int(os.getenv("WORKERS", "2"))
    
    # 初始化 SeqVecEmbedder 池
    print("Initializing SeqVecEmbedder pool...")
    embedder_pool = Queue()
    for _ in range(workers):
        embedder_pool.put(SeqVecEmbedder(
            weights_file="data/model_directory/weights.hdf5",
            options_file="data/model_directory/options.json"
        ))
    
    # 初始化线程池
    executor = ThreadPoolExecutor(max_workers=workers)

    # 加载预测模型
    print("Loading classification model...")
    try:
        classifier_model = joblib.load("data/RF_clf_SeqVecEmbeddings_trained.pickle")
        print("Classification model loaded successfully.")
    except Exception as e:
        print(f"Error loading classification model: {e}")
        raise RuntimeError(f"Failed to load classification model: {e}")

    # 挂载到 app.state，供后续请求使用
    app.state.embedder_pool = embedder_pool
    app.state.executor = executor
    app.state.classifier_model = classifier_model

    print("Application startup complete.")
    yield  # 应用开始接收请求

    # 清理：关闭线程池
    print("Shutting down executor...")
    executor.shutdown(wait=True)
    print("Application shutdown complete.")

# 创建 FastAPI 应用，绑定 lifespan
app = FastAPI(title="PhaLP Prediction API", lifespan=lifespan)

# --- 辅助函数（在线程池内运行的同步操作） ---

def sync_embed(seq: str) -> List[float]:
    """
    同步执行蛋白序列嵌入操作。
    此函数将在线程池中运行。
    """
    embedder = app.state.embedder_pool.get()
    try:
        embedding = embedder.embed(seq)
        reduced = embedder.reduce_per_protein(embedding)
        return reduced.tolist()
    finally:
        app.state.embedder_pool.put(embedder)

def sync_classify(seq: str) -> Union[float, int]:
    """
    同步执行蛋白序列嵌入和分类预测操作。
    此函数将在线程池中运行，并重用 embedder。
    """
    embedder = app.state.embedder_pool.get()
    try:
        # 1. 嵌入蛋白序列
        embedding = embedder.embed(seq)
        reduced_embedding = embedder.reduce_per_protein(embedding)
        
        # 2. 转换为 pandas DataFrame 并转置
        # joblib 加载的模型通常期望二维输入，这里将其转换为 DataFrame 并转置以匹配训练时的形状
        df_embedding = pandas.DataFrame(reduced_embedding).T
        
        # 3. 使用预加载的模型进行预测
        prediction = app.state.classifier_model.predict(df_embedding)[0]
        
        return prediction
    finally:
        app.state.embedder_pool.put(embedder)

# --- REST API 接口 ---

# Embed 接口
@app.post("/embed", response_model=SeqEmbedResponse)
async def embed_sequence(req: SeqRequest):
    """
    接收蛋白序列，返回其 1024 维的嵌入向量。
    """
    seq = req.sequence.strip().upper()
    if not seq:
        raise HTTPException(status_code=400, detail="sequence 字段不能为空")
    
    # 异步执行同步的嵌入操作
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            app.state.executor, sync_embed, seq
        )
        return SeqEmbedResponse(result=result)
    except Exception as e:
        print(f"Embedding failed for sequence: {seq[:50]}... Error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding 失败: {e}")

# Classify 接口
@app.post("/classify", response_model=SeqClassifyResponse)
async def classify_sequence(req: SeqRequest):
    """
    接收蛋白序列，返回其分类预测结果。
    """
    seq = req.sequence.strip().upper()
    if not seq:
        raise HTTPException(status_code=400, detail="sequence 字段不能为空")
    
    # 异步执行同步的嵌入和分类操作
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            app.state.executor, sync_classify, seq
        )
        return SeqClassifyResponse(result=result)
    except Exception as e:
        print(f"Classification failed for sequence: {seq[:50]}... Error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification 失败: {e}")

# 健康检查
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口，返回服务状态。
    """
    return HealthResponse(status="ok")