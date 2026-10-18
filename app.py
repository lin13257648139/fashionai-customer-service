"""
服装电商智能客服系统 - 核心模块
包含：智能体 + RAG + 工具调用 + 情绪分析 + 多模态 + 日志
"""

import os
import json
import logging
import base64
import requests
import time
import sqlite3
import math
import re
import hashlib
import inspect
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, List, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# 向量检索
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings

# 中文分词
import jieba

from dotenv import load_dotenv
from offline_engine import OfflineAssistant

load_dotenv()

# 初始化日志
logger = logging.getLogger("FashionAI")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# ============================================================
# 单例模式基类
# ============================================================

class Singleton(type):
    """单例模式元类"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# ============================================================
# 配置（单例模式）
# ============================================================

@dataclass
class Config(metaclass=Singleton):
    # 智谱 API
    ZHIPU_API_KEY: str = field(default_factory=lambda: os.getenv("ZHIPU_API_KEY", ""))
    ZHIPU_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"
    ZHIPU_MODEL: str = "glm-4-flash"
    
    # 豆包 API（图片分析）
    DOUBAO_API_KEY: str = field(default_factory=lambda: os.getenv("DOUBAO_API_KEY", ""))
    DOUBAO_VISION_MODEL: str = "doubao-1-5-vision-pro-32k-250115"
    DOUBAO_API_URL: str = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    # 未配置模型服务时默认启用本地规则与知识库降级能力。
    OFFLINE_MODE: bool = field(default_factory=lambda: os.getenv("FASHIONAI_OFFLINE", "auto").lower() != "false")
    
    # BGE 嵌入模型配置（本地路径）
    EMBEDDING_MODEL: str = os.path.join(os.path.dirname(__file__), "bge-small-zh")
    EMBEDDING_DEVICE: str = "cpu"  # 可选: "cpu", "cuda"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_NORMALIZE: bool = True  # BGE 必须设置为 True
    
    # 系统配置
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # 会话配置
    MAX_HISTORY_LENGTH: int = 20
    MAX_CONTEXT_TOKENS: int = 4000  # 上下文窗口限制
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保必要的目录存在
        Path("./data").mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)
    
    def reload(self):
        """重新加载配置（从环境变量）"""
        self.ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
        self.DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")


# ============================================================
# LLM 管理器（单例模式，连接复用）
# ============================================================

class LLMManager(metaclass=Singleton):
    """
    LLM 管理器 - 单例模式，复用 LLM 连接
    支持重试、熔断降级
    """
    
    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None
        self._llm_with_tools: Optional[Any] = None
        self._config = Config()
        self._consecutive_failures = 0
        self._max_failures = 5
        self._circuit_open = False
        self._circuit_timeout = 60  # 熔断超时时间（秒）
        self._last_failure_time = 0
        
        # 初始化 LLM
        self._init_llm()
    
    def _init_llm(self):
        """初始化 LLM"""
        try:
            api_key = self._config.ZHIPU_API_KEY
            if not api_key:
                self._llm = None
                self._llm_with_tools = None
                logger.info("ℹ️ 未配置模型 API，使用离线客服引擎")
                return
            
            self._llm = ChatOpenAI(
                model=self._config.ZHIPU_MODEL,
                temperature=0.7,
                api_key=api_key,
                base_url=self._config.ZHIPU_BASE_URL,
                max_retries=self._config.MAX_RETRIES,
                timeout=30,
            )
            self._llm_with_tools = self._llm.bind_tools(TOOLS)
            logger.info("✅ LLM 初始化成功")
            
        except Exception as e:
            logger.error(f"❌ LLM 初始化失败: {e}")
            self._llm = None
    
    def get_llm(self) -> Optional[ChatOpenAI]:
        """获取 LLM 实例"""
        if self._circuit_open:
            # 检查是否超过熔断超时
            if time.time() - self._last_failure_time > self._circuit_timeout:
                logger.info("🔄 熔断恢复，尝试重新初始化 LLM")
                self._circuit_open = False
                self._consecutive_failures = 0
                self._init_llm()
            else:
                return None
        
        if self._llm is None:
            self._init_llm()
        
        return self._llm
    
    def get_llm_with_tools(self):
        """获取绑定工具的 LLM 实例"""
        if self._circuit_open:
            return None
        
        if self._llm_with_tools is None and self._llm:
            self._llm_with_tools = self._llm.bind_tools(TOOLS)
        
        return self._llm_with_tools
    
    def record_failure(self):
        """记录失败"""
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        
        if self._consecutive_failures >= self._max_failures:
            logger.warning(f"⚠️ 触发熔断机制，连续失败 {self._consecutive_failures} 次")
            self._circuit_open = True
    
    def record_success(self):
        """记录成功"""
        self._consecutive_failures = 0
        self._circuit_open = False


# ============================================================
# BGE 嵌入模型加载（带缓存）
# ============================================================

# 全局缓存
_embedding_model_cache: Optional[HuggingFaceEmbeddings] = None


def load_embedding_model(use_cache: bool = True) -> Optional[HuggingFaceEmbeddings]:
    """
    加载 BGE 嵌入模型（带缓存）
    
    Args:
        use_cache: 是否使用缓存
    
    Returns:
        HuggingFaceEmbeddings 对象，失败返回 None
    """
    global _embedding_model_cache
    
    # 返回缓存
    if use_cache and _embedding_model_cache is not None:
        logger.info("✅ BGE 模型从缓存加载")
        return _embedding_model_cache
    
    config = Config()
    model_path = config.EMBEDDING_MODEL
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"❌ BGE 模型路径不存在: {model_path}")
        logger.error("请确保已将 BGE 模型下载到本地目录")
        return None
    
    try:
        logger.info(f"🔄 正在加载 BGE 嵌入模型: {model_path}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                'device': config.EMBEDDING_DEVICE,
            },
            encode_kwargs={
                'normalize_embeddings': config.EMBEDDING_NORMALIZE,  # BGE 必须为 True
                'batch_size': config.EMBEDDING_BATCH_SIZE,
            },
            cache_folder=None,  # 禁用缓存以支持本地模型
        )
        
        # 验证模型是否正确加载 - 测试嵌入
        test_embedding = embeddings.embed_query("测试")
        if test_embedding and len(test_embedding) > 0:
            logger.info(f"✅ BGE 模型加载成功！向量维度: {len(test_embedding)}")
            _embedding_model_cache = embeddings
            return embeddings
        else:
            logger.error("❌ BGE 模型验证失败：无法生成有效嵌入")
            return None
            
    except Exception as e:
        logger.error(f"❌ BGE 模型加载失败: {str(e)}")
        logger.error("请检查模型文件是否完整，或尝试重新下载模型")
        return None


def get_embedding_model() -> Optional[HuggingFaceEmbeddings]:
    """获取嵌入模型（使用缓存）"""
    return load_embedding_model(use_cache=True)


def clear_embedding_cache():
    """清除嵌入模型缓存"""
    global _embedding_model_cache
    _embedding_model_cache = None
    logger.info("🗑️ 嵌入模型缓存已清除")

# ============================================================
# 日志系统
# ============================================================

class Logger:
    """统一的日志系统"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 配置文件日志
        self.file_handler = logging.FileHandler(self.log_dir / f"agent_{datetime.now().strftime('%Y%m%d')}.log")
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # 配置控制台日志
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        self.logger = logging.getLogger("FashionAI")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def log_conversation(self, user_id: str, user_msg: str, assistant_msg: str, duration: float):
        """记录对话"""
        self.logger.info(f"CONVERSATION | user:{user_id} | duration:{duration}s | user:{user_msg[:50]} | assistant:{assistant_msg[:50]}")
    
    def log_error(self, error_type: str, error_msg: str, context: str = ""):
        """记录错误"""
        self.logger.error(f"ERROR | type:{error_type} | msg:{error_msg} | context:{context}")


# ============================================================
# 数据库（统计和持久化）
# ============================================================

class Database:
    """SQLite数据库，用于统计和持久化"""
    
    def __init__(self, db_path: str = "fashionai.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_message TEXT,
                    assistant_message TEXT,
                    emotion TEXT,
                    agent_mode TEXT,
                    duration REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    tool_name TEXT,
                    success BOOLEAN,
                    duration REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    rating INTEGER,
                    comment TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def save_conversation(self, session_id: str, user_msg: str, assistant_msg: str, 
                          emotion: str, agent_mode: str, duration: float) -> int:
        """保存对话记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (session_id, user_message, assistant_message, emotion, agent_mode, duration) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, user_msg, assistant_msg, emotion, agent_mode, duration)
            )
            conn.commit()
            row_id = cursor.lastrowid
            return row_id if row_id is not None else 0
    
    def save_tool_call(self, conversation_id: int, tool_name: str, success: bool, duration: float):
        """保存工具调用记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO tool_calls (conversation_id, tool_name, success, duration) VALUES (?, ?, ?, ?)",
                (conversation_id, tool_name, success, duration)
            )
            conn.commit()
    
    def save_feedback(self, conversation_id: int, rating: int, comment: str = ""):
        """保存用户反馈"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO feedback (conversation_id, rating, comment) VALUES (?, ?, ?)",
                (conversation_id, rating, comment)
            )
            conn.commit()
    
    def get_stats(self) -> dict:
        """获取统计数据"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            today = conn.execute("SELECT COUNT(*) FROM conversations WHERE DATE(timestamp) = DATE('now')").fetchone()[0]
            avg_duration = conn.execute("SELECT AVG(duration) FROM conversations").fetchone()[0] or 0
            emotions = conn.execute("SELECT emotion, COUNT(*) FROM conversations GROUP BY emotion").fetchall()
            tool_success = conn.execute("SELECT AVG(CASE WHEN success THEN 1 ELSE 0 END) FROM tool_calls").fetchone()[0] or 0
            avg_rating = conn.execute("SELECT AVG(rating) FROM feedback").fetchone()[0] or 0
            
            return {
                "total_conversations": total,
                "today_conversations": today,
                "avg_duration": round(avg_duration, 2),
                "emotion_distribution": dict(emotions),
                "tool_success_rate": round(tool_success * 100, 1),
                "avg_rating": round(avg_rating, 1)
            }


# ============================================================
# 知识库管理器（从外部文件动态加载）
# ============================================================

class KnowledgeManager:
    """知识库管理器 - 支持从文件动态加载"""
    
    def __init__(self, knowledge_dir: str = None):
        if knowledge_dir is None:
            knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
        self.knowledge_dir = Path(knowledge_dir)
        self._documents: List[dict] = []
        self._last_modified: float = 0
        self._load_knowledge()
    
    def _load_knowledge(self):
        """从文件动态加载知识"""
        self._documents = []
        latest_mtime = 0.0
        
        if not self.knowledge_dir.exists():
            logger.warning(f"⚠️ 知识库目录不存在: {self.knowledge_dir}")
            return
        
        for file_path in self.knowledge_dir.glob("*.txt"):
            try:
                mtime = file_path.stat().st_mtime
                latest_mtime = max(latest_mtime, mtime)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析文件内容
                category = file_path.stem  # 文件名作为类别
                if file_path.name == "product_info.txt":
                    self._parse_product_info(content, category, file_path.name)
                else:
                    for line in content.split('\n'):
                        line = line.strip()
                        if not line or line.startswith('【') or line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
                            continue
                        if '：' in line or ':' in line:
                            self._documents.append({
                                "content": line,
                                "category": category,
                                "source": file_path.name
                            })
                
                logger.info(f"📂 加载知识文件: {file_path.name}, 文档数: {len(self._documents)}")
                
            except Exception as e:
                logger.error(f"❌ 加载知识文件失败 {file_path}: {e}")

            self._last_modified = latest_mtime
        
        logger.info(f"✅ 知识库加载完成，共 {len(self._documents)} 个文档")
    
    def _parse_product_info(self, content: str, category: str, source: str):
        """解析 product_info.txt 中的商品信息"""
        lines = content.split('\n')
        products = []
        current_product = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('【') or ('、' in line and '装' in line):  # 如 "一、男装上衣"
                continue
            if line[0].isdigit() and '. ' in line and '商品名称：' in line:
                # 新商品开始
                if current_product:
                    products.append(current_product)
                current_product = {}
                # 解析商品名称
                current_product['name'] = line.split('商品名称：')[1].strip()
            elif current_product:  # 只有在有当前商品时才解析字段
                if '价格：' in line:
                    current_product['price'] = line.split('价格：')[1].strip()
                elif '颜色：' in line:
                    current_product['color'] = line.split('颜色：')[1].strip()
                elif '尺码：' in line:
                    current_product['size'] = line.split('尺码：')[1].strip()
                elif '材质：' in line:
                    current_product['material'] = line.split('材质：')[1].strip()
                elif '卖点：' in line:
                    current_product['selling_point'] = line.split('卖点：')[1].strip()
                elif '图片：' in line:
                    current_product['image_url'] = line.split('图片：')[1].strip()
                elif '链接：' in line:
                    current_product['shop_url'] = line.split('链接：')[1].strip()
        
        if current_product:
            products.append(current_product)
        
        # 生成知识条目
        for product in products:
            content_str = (
                f"商品：{product.get('name', '')}，价格：{product.get('price', '')}，颜色：{product.get('color', '')}，"
                f"尺码：{product.get('size', '')}，材质：{product.get('material', '')}，卖点：{product.get('selling_point', '')}，"
                f"图片：{product.get('image_url', '')}，链接：{product.get('shop_url', '')}"
            )
            self._documents.append({
                "content": content_str,
                "category": category,
                "source": source
            })
    
    def reload_if_modified(self):
        """检查并重新加载修改过的文件"""
        for file_path in self.knowledge_dir.glob("*.txt"):
            if file_path.stat().st_mtime > self._last_modified:
                self._load_knowledge()
                return True
        return False
    
    def get_documents(self) -> List[dict]:
        """获取所有文档"""
        return self._documents
    
    def get_texts(self) -> List[str]:
        """获取所有文档文本"""
        return [doc["content"] for doc in self._documents]


# ============================================================
# 向量知识库（支持 BGE 嵌入 + BM25 + 混合检索 + RRF 重排序）
# ============================================================

class VectorKnowledgeBase:
    """
    混合向量知识库（BM25 + BGE 嵌入 + RRF 重排序）
    支持从外部文件动态加载知识
    """
    
    def __init__(self, embeddings: Optional[HuggingFaceEmbeddings] = None, knowledge_dir: str = None):
        if knowledge_dir is None:
            knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
        self.documents: List[str] = []
        self.doc_ids: List[int] = []
        self.doc_metadata: List[dict] = []  # 添加元数据支持
        self.doc_vectors: np.ndarray = None
        
        # BM25 向量器（用于关键词检索）
        self._bm25_tokenizer = jieba
        self._bm25_scores: List[dict] = []
        self._bm25_k1 = 1.5
        self._bm25_b = 0.75
        
        # TF-IDF 向量器（用于混合检索）
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_vectors: np.ndarray = None
        
        # BGE 嵌入模型
        self.embeddings = embeddings
        self.embedding_dim: int = 0
        
        # RRF 重排序参数
        self.rrf_k = 60
        
        # 知识库管理器
        self.knowledge_manager = KnowledgeManager(knowledge_dir)
        
        # 向量持久化路径
        self._index_cache_path = Path("./data/vector_index.npz")
        
        self._init_knowledge()
    
    def _init_knowledge(self):
        """初始化知识库内容"""
        # 从文件动态加载知识
        knowledge_data = self.knowledge_manager.get_documents()
        
        if not knowledge_data:
            # 使用默认知识
            logger.warning("⚠️ 未从文件加载到知识，使用内置默认知识")
            knowledge_data = self._get_default_knowledge()
        
        # 使用数字索引
        self.documents = [doc["content"] if isinstance(doc, dict) else doc for doc in knowledge_data]
        self.doc_ids = list(range(len(self.documents)))
        
        # 保存元数据
        if knowledge_data and isinstance(knowledge_data[0], dict):
            self.doc_metadata = knowledge_data
        else:
            self.doc_metadata = [{"content": doc} for doc in self.documents]
        
        logger.info(f"📚 知识库加载完成，共 {len(self.documents)} 个文档")
        self._build_index()
    
    def _get_default_knowledge(self) -> List[dict]:
        """获取默认知识"""
        return [
            {"content": "XS码女装适合身高155-160cm体重40-50kg", "category": "size"},
            {"content": "S码女装适合身高160-165cm体重45-55kg", "category": "size"},
            {"content": "M码女装适合身高165-170cm体重50-60kg", "category": "size"},
            {"content": "L码女装适合身高165-170cm体重60-70kg", "category": "size"},
            {"content": "XL码女装适合身高170-175cm体重65-75kg", "category": "size"},
            {"content": "S码男装适合身高165-170cm体重55-65kg", "category": "size"},
            {"content": "M码男装适合身高170-175cm体重65-75kg", "category": "size"},
            {"content": "L码男装适合身高175-180cm体重75-85kg", "category": "size"},
            {"content": "XL码男装适合身高180-185cm体重85-95kg", "category": "size"},
            {"content": "XXL码男装适合身高185-190cm体重95-105kg", "category": "size"},
        ]

    def _get_corpus_signature(self) -> str:
        """基于当前文档计算语料签名，用于校验缓存是否过期。"""
        joined = "\n".join(self.documents)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()
    
    def _build_index(self):
        """构建索引"""
        if not self.documents:
            logger.warning("⚠️ 知识库为空，跳过索引构建")
            return

        corpus_signature = self._get_corpus_signature()
        
        # 尝试从缓存加载索引
        if self._index_cache_path.exists():
            try:
                data = np.load(self._index_cache_path)
                cached_vectors = data['doc_vectors'] if 'doc_vectors' in data.files else None
                cached_signature = ""
                if 'corpus_signature' in data.files:
                    cached_signature_value = data['corpus_signature']
                    cached_signature = str(cached_signature_value.item() if isinstance(cached_signature_value, np.ndarray) else cached_signature_value)

                cached_doc_count = -1
                if 'doc_count' in data.files:
                    cached_doc_count_value = data['doc_count']
                    cached_doc_count = int(cached_doc_count_value.item() if isinstance(cached_doc_count_value, np.ndarray) else cached_doc_count_value)

                cache_valid = (
                    isinstance(cached_vectors, np.ndarray)
                    and cached_vectors.ndim == 2
                    and cached_doc_count == len(self.documents)
                    and cached_signature == corpus_signature
                )

                if self.embeddings and cache_valid:
                    self.doc_vectors = cached_vectors
                    self.embedding_dim = self.doc_vectors.shape[1]
                    self.tfidf_vectors = self.vectorizer.fit_transform(self.documents)
                    logger.info(f"✅ 从缓存加载向量索引成功")
                    return
            except Exception as e:
                logger.warning(f"⚠️ 加载缓存索引失败: {e}")
        
        # 构建 TF-IDF 索引
        try:
            self.tfidf_vectors = self.vectorizer.fit_transform(self.documents)
            logger.info(f"✅ TF-IDF 索引构建完成，维度: {self.tfidf_vectors.shape}")
        except Exception as e:
            logger.error(f"❌ TF-IDF 索引构建失败: {e}")
            self.tfidf_vectors = None
        
        # 构建 BGE 嵌入索引
        if self.embeddings:
            try:
                logger.info("🔄 正在生成 BGE 嵌入向量...")
                self.doc_vectors = np.array(self.embeddings.embed_documents(self.documents))
                self.embedding_dim = self.doc_vectors.shape[1]
                logger.info(f"✅ BGE 嵌入索引构建完成，维度: {self.doc_vectors.shape}")
                
                # 保存索引缓存
                self._save_index_cache()
            except Exception as e:
                logger.error(f"❌ BGE 嵌入索引构建失败: {e}")
                self.doc_vectors = None
        else:
            logger.warning("⚠️ BGE 嵌入模型未加载，仅使用 TF-IDF 检索")
    
    def _save_index_cache(self):
        """保存向量索引到缓存"""
        if self.doc_vectors is not None:
            try:
                self._index_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    self._index_cache_path,
                    doc_vectors=self.doc_vectors,
                    embedding_dim=self.embedding_dim,
                    doc_count=len(self.documents),
                    corpus_signature=self._get_corpus_signature()
                )
                logger.info(f"💾 向量索引已缓存到: {self._index_cache_path}")
            except Exception as e:
                logger.warning(f"⚠️ 保存索引缓存失败: {e}")
    
    def _bm25_score(self, query: str) -> List[dict]:
        """BM25 关键词检索"""
        if not self.documents:
            return []
        
        try:
            # 分词
            query_terms = list(self._bm25_tokenizer.cut(query))
            
            # 构建词频和文档频率
            doc_freq = {}
            term_freqs = []
            
            for doc in self.documents:
                doc_terms = list(self._bm25_tokenizer.cut(doc))
                term_freq = {}
                for term in doc_terms:
                    term_freq[term] = term_freq.get(term, 0) + 1
                term_freqs.append(term_freq)
                for term in set(doc_terms):
                    doc_freq[term] = doc_freq.get(term, 0) + 1
            
            n = len(self.documents)
            avgdl = sum(len(d.split()) for d in self.documents) / n
            
            results = []
            for idx, doc in enumerate(self.documents):
                score = 0
                for term in query_terms:
                    if term in term_freqs[idx]:
                        tf = term_freqs[idx][term]
                        df = doc_freq.get(term, 0)
                        if df == 0:
                            continue
                        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                        score += idf * (tf * (self._bm25_k1 + 1)) / (tf + self._bm25_k1 * (1 - self._bm25_b + self._bm25_b * len(doc) / avgdl))
                
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "content": self.documents[idx],
                    "score": score,
                    "metadata": self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
                })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:10]
        except Exception as e:
            logger.error(f"❌ BM25 检索失败: {e}")
            return []
    
    def _tfidf_search(self, query: str, top_k: int = 10) -> List[dict]:
        """TF-IDF 检索"""
        if self.tfidf_vectors is None:
            return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_vectors)[0]
            
            results = []
            for idx in range(len(self.documents)):
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "content": self.documents[idx],
                    "score": float(similarities[idx]),
                    "metadata": self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
                })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"❌ TF-IDF 检索失败: {e}")
            return []
    
    def _embedding_search(self, query: str, top_k: int = 10) -> List[dict]:
        """BGE 嵌入向量检索"""
        if self.doc_vectors is None or self.embeddings is None:
            return []
        
        try:
            query_vec = np.array(self.embeddings.embed_query(query))
            similarities = cosine_similarity([query_vec], self.doc_vectors)[0]
            
            results = []
            for idx in range(len(self.documents)):
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "content": self.documents[idx],
                    "score": float(similarities[idx]),
                    "metadata": self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
                })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"❌ BGE 嵌入检索失败: {e}")
            return []
    
    def _rrf_rerank(self, bm25_results: List[dict], tfidf_results: List[dict], 
                    embedding_results: List[dict], top_k: int = 3) -> List[dict]:
        """
        RRF (Reciprocal Rank Fusion) 重排序
        融合 BM25、TF-IDF 和嵌入向量的检索结果
        """
        rrf_scores = {}
        doc_content_map = {r["doc_id"]: r["content"] for r in bm25_results + tfidf_results + embedding_results}
        
        # BM25 结果贡献
        for rank, result in enumerate(bm25_results):
            doc_id = result["doc_id"]
            score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        
        # TF-IDF 结果贡献
        for rank, result in enumerate(tfidf_results):
            doc_id = result["doc_id"]
            score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        
        # 嵌入结果贡献
        for rank, result in enumerate(embedding_results):
            doc_id = result["doc_id"]
            score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        
        # 按 RRF 分数排序
        sorted_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_id, rrf_score in sorted_doc_ids[:top_k]:
            if doc_id in doc_content_map:
                final_results.append({
                    "doc_id": doc_id,
                    "content": doc_content_map[doc_id],
                    "score": rrf_score,
                    "metadata": self.doc_metadata[doc_id] if doc_id < len(self.doc_metadata) else {}
                })
        
        return final_results
    
    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """
        混合检索：BM25 + TF-IDF + BGE 嵌入 + RRF 重排序
        """
        # 检查知识库是否更新
        if self.knowledge_manager.reload_if_modified():
            logger.info("🔄 检测到知识文件更新，重建知识库索引")
            self._init_knowledge()
        
        # BM25 检索
        bm25_results = self._bm25_score(query)
        
        # TF-IDF 检索
        tfidf_results = self._tfidf_search(query, top_k=10)
        
        # BGE 嵌入检索
        embedding_results = self._embedding_search(query, top_k=10)
        
        # 合并结果
        all_results = bm25_results + tfidf_results + embedding_results
        if not all_results:
            return []
        
        # 如果只有一种结果
        if not tfidf_results and not embedding_results:
            return bm25_results[:top_k]
        if not bm25_results and not embedding_results:
            return tfidf_results[:top_k]
        if not bm25_results and not tfidf_results:
            return embedding_results[:top_k]
        
        # RRF 重排序
        return self._rrf_rerank(bm25_results, tfidf_results, embedding_results, top_k)
    
    def get_context(self, query: str) -> str:
        """获取检索上下文字符串"""
        results = self.search(query)
        if not results:
            return ""
        return "\n".join([f"- {r['content']}" for r in results])
    
    def add_document(self, content: str, category: str = "general", source: str = "manual"):
        """动态添加文档"""
        doc_id = len(self.documents)
        self.documents.append(content)
        self.doc_ids.append(doc_id)
        self.doc_metadata.append({"content": content, "category": category, "source": source})
        
        # 重建索引
        self._build_index()
        logger.info(f"➕ 添加文档: {content[:50]}...")
    
    def reload(self):
        """重新加载知识库"""
        self._init_knowledge()


# ============================================================
# 工具定义和实现
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_size_recommendation",
            "description": "根据用户性别、身高和体重推荐最合适的服装尺码",
            "parameters": {
                "type": "object",
                "properties": {
                    "gender": {"type": "string", "description": "用户性别，男或女"},
                    "height": {"type": "integer", "description": "身高cm"},
                    "weight": {"type": "integer", "description": "体重kg"}
                },
                "required": ["gender", "height", "weight"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询订单状态和物流信息，返回订单商品明细（含商品图片链接）",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "查询商品库存",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "size": {"type": "string"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_products",
            "description": "推荐商品，返回商品名称、价格、描述、图片链接和购买链接；默认先给多款推荐，至少 3 款，只有用户明确要求时再展示链接和图片",
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {"type": "string", "description": "风格"},
                    "gender": {"type": "string", "description": "用户性别（男/女），可选"},
                    "limit": {"type": "integer", "description": "返回商品数量，默认3，建议2-3"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_coupon",
            "description": "获取优惠券",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "description": "会员等级"}
                }
            }
        }
    }
]


class ToolExecutor:

    _product_catalog_cache: Optional[List[dict]] = None
    _product_catalog_mtime: float = 0.0

    @staticmethod
    def _product_info_path() -> Path:
        return Path(__file__).resolve().parent / "knowledge" / "product_info.txt"

    @staticmethod
    def _normalize_gender(gender: str) -> str:
        g = str(gender or "").strip().lower()
        if g in ["男", "男生", "先生", "male", "man"]:
            return "男"
        if g in ["女", "女生", "女士", "小姐", "female", "woman"]:
            return "女"
        return ""

    @staticmethod
    def _load_product_catalog(force_reload: bool = False) -> List[dict]:
        path = ToolExecutor._product_info_path()
        if not path.exists():
            return []

        mtime = path.stat().st_mtime
        if (not force_reload and ToolExecutor._product_catalog_cache is not None
                and mtime <= ToolExecutor._product_catalog_mtime):
            return ToolExecutor._product_catalog_cache

        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines()]
        products: List[dict] = []
        current: Dict[str, Any] = {}
        current_gender = ""
        current_category = ""

        def flush_product():
            if not current.get("name"):
                return

            price_raw = str(current.get("price", "")).replace("￥", "").strip()
            try:
                price_value: Any = float(price_raw) if "." in price_raw else int(price_raw)
            except Exception:
                price_value = price_raw

            material = str(current.get("material", "")).strip()
            color = str(current.get("color", "")).strip()
            selling_point = str(current.get("selling_point", "")).strip()
            desc_parts = []
            if selling_point:
                desc_parts.append(selling_point)
            if material:
                desc_parts.append(f"材质：{material}")
            if color:
                desc_parts.append(f"颜色：{color}")

            products.append({
                "name": current.get("name", ""),
                "price": price_value,
                "desc": "，".join(desc_parts) if desc_parts else "品质好，穿着舒适",
                "image_url": current.get("image_url", ""),
                "shop_url": current.get("shop_url", ""),
                "style_tags": [current_category, material, color, selling_point],
                "gender": current_gender,
                "category": current_category
            })

        for line in lines:
            if not line:
                continue

            if line.startswith("一、") or line.startswith("二、") or line.startswith("三、") or line.startswith("四、") or line.startswith("五、") or line.startswith("六、"):
                flush_product()
                current = {}
                current_category = line
                if "男装" in line:
                    current_gender = "男"
                elif "女装" in line:
                    current_gender = "女"
                else:
                    current_gender = ""
                continue

            if re.match(r"^\d+\.\s*商品名称：", line):
                flush_product()
                current = {
                    "name": line.split("商品名称：", 1)[1].strip(),
                    "gender": current_gender,
                    "category": current_category,
                }
                continue

            if not current:
                continue

            if "价格：" in line:
                current["price"] = line.split("价格：", 1)[1].strip()
            elif "颜色：" in line:
                current["color"] = line.split("颜色：", 1)[1].strip()
            elif "尺码：" in line:
                current["size"] = line.split("尺码：", 1)[1].strip()
            elif "材质：" in line:
                current["material"] = line.split("材质：", 1)[1].strip()
            elif "卖点：" in line:
                current["selling_point"] = line.split("卖点：", 1)[1].strip()
            elif "图片：" in line:
                current["image_url"] = line.split("图片：", 1)[1].strip()
            elif "链接：" in line:
                current["shop_url"] = line.split("链接：", 1)[1].strip()

        flush_product()

        ToolExecutor._product_catalog_cache = products
        ToolExecutor._product_catalog_mtime = mtime
        return products

    @staticmethod
    def list_products(force_reload: bool = False) -> List[dict]:
        """对外暴露统一商品目录，供 API/工具复用，避免重复解析。"""
        return ToolExecutor._load_product_catalog(force_reload=force_reload)
    
    @staticmethod
    def get_size_recommendation(height: int, weight: int, gender: str) -> str:
        gender_value = str(gender or "").strip().lower()
        if height <= 0 or weight <= 0:
            return json.dumps({"success": False, "error": "请提供有效的身高和体重信息。"})

        if gender_value in ["男", "男生", "先生", "male", "man"]:
            if height < 160:
                size = "XS" if weight < 50 else "S" if weight < 60 else "M"
            elif height < 170:
                size = "S" if weight < 55 else "M" if weight < 70 else "L"
            elif height < 180:
                size = "M" if weight < 65 else "L" if weight < 80 else "XL"
            else:
                size = "L" if weight < 75 else "XL" if weight < 90 else "XXL"
        elif gender_value in ["女", "女生", "女士", "小姐", "female", "woman"]:
            if height < 155:
                size = "XS" if weight < 45 else "S" if weight < 55 else "M"
            elif height < 165:
                size = "S" if weight < 50 else "M" if weight < 60 else "L"
            elif height < 175:
                size = "M" if weight < 60 else "L" if weight < 70 else "XL"
            else:
                size = "L" if weight < 70 else "XL" if weight < 80 else "XXL"
        else:
            return json.dumps({"success": False, "error": "缺少性别信息，请提供男或女，以便推荐更准确的尺码。"})

        return json.dumps({"success": True, "size": size, "advice": f"推荐{size}码"})
    
    @staticmethod
    def query_order(order_id: str) -> str:
        orders = {
            "ORD123456": {
                "status": "已签收",
                "total": 59,
                "items": [
                    {
                        "name": "纯棉T恤",
                        "price": 59,
                        "quantity": 1,
                        "image_url": "/images/products/tshirt.png",
                        "shop_url": "https://shop.fashionai.com/products/tshirt-cotton"
                    }
                ]
            },
            "ORD123457": {
                "status": "配送中",
                "total": 99,
                "items": [
                    {
                        "name": "牛仔裤",
                        "price": 99,
                        "quantity": 1,
                        "image_url": "/images/products/jeans.png",
                        "shop_url": "https://shop.fashionai.com/products/jeans-classic"
                    }
                ]
            },
            "ORD123458": {
                "status": "已付款",
                "total": 129,
                "items": [
                    {
                        "name": "针织开衫",
                        "price": 129,
                        "quantity": 1,
                        "image_url": "/images/products/cardigan.png",
                        "shop_url": "https://shop.fashionai.com/products/cardigan-knit"
                    }
                ]
            }
        }
        order = orders.get(order_id.upper())
        if order:
            return json.dumps({"success": True, "order_id": order_id.upper(), **order})
        return json.dumps({"success": False, "error": "订单不存在"})
    
    @staticmethod
    def check_stock(product_name: str, size: str = "M") -> str:
        stock_db = {
            "T恤": {"XS": 30, "S": 50, "M": 100, "L": 80, "XL": 40},
            "牛仔裤": {"28": 30, "30": 60, "32": 80, "34": 40},
            "卫衣": {"S": 40, "M": 80, "L": 60, "XL": 30}
        }
        for name, sizes in stock_db.items():
            if product_name in name:
                qty = sizes.get(size, 0)
                return json.dumps({"success": True, "product": name, "size": size, "stock": qty, "available": qty > 0})
        return json.dumps({"success": False, "error": "商品不存在"})
    
    @staticmethod
    def recommend_products(style: str = "", gender: str = "", limit: int = 3) -> str:
        style_value = str(style or "").strip().lower()
        gender_value = ToolExecutor._normalize_gender(gender)

        products = ToolExecutor._load_product_catalog()
        if not products:
            return json.dumps({"success": False, "error": "商品库为空，请检查 product_info.txt"})

        style_alias_map = {
            "紧身裤": {"category": "裤", "fit": ["紧身", "弹力", "贴身", "修身", "运动", "骑行"]},
            "打底裤": {"category": "裤", "fit": ["紧身", "弹力", "贴身", "打底"]},
            "瑜伽裤": {"category": "裤", "fit": ["瑜伽", "高腰", "提臀", "弹力", "紧身"]},
            "运动裤": {"category": "裤", "fit": ["运动", "束脚", "弹力", "速干"]},
            "阔腿裤": {"category": "裤", "fit": ["阔腿", "显瘦", "垂坠", "高腰"]},
            "直筒裤": {"category": "裤", "fit": ["直筒", "修饰腿型", "百搭"]},
            "修身裤": {"category": "裤", "fit": ["修身", "显瘦", "弹力"]},
            "裤子": {"category": "裤", "fit": []},
        }

        if gender_value:
            gender_filtered = [p for p in products if p.get("gender") == gender_value]
            products = gender_filtered or products

        if style_value:
            style_hint = style_alias_map.get(style_value, {})
            category_hint = str(style_hint.get("category", "") or "").lower()
            fit_tokens = [str(token).lower() for token in style_hint.get("fit", [])]

            scored_products = []
            for product in products:
                name = str(product.get("name", "")).lower()
                desc = str(product.get("desc", "")).lower()
                tags = [str(tag or "").lower() for tag in product.get("style_tags", [])]
                category = str(product.get("category", "")).lower()

                score = 0
                if style_value and style_value in name:
                    score += 6
                if style_value and style_value in desc:
                    score += 4
                if style_value and any(style_value in tag for tag in tags):
                    score += 4

                if category_hint and category_hint in category:
                    score += 5
                if category_hint and category_hint in name:
                    score += 3

                if fit_tokens:
                    fit_hit = any(token in name or token in desc or any(token in tag for tag in tags) for token in fit_tokens)
                    if fit_hit:
                        score += 3

                if style_value in {"裤子", "紧身裤", "打底裤", "瑜伽裤", "运动裤", "阔腿裤", "直筒裤", "修身裤"}:
                    if "裤" in category or "裤" in name:
                        score += 4

                if score > 0:
                    scored_products.append((score, product))

            if scored_products:
                scored_products.sort(key=lambda item: (-item[0], str(item[1].get("name", ""))))
                products = [item[1] for item in scored_products]

        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 3
        limit_value = max(1, min(6, limit_value))

        # 只返回前6款，避免响应过长
        return json.dumps({
            "success": True,
            "style": style or "通用",
            "gender": gender_value or "未指定",
            "products": products[:limit_value]
        }, ensure_ascii=False)
    
    @staticmethod
    def get_coupon(level: str = "普通") -> str:
        coupons = {
            "新用户": [{"name": "新人券", "condition": "满50减10"}],
            "普通": [{"name": "会员券", "condition": "满100减10"}],
            "VIP": [{"name": "VIP券", "condition": "满100减20"}]
        }
        return json.dumps({"success": True, "coupons": coupons.get(level, coupons["普通"])})
    
    @staticmethod
    def execute(name: str, args: dict) -> str:
        func = getattr(ToolExecutor, name, None)
        if func:
            try:
                safe_args = args if isinstance(args, dict) else {}
                signature = inspect.signature(func)
                parameters = signature.parameters
                accepts_var_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in parameters.values()
                )

                if accepts_var_kwargs:
                    filtered_args = safe_args
                    ignored_args = []
                else:
                    allowed_keys = {
                        key for key, param in parameters.items()
                        if param.kind in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        )
                    }
                    filtered_args = {k: v for k, v in safe_args.items() if k in allowed_keys}
                    ignored_args = [k for k in safe_args.keys() if k not in allowed_keys]

                if ignored_args:
                    logger.warning(
                        f"⚠️ 工具 {name} 忽略不支持参数: {ignored_args}"
                    )

                return func(**filtered_args)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})
        return json.dumps({"success": False, "error": "未知工具"})


# ============================================================
# 情绪分析器
# ============================================================

_emotion_llm_client: Optional[ChatOpenAI] = None
_emotion_result_cache: Dict[str, tuple[float, dict]] = {}
_emotion_cache_ttl_seconds = 180
_emotion_cache_max_entries = 200


def _get_emotion_llm() -> Optional[ChatOpenAI]:
    """获取情绪分析 LLM（连接复用）。"""
    global _emotion_llm_client
    if _emotion_llm_client is not None:
        return _emotion_llm_client

    try:
        _emotion_llm_client = ChatOpenAI(
            model="glm-4-flash",
            temperature=0.1,
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        return _emotion_llm_client
    except Exception as e:
        logger.warning(f"⚠️ 情绪分析 LLM 初始化失败: {e}")
        _emotion_llm_client = None
        return None


def _cleanup_emotion_cache():
    now = time.time()
    expired = [k for k, (ts, _) in _emotion_result_cache.items() if now - ts > _emotion_cache_ttl_seconds]
    for k in expired:
        _emotion_result_cache.pop(k, None)

    if len(_emotion_result_cache) > _emotion_cache_max_entries:
        oldest_keys = sorted(_emotion_result_cache.items(), key=lambda item: item[1][0])
        over = len(_emotion_result_cache) - _emotion_cache_max_entries
        for k, _ in oldest_keys[:over]:
            _emotion_result_cache.pop(k, None)


def _cached_llm_analyze(message: str) -> dict:
    """LLM 情绪分析（连接复用 + 结果缓存）"""
    try:
        text = str(message or "").strip()
        if not text:
            return {"emotion": "中性", "confidence": 0.0, "needs_empathy": False}

        _cleanup_emotion_cache()
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = _emotion_result_cache.get(cache_key)
        if cached and (time.time() - cached[0] <= _emotion_cache_ttl_seconds):
            return cached[1]

        llm = _get_emotion_llm()
        if llm is None:
            return {"emotion": "中性", "confidence": 0.0, "needs_empathy": False}
        
        prompt = f"分析情绪：{text}"
        
        messages = [
            SystemMessage(content="你是情绪分析助手。请判断用户情绪属于：愤怒/焦急/失望/满意/中性，只需给结论短语。"),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        content = str(response.content).strip()

        result = {"emotion": "中性", "confidence": 0.5, "needs_empathy": False}
        
        if "愤怒" in content:
            result = {"emotion": "愤怒", "confidence": 0.8, "needs_empathy": True}
        elif "焦急" in content:
            result = {"emotion": "焦急", "confidence": 0.8, "needs_empathy": True}
        elif "失望" in content:
            result = {"emotion": "失望", "confidence": 0.75, "needs_empathy": True}
        elif "满意" in content:
            result = {"emotion": "满意", "confidence": 0.8, "needs_empathy": False}

        _emotion_result_cache[cache_key] = (time.time(), result)
        return result

    except Exception as e:
        logger.warning(f"⚠️ 情绪分析失败，回退中性: {e}")
        return {"emotion": "中性", "confidence": 0.0, "needs_empathy": False}


class EmotionAnalyzer:
    """情绪分析器"""
    
    EMOTION_KEYWORDS = {
        "愤怒": ["垃圾", "差评", "投诉", "生气", "太差", "烂"],
        "焦急": ["快点", "急", "马上", "尽快", "催"],
        "失望": ["失望", "不好", "一般", "不行"],
        "满意": ["满意", "不错", "很好", "喜欢", "赞"]
    }
    
    @classmethod
    def _rule_based_analyze(cls, text: str) -> dict:
        text_lower = text.lower()
        
        for emotion, keywords in cls.EMOTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return {"emotion": emotion, "confidence": 0.7}
        
        return {"emotion": None, "confidence": 0.0}
    
    @classmethod
    def analyze(cls, text: str) -> dict:
        rule_result = cls._rule_based_analyze(text)
        
        if rule_result["confidence"] >= 0.7:
            return {
                "emotion": rule_result["emotion"],
                "needs_empathy": rule_result["emotion"] in ["愤怒", "焦急", "失望"]
            }
        
        llm_result = _cached_llm_analyze(text)
        return {
            "emotion": llm_result["emotion"],
            "needs_empathy": llm_result["needs_empathy"]
        }
    
    @classmethod
    def get_empathy(cls, emotion: str, message: str) -> str:
        if not emotion:
            return ""

        return ""


class SizeIntentAnalyzer:
    """简单的购买/退换货/浏览意图分类器"""

    BUYING_KEYWORDS = [
        "购买", "买", "加到购物车", "下单", "推荐尺码", "尺码推荐", "优惠券", "折扣", "价格"
    ]
    RETURNING_KEYWORDS = [
        "退货", "换货", "退款", "退换", "退换货", "质量问题", "不合适", "太小", "太大"]
    BROWSING_KEYWORDS = [
        "推荐", "款式", "搭配", "看看", "热销", "新品", "风格"
    ]

    @classmethod
    def classify(cls, user_message: str, user_history: list[dict] | None = None) -> dict:
        text = user_message.lower()

        if any(kw in text for kw in cls.RETURNING_KEYWORDS):
            return {"intent": "RETURNING"}
        if any(kw in text for kw in cls.BUYING_KEYWORDS):
            return {"intent": "BUYING"}
        if any(kw in text for kw in cls.BROWSING_KEYWORDS):
            return {"intent": "BROWSING"}

        if "退" in text or "换" in text:
            return {"intent": "RETURNING"}
        if "买" in text or "购买" in text or "购物车" in text or "下单" in text:
            return {"intent": "BUYING"}

        return {"intent": "BROWSING"}


# ============================================================
# 图片分析器
# ============================================================

class ImageAnalyzer:
    
    @staticmethod
    def analyze(image_base64: str, mime_type: str = "image/jpeg", retry_count: int = 3) -> str | None:
        config = Config()
        
        if not config.DOUBAO_API_KEY:
            return "需要配置API Key"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.DOUBAO_API_KEY}"
        }
        
        prompt = "分析服装图片：类型、颜色、款式、搭配建议"
        
        data = {
            "model": config.DOUBAO_VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(config.DOUBAO_API_URL, headers=headers, json=data, timeout=30)
                result = response.json()
                
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
            except Exception:
                if attempt < retry_count - 1:
                    time.sleep(config.RETRY_DELAY)
        
        return None


# ============================================================
# 智能体核心
# ============================================================

class AgentState(Enum):
    IDLE = "空闲"
    PLANNING = "规划中"
    EXECUTING = "执行中"
    OBSERVING = "观察中"
    REFLECTING = "反思中"
    DONE = "完成"


@dataclass
class Action:
    tool_name: str
    arguments: dict
    result: str | None = None


class AgentStrategy(ABC):
    """智能体策略抽象基类：统一不同策略的对外调用接口。"""

    @abstractmethod
    def run(self, user_message: str, user_info: dict, image_context: str = "",
            history: list | None = None, precomputed_rag_context: str = "",
            speed_mode: str = "balanced") -> str:
        """执行一次策略推理并返回最终回复。"""
        raise NotImplementedError

    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """可选：返回推理轨迹；默认无轨迹。"""
        return []


class ReActAgent(AgentStrategy):
    """
    完整的 ReAct 智能体（Reasoning + Acting + Reflection）
    
    实现真正的 ReAct 循环：
    1. Thought（思考）：分析当前情况
    2. Action（行动）：调用工具
    3. Observation（观察）：获取工具执行结果
    4. Reflection（反思）：评估是否需要继续或给出答案
    """
    
    # ReAct 状态枚举
    class State(Enum):
        THINK = "think"      # 思考阶段
        ACT = "act"          # 行动阶段
        OBSERVE = "observe"  # 观察阶段
        REFLECT = "reflect"  # 反思阶段
        DONE = "done"        # 完成
    
    def __init__(self, llm: ChatOpenAI, knowledge_base: VectorKnowledgeBase):
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.tool_executor = ToolExecutor()
        self.max_iterations = 5
        self.max_reflection_rounds = 3  # 最大反思轮次
        self.reasoning_trace: List[Dict] = []  # 推理轨迹记录
        
        # 绑定工具的 LLM
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        
        # 工具名称映射
        self.tool_mapping = {
            '尺码': 'get_size_recommendation', '订单': 'query_order',
            '库存': 'check_stock',
            '推荐': 'recommend_products', '优惠券': 'get_coupon',
            'get_size_recommendation': 'get_size_recommendation',
            'query_order': 'query_order', 'check_stock': 'check_stock',
            'recommend_products': 'recommend_products', 'get_coupon': 'get_coupon',
        }
    
    def _get_system_prompt(self, user_info: dict, rag_context: str, image_context: str) -> str:
        """生成系统提示词"""
        user_info_str = f"性别：{user_info.get('gender', '未知')}，身高：{user_info.get('height', '未知')}cm，体重：{user_info.get('weight', '未知')}kg"
        
        image_section = (f"## 图片分析结果\n{image_context}") if image_context else ""
        prompt = f"""你是 FashionAI 智能客服，使用 ReAct（Reasoning + Acting + Reflection）模式。

## 用户信息
{user_info_str}

请注意：系统初始状态不知道用户的性别、身高和体重。只有当用户在对话中明确提供这些信息时，才将其记录为用户资料。如果用户的请求与尺码或搭配相关，而且缺少性别、身高或体重中的任意一项，先主动询问这些信息，再给出推荐。

## 知识库参考
{rag_context if rag_context else "暂无相关知识"}

{image_section}

## 可用工具
- get_size_recommendation: 根据性别、身高和体重推荐尺码
- query_order: 查询订单状态
- check_stock: 查询商品库存
- recommend_products: 推荐商品
- get_coupon: 获取优惠券

## ReAct 输出格式
请严格按以下格式输出：

思考：<分析用户问题，确定需要采取的行动>
行动：<工具名称>(参数)
观察：<工具执行结果>
反思：<评估结果，决定是否需要继续推理或给出最终答案>
最终答案：<最终回答> 或 继续推理

## 示例
思考：用户想知道175cm体重70kg穿什么尺码，需要调用尺码推荐工具
行动：get_size_recommendation(height=175, weight=70)
观察：{{"success": true, "size": "L", "advice": "推荐L码"}}
反思：工具返回了L码推荐，这正好回答了用户问题
最终答案：根据您的身高175cm和体重70kg，推荐L码
"""
        return prompt
    
    def _get_reflection_prompt(self, question: str, context: str) -> str:
        """生成反思提示词"""
        return f"""基于以下信息进行反思：

问题：{question}
上下文：{context}

请判断：
1. 当前信息是否足以回答用户问题？
2. 是否需要调用更多工具？
3. 如果需要继续，应该调用什么工具？

如果问题已解决，回答"答案完整"。
如果需要继续推理，提出下一个行动。"""
    
    def _parse_llm_response(self, response: Any) -> Dict[str, Any]:
        """解析 LLM 响应（支持工具调用和文本响应）"""
        result = {
            "type": "text",
            "content": "",
            "tool_calls": [],
            "raw": response
        }
        
        # 检查是否有工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            result["type"] = "tool_call"
            result["tool_calls"] = response.tool_calls
            result["content"] = response.content if hasattr(response, 'content') else ""
            return result
        
        # 纯文本响应
        content = response.content if hasattr(response, 'content') else str(response)
        result["content"] = content
        
        # 尝试解析结构化输出
        parsed = self._parse_structured_output(content)
        if parsed:
            return {**result, **parsed}
        
        return result
    
    def _parse_structured_output(self, text: str) -> Dict[str, Any]:
        """
        解析结构化输出（思考、行动、观察、反思、最终答案）
        改进的解析逻辑，更加健壮
        """
        result = {
            "thought": "",
            "action": None,
            "observation": "",
            "reflection": "",
            "final_answer": None,
            "should_continue": True
        }
        
        if not text:
            return result
        
        # 使用正则表达式匹配各个部分
        import re
        
        # 匹配思考部分
        thought_match = re.search(r'思考[：:]\s*(.+?)(?=\n行动|\n观察|\n反思|\n最终答案|$)', text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # 匹配行动部分
        action_match = re.search(r'行动[：:]\s*(.+?)(?=\n观察|\n反思|\n最终答案|$)', text, re.DOTALL)
        if action_match:
            action_str = action_match.group(1).strip()
            if '无需调用' not in action_str and '不需要' not in action_str:
                parsed_action = self._parse_action_string(action_str)
                if parsed_action:
                    result["action"] = parsed_action
        
        # 匹配观察部分
        obs_match = re.search(r'观察[：:]\s*(.+?)(?=\n反思|\n最终答案|$)', text, re.DOTALL)
        if obs_match:
            result["observation"] = obs_match.group(1).strip()
        
        # 匹配反思部分
        refl_match = re.search(r'反思[：:]\s*(.+?)(?=\n最终答案|$)', text, re.DOTALL)
        if refl_match:
            result["reflection"] = refl_match.group(1).strip()
        
        # 匹配最终答案
        answer_match = re.search(r'最终答案[：:]\s*(.+)', text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            result["final_answer"] = answer
            # 检查是否继续推理
            if any(keyword in answer for keyword in ['继续推理', '需要更多信息', '继续分析']):
                result["should_continue"] = True
            else:
                result["should_continue"] = False
        
        return result
    
    def _parse_action_string(self, action_str: str) -> Optional[Dict]:
        """解析行动字符串，提取工具名称和参数"""
        import re
        
        # 匹配工具调用格式：tool_name(arg1=value1, arg2=value2)
        # 或：工具名称 参数
        call_match = re.search(r'(\w+)\s*\((.*)\)', action_str, re.DOTALL)
        
        if call_match:
            tool_name = call_match.group(1)
            args_str = call_match.group(2)
            
            # 解析参数
            args = {}
            param_matches = re.findall(r'(\w+)\s*=\s*([^,)]+)', args_str)
            for key, value in param_matches:
                value = value.strip().strip('"\'')
                if value.isdigit():
                    args[key] = int(value)
                else:
                    args[key] = value
            
            tool_name = self.tool_mapping.get(tool_name, tool_name)
            if tool_name in self.tool_mapping.values():
                return {"name": tool_name, "args": args}
        
        return None
    
    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """执行工具并返回结果"""
        try:
            result = self.tool_executor.execute(tool_name, args)
            return result
        except Exception as e:
            logger.error(f"❌ 工具执行失败 {tool_name}: {e}")
            return json.dumps({"success": False, "error": str(e)})
    
    def _should_continue(self, reflection: str, final_answer: Optional[str], 
                         tool_result: Optional[str], round_num: int) -> bool:
        """
        判断是否应该继续推理
        
        停止条件：
        1. 已经得到最终答案
        2. 反思表明问题已解决
        3. 达到最大轮次
        4. 工具执行失败且无法恢复
        """
        # 达到最大轮次
        if round_num >= self.max_iterations:
            return False
        
        # 有最终答案
        if final_answer and not any(kw in final_answer for kw in ['继续推理', '需要', '下一步']):
            return False
        
        # 反思表明问题已解决
        if reflection:
            if any(phrase in reflection for phrase in ['问题已解决', '答案完整', '可以回答', '已获取足够信息']):
                return False
        
        # 工具执行失败
        if tool_result:
            try:
                data = json.loads(tool_result)
                if not data.get("success", True) and "error" in data:
                    # 如果是工具执行错误，且反思中没有好的替代方案，则停止
                    if reflection and '尝试' in reflection:
                        return True
                    return False
            except:
                pass
        
        return True
    
    def run(self, user_message: str, user_info: dict, image_context: str = "",
            history: list | None = None, precomputed_rag_context: str = "",
            speed_mode: str = "balanced") -> str:
        """
        执行完整的 ReAct 推理循环
        
        Args:
            user_message: 用户问题
            user_info: 用户信息（身高、体重等）
            image_context: 图片分析上下文
        
        Returns:
            最终回答
        """
        self.reasoning_trace = []
        
        # 获取 RAG 上下文
        rag_context = self.knowledge_base.get_context(user_message)
        
        # 构建消息历史
        system_prompt = self._get_system_prompt(user_info, rag_context, image_context)
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请回答以下问题：{user_message}")
        ]
        
        current_state = self.State.THINK
        tool_result = None
        iteration = 0
        empty_response_retries = 0
        
        try:
            while iteration < self.max_iterations:
                iteration += 1
                logger.info(f"🔄 ReAct 第 {iteration} 轮迭代")
                
                # 1. 思考 + 行动阶段：调用 LLM
                response = self.llm_with_tools.invoke(messages)
                parsed = self._parse_llm_response(response)
                
                # 记录推理轨迹
                trace_entry = {
                    "iteration": iteration,
                    "state": current_state.value,
                    "response_content": parsed.get("content", ""),
                    "tool_calls": parsed.get("tool_calls", [])
                }

                content_text = str(parsed.get("content", "") or "").strip()
                
                # 2. 处理工具调用
                if parsed["type"] == "tool_call" and parsed["tool_calls"]:
                    for tool_call in parsed["tool_calls"]:
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('args', {})
                        
                        logger.info(f"🔧 调用工具: {tool_name}({tool_args})")
                        
                        # 执行工具
                        tool_result = self._execute_tool(tool_name, tool_args)
                        
                        # 添加到消息历史
                        messages.append(AIMessage(
                            content=parsed.get("content", ""),
                            tool_calls=[tool_call]
                        ))
                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call.get('id', f'call_{iteration}')
                        ))
                        
                        trace_entry["tool_name"] = tool_name
                        trace_entry["tool_result"] = tool_result
                
                # 3. 处理文本响应（包含思考、行动、观察、反思、最终答案）
                elif content_text:
                    parsed_struct = self._parse_structured_output(content_text)
                    
                    trace_entry["parsed"] = parsed_struct
                    
                    # 检查是否有最终答案
                    if parsed_struct.get("final_answer") and not parsed_struct.get("should_continue", True):
                        self.reasoning_trace.append(trace_entry)
                        logger.info(f"✅ ReAct 完成，最终答案: {parsed_struct['final_answer'][:50]}...")
                        return parsed_struct["final_answer"]
                    
                    # 检查是否应该继续
                    if not self._should_continue(
                        parsed_struct.get("reflection", ""),
                        parsed_struct.get("final_answer"),
                        tool_result,
                        iteration
                    ):
                        self.reasoning_trace.append(trace_entry)
                        # 如果没有最终答案但需要结束，尝试从内容中提取
                        if parsed_struct.get("final_answer"):
                            return parsed_struct["final_answer"]
                        return parsed.get("content", "抱歉，无法处理您的问题。")
                    
                    # 添加 AI 响应到消息历史
                    messages.append(AIMessage(content=content_text))
                    
                    # 如果 LLM 返回了结构化输出但没有工具调用，让它继续
                    if parsed_struct.get("action") is None and parsed_struct.get("thought"):
                        # 添加一个观察提示
                        observation_hint = f"观察：{tool_result}" if tool_result else "观察：暂无"
                        reflection_hint = f"反思：基于以上分析，{'问题已解决' if tool_result else '需要继续分析'}"
                        
                        messages.append(HumanMessage(
                            content=f"{observation_hint}\n{reflection_hint}"
                        ))
                        tool_result = None
                else:
                    # LLM 没有返回有效响应
                    raw_content = getattr(response, "content", None)
                    tool_calls_count = len(getattr(response, "tool_calls", []) or [])
                    logger.warning(
                        f"⚠️ LLM 未返回有效响应（轮次={iteration}, raw_content={repr(raw_content)}, tool_calls={tool_calls_count}, retries={empty_response_retries}）"
                    )

                    if empty_response_retries < 1:
                        empty_response_retries += 1
                        # 给模型一次恢复机会，强制要求直接输出最终答案。
                        messages.append(HumanMessage(
                            content="你上一轮输出为空。请直接给出最终答案；若信息不足，请明确指出缺少哪些信息。"
                        ))
                        self.reasoning_trace.append(trace_entry)
                        continue

                    if iteration >= 2:  # 经过几轮尝试后，尝试给出答案
                        break
                
                self.reasoning_trace.append(trace_entry)
            
            # 达到最大迭代次数，尝试从历史中提取答案
            if self.reasoning_trace:
                for trace in reversed(self.reasoning_trace):
                    if trace.get("parsed", {}).get("final_answer"):
                        return trace["parsed"]["final_answer"]
                    if trace.get("tool_result"):
                        try:
                            data = json.loads(trace["tool_result"])
                            if data.get("success"):
                                return f"根据查询结果：{trace['tool_result']}"
                        except:
                            pass
            
            return "问题较复杂，请分步提问或提供更多信息"
            
        except Exception as e:
            logger.error(f"❌ ReAct 执行异常: {e}")
            return f"处理出错：{str(e)}"
    
    def get_reasoning_trace(self) -> List[Dict]:
        """获取推理轨迹"""
        return self.reasoning_trace


def _extract_json_payload(text: Any) -> Optional[dict]:
    """从 LLM 输出中尽量提取 JSON 对象。"""
    if isinstance(text, dict):
        return text
    if not text:
        return None

    raw = str(text).strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw)

    candidates = [raw]
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    return None


def _shorten_text(text: Any, limit: int = 260) -> str:
    value = str(text or '').strip()
    if len(value) <= limit:
        return value
    return value[:limit - 3] + '...'


@dataclass
class PlannedStep:
    step_id: int
    objective: str
    tool_name: str = ''
    tool_args: dict = field(default_factory=dict)
    expected_result: str = ''
    rationale: str = ''


class PlannerAgent:
    """负责分析用户意图并生成执行计划。"""

    def __init__(self, llm: ChatOpenAI, knowledge_base: VectorKnowledgeBase):
        self.llm = llm
        self.knowledge_base = knowledge_base

    def _build_prompt(
        self,
        user_message: str,
        user_info: dict,
        rag_context: str,
        image_context: str,
        history: list | None,
        previous_critique: str = '',
        speed_mode: str = 'balanced',
    ) -> list[BaseMessage]:
        history_text = []
        for item in (history or [])[-8:]:
            role = '用户' if item.get('role') == 'user' else '助手'
            content = str(item.get('content', '')).strip()
            if content:
                history_text.append(f'{role}: {content}')
        history_block = '\n'.join(history_text) if history_text else '暂无'

        system_prompt = f"""你是 PlannerAgent，职责是分析用户意图并生成可执行计划。

你必须基于用户问题、用户画像、知识库与历史对话进行推理，然后只输出合法 JSON，不要输出 Markdown，不要输出解释性文本。

用户画像：
- 性别：{user_info.get('gender', '未知')}
- 身高：{user_info.get('height', '未知')}cm
- 体重：{user_info.get('weight', '未知')}kg

知识库参考：
{rag_context if rag_context else '暂无相关知识'}

图片上下文：
{image_context if image_context else '暂无图片上下文'}

历史对话：
{history_block}

上一轮评审意见：
{previous_critique if previous_critique else '无'}

输出要求：
- 只返回一个 JSON 对象
- steps 至少包含 1 个步骤，最多 4 个步骤
- 每个步骤必须包含 step_id, objective, tool_name, tool_args, expected_result, rationale
- tool_name 为空字符串表示该步骤仅依赖推理
- clarification_questions 最多 2 个
- response_mode 取值为 direct、tool、clarify、multi_step 之一

JSON 结构：
{{
  "intent_summary": "...",
  "response_mode": "direct",
  "confidence": 0.0,
  "clarification_questions": [],
  "steps": [
    {{
      "step_id": 1,
      "objective": "...",
      "tool_name": "",
      "tool_args": {{}},
      "expected_result": "...",
      "rationale": "..."
    }}
  ],
  "final_answer_focus": "...",
  "knowledge_usage": "..."
}}"""

        human_prompt = f"""用户问题：{user_message}
响应速度：{speed_mode}

请生成执行计划。"""
        return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    def create_plan(
        self,
        user_message: str,
        user_info: dict,
        rag_context: str,
        image_context: str,
        history: list | None = None,
        previous_critique: str = '',
        speed_mode: str = 'balanced',
    ) -> dict:
        if not self.llm:
            return {
                'intent_summary': 'LLM 不可用',
                'response_mode': 'direct',
                'confidence': 0.0,
                'clarification_questions': [],
                'steps': [],
                'final_answer_focus': '直接告知系统暂不可用',
                'knowledge_usage': '',
                'raw_response': '',
            }

        messages = self._build_prompt(user_message, user_info, rag_context, image_context, history, previous_critique, speed_mode)
        response_text = ''
        for retry in range(2):
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            parsed = _extract_json_payload(response_text)
            if parsed:
                parsed['raw_response'] = response_text
                return parsed

            messages = [
                SystemMessage(content='你是 PlannerAgent。请把上一条内容转换为严格合法的 JSON 对象，只输出 JSON，不要任何其他文字。'),
                HumanMessage(content=response_text),
            ]

        return {
            'intent_summary': '解析失败',
            'response_mode': 'direct',
            'confidence': 0.0,
            'clarification_questions': [],
            'steps': [],
            'final_answer_focus': '根据用户问题直接回复',
            'knowledge_usage': rag_context,
            'raw_response': response_text,
        }


class ExecutorAgent:
    """负责按照计划执行步骤并调用工具。"""

    def __init__(self, llm: ChatOpenAI, knowledge_base: VectorKnowledgeBase, tool_executor: ToolExecutor):
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.tool_executor = tool_executor

    def _build_step_prompt(
        self,
        user_message: str,
        user_info: dict,
        rag_context: str,
        image_context: str,
        history: list | None,
        plan: dict,
        step: dict,
        step_context: str,
        speed_mode: str,
    ) -> list[BaseMessage]:
        history_text = []
        for item in (history or [])[-8:]:
            role = '用户' if item.get('role') == 'user' else '助手'
            content = str(item.get('content', '')).strip()
            if content:
                history_text.append(f'{role}: {content}')
        history_block = '\n'.join(history_text) if history_text else '暂无'

        system_prompt = f"""你是 ExecutorAgent，职责是执行 PlannerAgent 给出的单步任务，并在必要时调用工具。

你必须先进行推理，再决定是否调用工具。只输出合法 JSON，不要输出 Markdown，不要输出解释性文本。

用户画像：
- 性别：{user_info.get('gender', '未知')}
- 身高：{user_info.get('height', '未知')}cm
- 体重：{user_info.get('weight', '未知')}kg

知识库参考：
{rag_context if rag_context else '暂无相关知识'}

图片上下文：
{image_context if image_context else '暂无图片上下文'}

历史对话：
{history_block}

当前计划：
{json.dumps(plan, ensure_ascii=False)}

当前步骤：
{json.dumps(step, ensure_ascii=False)}

已有上下文：
{step_context if step_context else '无'}

输出要求：
- 只返回一个 JSON 对象
- action_type 取值为 tool、respond、clarify 之一
- 如果 action_type 为 tool，则必须给出 tool_name 和 tool_args
- 如果 action_type 为 respond，则给出 assistant_message

JSON 结构：
{{
  "action_type": "tool",
  "tool_name": "recommend_products",
  "tool_args": {{"style": "...", "gender": "..."}},
  "reasoning": "...",
  "assistant_message": "",
  "step_summary": "..."
}}"""

        human_prompt = f"""用户问题：{user_message}
响应速度：{speed_mode}

请执行当前步骤。"""
        return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    def _build_summary_prompt(
        self,
        user_message: str,
        plan: dict,
        step: dict,
        directive: dict,
        tool_result: str,
        step_context: str,
        speed_mode: str,
    ) -> list[BaseMessage]:
        system_prompt = """你是 ExecutorAgent。请根据步骤、工具结果和上下文，生成自然、准确、口语化的步骤结论，只输出合法 JSON。"""
        human_prompt = json.dumps({
            'user_message': user_message,
            'plan': plan,
            'step': step,
            'directive': directive,
            'tool_result': tool_result,
            'step_context': step_context,
            'speed_mode': speed_mode,
            'output_schema': {
                'step_summary': '...',
                'assistant_message': '...',
            }
        }, ensure_ascii=False)
        return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    def _is_meta_or_instruction_text(self, text: str) -> bool:
        """识别非面向用户的任务描述文本。"""
        value = str(text or '').strip()
        if not value:
            return True

        if value.endswith('|'):
            return True

        normalized = re.sub(r'\s+', '', value)
        markers = [
            "根据用户", "了解用户", "询问用户", "提供具体", "执行步骤", "步骤", "继续推理",
            "推荐2-3款", "推荐23款", "计划", "执行", "objective", "step_summary"
        ]
        if any(m in normalized for m in markers):
            return True

        # 常见指令式语句（如“获取今天的天气信息|”）不是面向用户的最终回复
        if re.match(r'^(获取|查询|检查|调用|执行).*(信息|数据|结果)\|?$', value):
            return True

        if re.match(r'^(根据|需要|请先|先).*(推荐|提供|确认)', value):
            return True
        return False

    def _build_recommendation_reply_from_tool_result(self, tool_result: str, user_message: str) -> str:
        """基于推荐工具结果生成可直接返回给用户的话术。"""
        try:
            data = json.loads(str(tool_result or ''))
        except Exception:
            return ''

        if not isinstance(data, dict) or not data.get('success'):
            return ''

        products = data.get('products', [])
        if not isinstance(products, list) or not products:
            return ''

        picked = products[:3]
        user_text = str(user_message or '')
        if any(k in user_text for k in ["运动", "跑步", "健身", "训练"]):
            intro = "可以，运动场景我先给你挑了这几款更实穿的："
            closing = "你更常做跑步还是健身？我可以再按场景把面料和版型细化到1-2款。"
        else:
            intro = "可以的，我先给你推荐这几款："
            closing = "如果你告诉我预算和偏好的版型（宽松/修身），我再帮你缩到最合适的1-2款。"

        lines = [intro]
        for idx, item in enumerate(picked, 1):
            if not isinstance(item, dict):
                continue
            name = str(item.get('name', f'商品{idx}') or f'商品{idx}').strip()
            price = item.get('price', '')
            desc = str(item.get('desc', '') or '').strip()
            desc = re.split(r'[；;。.!！?？]', desc)[0].strip()
            tail = f"，{desc}" if desc else ""
            if price != '':
                lines.append(f"{idx}. {name}（¥{price}）{tail}")
            else:
                lines.append(f"{idx}. {name}{tail}")

        lines.append(closing)
        return "\n".join(lines).strip()

    def execute_plan(
        self,
        user_message: str,
        user_info: dict,
        rag_context: str,
        image_context: str,
        history: list | None,
        plan: dict,
        speed_mode: str = 'balanced',
    ) -> dict:
        if not self.llm:
            return {
                'step_results': [],
                'draft_answer': '抱歉，系统暂时不可用。',
                'tool_calls': [],
                'execution_summary': 'LLM 不可用',
            }

        step_results: list[dict] = []
        tool_calls: list[dict] = []
        accumulated_context = []
        planned_steps = plan.get('steps', []) if isinstance(plan, dict) else []

        for step_index, raw_step in enumerate(planned_steps, 1):
            if not isinstance(raw_step, dict):
                continue

            step = PlannedStep(
                step_id=int(raw_step.get('step_id', step_index) or step_index),
                objective=str(raw_step.get('objective', '') or ''),
                tool_name=str(raw_step.get('tool_name', '') or ''),
                tool_args=raw_step.get('tool_args', {}) if isinstance(raw_step.get('tool_args', {}), dict) else {},
                expected_result=str(raw_step.get('expected_result', '') or ''),
                rationale=str(raw_step.get('rationale', '') or ''),
            )

            logger.info(f"🔧 Executor 准备执行步骤 {step.step_id}: {step.objective}")
            step_context = '\n'.join(accumulated_context[-4:])
            directive_messages = self._build_step_prompt(
                user_message=user_message,
                user_info=user_info,
                rag_context=rag_context,
                image_context=image_context,
                history=history,
                plan=plan,
                step=raw_step,
                step_context=step_context,
                speed_mode=speed_mode,
            )

            directive_text = ''
            directive = {}
            for retry in range(2):
                response = self.llm.invoke(directive_messages)
                directive_text = response.content if hasattr(response, 'content') else str(response)
                directive = _extract_json_payload(directive_text) or {}
                if directive:
                    break
                directive_messages = [
                    SystemMessage(content='你是 ExecutorAgent。请将上一条内容转换为严格合法的 JSON 对象，只输出 JSON。'),
                    HumanMessage(content=directive_text),
                ]

            action_type = str(directive.get('action_type', 'respond') or 'respond').strip().lower()
            tool_result = ''
            assistant_message = str(directive.get('assistant_message', '') or '').strip()
            requested_tool_name = str(directive.get('tool_name', step.tool_name) or step.tool_name).strip()

            if action_type == 'tool' and requested_tool_name:
                tool_name = requested_tool_name
                tool_args = directive.get('tool_args', step.tool_args)
                if not isinstance(tool_args, dict):
                    tool_args = step.tool_args
                logger.info(f"⚙️ Executor 调用工具: {tool_name} | 参数: {tool_args}")
                tool_result = self.tool_executor.execute(tool_name, tool_args)
                tool_calls.append({
                    'step_id': step.step_id,
                    'tool_name': tool_name,
                    'tool_args': tool_args,
                    'tool_result': tool_result,
                })
                logger.info(f"📦 工具结果: {_shorten_text(tool_result)}")

                summary_messages = self._build_summary_prompt(
                    user_message=user_message,
                    plan=plan,
                    step=raw_step,
                    directive=directive,
                    tool_result=tool_result,
                    step_context=step_context,
                    speed_mode=speed_mode,
                )
                summary_response = self.llm.invoke(summary_messages)
                summary_text = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
                summary_json = _extract_json_payload(summary_text) or {}
                step_summary = str(summary_json.get('step_summary', '') or directive.get('step_summary', '') or '').strip()
                assistant_message = str(summary_json.get('assistant_message', '') or assistant_message or step_summary).strip()

                if tool_name == 'recommend_products':
                    fallback_message = self._build_recommendation_reply_from_tool_result(tool_result, user_message)
                    if fallback_message:
                        if (not assistant_message) or self._is_meta_or_instruction_text(assistant_message):
                            assistant_message = fallback_message
                        if (not step_summary) or self._is_meta_or_instruction_text(step_summary):
                            step_summary = fallback_message
            else:
                step_summary = str(directive.get('step_summary', '') or assistant_message or step.objective).strip()
                if not assistant_message:
                    assistant_message = step_summary

            accumulated_context.append(f"步骤{step.step_id}: {step_summary or assistant_message}")
            step_results.append({
                'step_id': step.step_id,
                'objective': step.objective,
                'tool_name': requested_tool_name if action_type == 'tool' else '',
                'tool_args': step.tool_args if action_type == 'tool' else {},
                'action_type': action_type,
                'tool_result': tool_result,
                'step_summary': step_summary,
                'assistant_message': assistant_message,
                'reasoning': str(directive.get('reasoning', '') or '').strip(),
            })
            logger.info(f"✅ Executor 完成步骤 {step.step_id}: {_shorten_text(step_summary or assistant_message)}")

        synthesis_messages = [
            SystemMessage(content="你是 ExecutorAgent。请把多步骤执行结果整合成自然、准确、口语化的客服回复，只输出合法 JSON。"),
            HumanMessage(content=json.dumps({
                'user_message': user_message,
                'user_info': user_info,
                'rag_context': rag_context,
                'image_context': image_context,
                'plan': plan,
                'step_results': step_results,
                'speed_mode': speed_mode,
                'output_schema': {
                    'draft_answer': '...',
                    'execution_summary': '...',
                }
            }, ensure_ascii=False)),
        ]
        synthesis_response = self.llm.invoke(synthesis_messages)
        synthesis_text = synthesis_response.content if hasattr(synthesis_response, 'content') else str(synthesis_response)
        synthesis_json = _extract_json_payload(synthesis_text) or {}
        draft_answer = str(synthesis_json.get('draft_answer', '') or '').strip()
        execution_summary = str(synthesis_json.get('execution_summary', '') or '').strip()

        if self._is_meta_or_instruction_text(draft_answer):
            draft_answer = ''

        if not draft_answer:
            draft_answer = '\n'.join([item.get('assistant_message', '') for item in step_results if item.get('assistant_message')]).strip()

        if self._is_meta_or_instruction_text(draft_answer):
            draft_answer = ''

        if not draft_answer:
            draft_answer = "我暂时无法直接执行这个开放域查询，但可以继续帮您做服装与购物相关建议。"

        if self._is_meta_or_instruction_text(execution_summary):
            execution_summary = ''

        return {
            'step_results': step_results,
            'tool_calls': tool_calls,
            'draft_answer': draft_answer,
            'execution_summary': execution_summary or _shorten_text(draft_answer, 180),
            'raw_synthesis': synthesis_text,
        }


class CriticAgent:
    """负责评估执行结果是否满足用户需求。"""

    def __init__(self, llm: ChatOpenAI, knowledge_base: VectorKnowledgeBase):
        self.llm = llm
        self.knowledge_base = knowledge_base

    def evaluate(
        self,
        user_message: str,
        user_info: dict,
        rag_context: str,
        image_context: str,
        history: list | None,
        plan: dict,
        execution_result: dict,
        previous_critique: str = '',
        iteration: int = 1,
        speed_mode: str = 'balanced',
    ) -> dict:
        if not self.llm:
            return {
                'satisfied': False,
                'confidence': 0.0,
                'reason': 'LLM 不可用',
                'missing_points': [],
                'revision_instruction': '',
                'final_answer': execution_result.get('draft_answer', '') if isinstance(execution_result, dict) else '',
                'should_continue': False,
            }

        history_text = []
        for item in (history or [])[-8:]:
            role = '用户' if item.get('role') == 'user' else '助手'
            content = str(item.get('content', '')).strip()
            if content:
                history_text.append(f'{role}: {content}')
        history_block = '\n'.join(history_text) if history_text else '暂无'

        system_prompt = f"""你是 CriticAgent，职责是判断当前执行结果是否真正满足用户需求，并给出下一步评审意见。

你必须结合用户问题、用户画像、知识库、计划和执行结果进行评估。只输出合法 JSON，不要输出 Markdown，不要输出解释性文本。

用户画像：
- 性别：{user_info.get('gender', '未知')}
- 身高：{user_info.get('height', '未知')}cm
- 体重：{user_info.get('weight', '未知')}kg

知识库参考：
{rag_context if rag_context else '暂无相关知识'}

图片上下文：
{image_context if image_context else '暂无图片上下文'}

历史对话：
{history_block}

上一轮评审意见：
{previous_critique if previous_critique else '无'}

输出要求：
- 只返回一个 JSON 对象
- satisfied 表示是否已经满足用户需求
- should_continue 表示是否需要进入下一轮规划与执行
- final_answer 应该给出最终可直接返回给用户的自然语言

JSON 结构：
{{
  "satisfied": true,
  "confidence": 0.0,
  "reason": "...",
  "missing_points": [],
  "revision_instruction": "...",
  "final_answer": "...",
  "should_continue": false
}}"""

        human_prompt = json.dumps({
            'iteration': iteration,
            'user_message': user_message,
            'plan': plan,
            'execution_result': execution_result,
            'speed_mode': speed_mode,
        }, ensure_ascii=False)

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response_text = ''
        for retry in range(2):
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            parsed = _extract_json_payload(response_text)
            if parsed:
                parsed['raw_response'] = response_text
                return parsed
            messages = [
                SystemMessage(content='你是 CriticAgent。请将上一条内容转换为严格合法的 JSON 对象，只输出 JSON。'),
                HumanMessage(content=response_text),
            ]

        return {
            'satisfied': False,
            'confidence': 0.0,
            'reason': '解析失败',
            'missing_points': [],
            'revision_instruction': '请根据已有执行结果继续改写回答。',
            'final_answer': str(execution_result.get('draft_answer', '') if isinstance(execution_result, dict) else ''),
            'should_continue': False,
            'raw_response': response_text,
        }


class MultiAgentSystem(AgentStrategy):
    """多智能体协作系统（LLM统一编排，避免硬编码回复）"""

    def __init__(self, llm: ChatOpenAI, knowledge_base: VectorKnowledgeBase):
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.tool_executor = ToolExecutor()
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.classifier_llm = self.llm.bind(temperature=0, max_tokens=10) if self.llm else None
        self.max_iterations = 3
        self.reasoning_trace: list[dict] = []

        self.planner_agent = PlannerAgent(self.llm, self.knowledge_base)
        self.executor_agent = ExecutorAgent(self.llm, self.knowledge_base, self.tool_executor)
        self.critic_agent = CriticAgent(self.llm, self.knowledge_base)

        self.routing = {
            "size": ["尺码", "身高", "体重", "穿什么码", "尺码推荐"],
            "order": ["订单", "物流", "发货", "快递", "订单号"],
            "return": ["退货", "换货", "退款", "售后"],
            "stock": ["库存", "有货", "缺货", "补货"],
            "recommend": ["推荐", "买什么", "款式", "热销", "图片", "发个图片", "看看"]
        }

        self.max_tool_rounds = 3
        self.last_topic: str = ""
        self.last_question: str = ""
        self.last_user_gender: str = ""
        self.last_recommended_products: list[dict] = []
        self.last_recommend_style: str = ""
        self.confirm_words = ["是的", "对", "嗯", "好", "可以", "行", "没错", "继续", "来一个", "来一版", "发我", "看下", "看看"]

        # 会话级缓存：降低重复分类、重复工具调用和重复知识检索延迟
        self._cache_ttl_seconds = 180
        self._max_cache_entries = 256
        self._message_type_cache: dict[str, tuple[float, str]] = {}
        self._tool_result_cache: dict[str, tuple[float, str]] = {}
        self._rag_context_cache: dict[str, tuple[float, str]] = {}

    def get_reasoning_trace(self) -> List[Dict]:
        """获取多智能体推理轨迹。"""
        return self.reasoning_trace

    def _build_clarification_reply(self, questions: list) -> str:
        """将 Planner 的补充问题转换为可直接返回给用户的话术。"""
        texts: list[str] = []
        for item in questions or []:
            if isinstance(item, dict):
                q = str(item.get('question', '') or '').strip()
            else:
                q = str(item or '').strip()
            if q and q not in texts:
                texts.append(q)

        if not texts:
            return "为了更准确地推荐，方便告诉我您的具体偏好吗？"

        if len(texts) == 1:
            return texts[0]

        return "为了更准确地推荐，我先确认两点：\n" + "\n".join([f"{idx}. {q}" for idx, q in enumerate(texts[:2], 1)])

    def _cleanup_cache(self, cache: dict[str, tuple[float, str]]):
        now = time.time()
        expired_keys = [k for k, (ts, _) in cache.items() if now - ts > self._cache_ttl_seconds]
        for key in expired_keys:
            cache.pop(key, None)

        if len(cache) > self._max_cache_entries:
            overflow = len(cache) - self._max_cache_entries
            oldest_keys = sorted(cache.items(), key=lambda item: item[1][0])[:overflow]
            for key, _ in oldest_keys:
                cache.pop(key, None)

    def _get_cached_value(self, cache: dict[str, tuple[float, str]], key: str) -> Optional[str]:
        self._cleanup_cache(cache)
        entry = cache.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > self._cache_ttl_seconds:
            cache.pop(key, None)
            return None
        return value

    def _set_cached_value(self, cache: dict[str, tuple[float, str]], key: str, value: str):
        cache[key] = (time.time(), value)
        self._cleanup_cache(cache)

    def _build_cache_key(self, payload: dict) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _compact_repeated_text(self, text: str) -> str:
        """压缩重复语句，避免多轮汇总导致文本膨胀。"""
        value = str(text or '').strip()
        if not value:
            return ''

        parts = [p.strip() for p in re.split(r'\n+', value) if p.strip()]
        seen = set()
        compact_parts = []
        for part in parts:
            key = re.sub(r'\s+', '', part)
            if key in seen:
                continue
            seen.add(key)
            compact_parts.append(part)
        return '\n'.join(compact_parts).strip()

    def _humanize_reply(self, user_message: str, reply_text: str) -> str:
        """统一润色最终回复，减少流程化/机械化表达。"""
        if self._is_fit_comparison_query(user_message):
            raw_text = str(reply_text or '').strip()
            # 兜底拦截：如果模型把咨询问题回答成“任务标题”，改写为真人建议。
            if re.match(r'^推荐\s*\d+\s*[-~到]\s*\d+\s*款', raw_text) or raw_text.startswith("推荐"):
                return self._build_fit_comparison_reply(user_message)

        text = self._compact_repeated_text(str(reply_text or '').strip())
        if not text:
            return "我在呢，您可以直接告诉我想解决的问题，我马上帮您处理。"

        internal_markers = [
            "了解用户", "根据用户需求", "根据用户偏好", "询问用户", "执行步骤", "步骤", "继续推理",
            "多智能体", "planner", "executor", "critic", "response_mode", "tool_calls"
        ]
        lines = [ln.strip() for ln in re.split(r'\n+', text) if ln.strip()]
        cleaned_lines = []
        for line in lines:
            compact = re.sub(r'\s+', '', line).lower()
            if any(marker in compact for marker in internal_markers):
                continue
            cleaned_lines.append(line)

        if cleaned_lines:
            text = "\n".join(cleaned_lines).strip()

        # 尽量把“先问后答”改为“先给帮助再补问”，更像真人导购/客服
        if text.startswith("请问") and self._is_recommend_context(user_message, None):
            text = f"可以的，我先给您推荐几款合适的，您看看偏好后我再帮您细化。\n{text}"

        if len(re.sub(r'\s+', '', text)) < 8:
            user_text = str(user_message or '')
            if any(k in user_text for k in ["订单", "物流", "发货", "快递"]):
                return "我来帮您查订单进度。方便的话把订单号发我，或者告诉我下单手机号和姓名，我马上帮您核对。"
            if any(k in user_text for k in ["退货", "换货", "退款", "售后"]):
                return "可以处理售后，我先帮您看最快方案。您把订单号发我，我按您的情况给您具体步骤。"
            if any(k in user_text for k in ["推荐", "买", "衣服", "穿搭", "款式"]):
                return "可以，我先给您推荐几款合适的。您更偏向日常通勤、休闲还是运动风？"
            return "收到，我来帮您处理。您可以再具体说下需求，我会直接给您可执行的建议。"

        return text

    def _build_clarification_reply(self, questions: list) -> str:
        normalized = []
        for q in questions or []:
            if isinstance(q, dict):
                text = str(q.get("question", "") or q.get("content", "") or "").strip()
            else:
                text = str(q).strip()
            if text:
                normalized.append(text)
        if not normalized:
            return "我可以马上给您推荐，不过为了更贴合您的需求，还想确认一个小问题：您更偏好哪类服装和风格？"
        unique_questions = list(dict.fromkeys(normalized))[:2]
        if len(unique_questions) == 1:
            return f"我可以马上给您推荐，不过先确认一下：{unique_questions[0]}"
        return (
            "我可以马上给您推荐，不过先确认两个小问题：\n"
            f"1. {unique_questions[0]}\n"
            f"2. {unique_questions[1]}"
        )

    def _is_no_preference_reply(self, message: str) -> bool:
        text = re.sub(r"\s+", "", str(message or "").lower())
        if not text:
            return False
        phrases = [
            "没有要求", "没要求", "无要求", "都可以", "都行", "随便", "无所谓", "不限", "没啥要求",
            "你推荐就行", "都可", "都ok", "没偏好"
        ]
        return any(p in text for p in phrases)

    def _is_recommend_context(self, user_message: str, history: list | None) -> bool:
        text = str(user_message or "").lower()
        recommend_keywords = ["推荐", "买", "款", "t恤", "短袖", "上衣", "看看", "穿搭"]
        if any(k in text for k in recommend_keywords):
            return True
        if self.last_topic == "recommend":
            return True

        for item in reversed((history or [])[-8:]):
            content = str(item.get("content", "") if isinstance(item, dict) else "").lower()
            if any(k in content for k in recommend_keywords):
                return True
        return False

    def _extract_recommend_style(self, user_message: str, history: list | None) -> str:
        style_aliases = {
            "紧身裤": "紧身裤",
            "打底裤": "打底裤",
            "瑜伽裤": "瑜伽裤",
            "运动裤": "运动裤",
            "阔腿裤": "阔腿裤",
            "直筒裤": "直筒裤",
            "修身裤": "修身裤",
        }
        style_tokens = ["t恤", "短袖", "衬衫", "卫衣", "外套", "牛仔裤", "裤子", "连衣裙", "半身裙"]

        def _pick(text: str) -> str:
            lower_text = str(text or "").lower()
            for alias, normalized in style_aliases.items():
                if alias in lower_text:
                    return normalized
            for token in style_tokens:
                if token in lower_text:
                    return token
            return ""

        direct = _pick(user_message)
        if direct:
            return direct

        for item in reversed((history or [])[-8:]):
            content = str(item.get("content", "") if isinstance(item, dict) else "")
            picked = _pick(content)
            if picked:
                return picked

        return ""

    def _build_direct_recommend_plan(self, plan: dict, user_message: str, user_info: dict, history: list | None) -> dict:
        style = self._extract_recommend_style(user_message, history)
        gender = str(user_info.get("gender", "") or self.last_user_gender or "").strip()

        tool_args: dict[str, Any] = {"limit": 3}
        if style:
            tool_args["style"] = style
        if gender:
            tool_args["gender"] = gender

        forced_step = {
            "step_id": 1,
            "objective": "直接推荐2-3款符合需求的商品",
            "tool_name": "recommend_products",
            "tool_args": tool_args,
            "expected_result": "给出2-3款商品推荐，不再继续追问偏好",
            "rationale": "用户已明确无额外偏好，应直接给出候选商品提升体验",
        }

        plan_copy = dict(plan or {})
        plan_copy["intent_summary"] = str(plan_copy.get("intent_summary", "") or "用户需要直接商品推荐")
        plan_copy["response_mode"] = "tool"
        plan_copy["clarification_questions"] = []
        plan_copy["steps"] = [forced_step]
        return plan_copy

    def _should_skip_recommend_clarification(self, user_message: str, history: list | None) -> bool:
        return self._is_no_preference_reply(user_message) and self._is_recommend_context(user_message, history)

    def _should_prefer_recommend_first(self, user_message: str, history: list | None, plan: dict | None = None) -> bool:
        """判断是否应该先推荐再追问，避免机械化先盘问。"""
        if self._is_fit_comparison_query(user_message):
            return False

        text = str(user_message or "").lower()
        intent_keywords = [
            "买", "购买", "推荐", "衣服", "上衣", "裤子", "裙", "外套", "穿搭", "搭配", "看看", "来几款", "有什么"
        ]
        if any(k in text for k in intent_keywords):
            return True

        if self._is_recommend_context(user_message, history):
            return True

        plan_data = plan if isinstance(plan, dict) else {}
        response_mode = str(plan_data.get("response_mode", "") or "").strip().lower()
        if response_mode in {"tool", "multi_step"}:
            return True

        return False

    def _is_fit_comparison_query(self, user_message: str) -> bool:
        """识别“宽松还是紧身”这类咨询型问题，避免误走商品推荐模板。"""
        text = str(user_message or "").strip().lower()
        if not text:
            return False

        fit_tokens = ["宽松", "紧身", "修身", "版型", "贴身"]
        compare_tokens = ["还是", "哪个好", "更好", "区别", "怎么选", "选哪个", "适合"]
        sport_tokens = ["运动", "跑步", "健身", "训练", "瑜伽", "篮球", "足球"]

        fit_hit = any(token in text for token in fit_tokens)
        compare_hit = any(token in text for token in compare_tokens)
        sport_hit = any(token in text for token in sport_tokens)

        if fit_hit and compare_hit:
            return True
        if fit_hit and sport_hit and "裤" in text:
            return True
        return False

    def _build_fit_comparison_reply(self, user_message: str) -> str:
        """生成更像真人导购的版型建议回复。"""
        text = str(user_message or "")
        if "运动" in text or "跑步" in text or "健身" in text:
            return (
                "这个问题问得很专业。运动场景里通常这样选：\n"
                "1. 跑步/高强度训练：偏修身或微弹贴合更利落，不容易摩擦和兜风。\n"
                "2. 力量训练/日常通勤：偏宽松更舒适，活动空间更大。\n"
                "3. 如果只买一条通用款：建议选“微宽松 + 弹力面料 + 束脚/锥形版型”，兼顾舒适和运动表现。\n"
                "如果你告诉我主要运动类型（跑步、健身还是球类）和你偏好的裤长，我可以直接给你一版更精准的推荐。"
            )

        return (
            "如果你在“宽松”和“紧身”之间犹豫，可以按场景选：高强度运动偏紧一点更跟脚，日常和轻运动偏宽松更舒服。\n"
            "想要不踩雷的话，先选微宽松弹力款最稳。"
        )

    def _is_low_progress_round(self, execution_result: dict, plan: dict) -> bool:
        """判断当前轮次是否缺少有效新增信息。"""
        tool_calls = execution_result.get('tool_calls', []) if isinstance(execution_result, dict) else []
        step_results = execution_result.get('step_results', []) if isinstance(execution_result, dict) else []
        clarifications = plan.get('clarification_questions', []) if isinstance(plan, dict) else []

        if tool_calls:
            return False
        if clarifications:
            return True

        summaries = []
        for item in step_results:
            if not isinstance(item, dict):
                continue
            summary = str(item.get('step_summary', '') or item.get('assistant_message', '')).strip()
            if summary:
                summaries.append(summary)

        if not summaries:
            return True

        uniq = {re.sub(r'\s+', '', s) for s in summaries}
        return len(uniq) <= 1

    def _dynamic_tool_budget(self, message: str, speed_mode: str = "balanced") -> int:
        """根据问题复杂度动态设置工具调用预算，不降低复杂问题能力。"""
        text = (message or "").strip()
        if speed_mode == "fast":
            return 1 if len(text) <= 20 else 2
        if speed_mode == "quality":
            return self.max_tool_rounds
        if any(k in text for k in ["订单", "快递", "物流", "退货", "换货", "退款", "库存", "尺码", "推荐", "图片"]):
            return min(self.max_tool_rounds, 3)
        if len(text) <= 8:
            return 1
        return min(self.max_tool_rounds, 2)

    def _route(self, message: str) -> str:
        for agent, keywords in self.routing.items():
            if any(kw in message for kw in keywords):
                return agent
        return "general"

    def _is_confirm_reply(self, message: str) -> bool:
        msg = (message or "").strip().lower()
        if any(confirm in msg for confirm in self.confirm_words):
            return True
        # 短句承接：用户常用“继续/来吧/发我看看”来延续上一任务
        return bool(re.fullmatch(r"(继续|来吧|来个|来一版|发我|发我看下|发来看看|看看|看一下|好|行|可以)", msg))

    def _should_inherit_topic(self, message: str) -> bool:
        return self._is_confirm_reply(message) and self.last_topic and self.last_topic != "general"

    def _build_system_prompt(self, user_info: dict, rag_context: str, image_context: str) -> str:
        prompt = f"""你是 FashionAI 的资深服装电商客服，请像真人客服一样自然聊天。

【说话风格】
- 口语化、自然、有温度，不要像机器背书
- 优先理解用户意图，再给清晰可执行建议
- 信息不足时，先追问关键字段（一次不超过2个问题）
- 不要输出硬编码模板句，不要机械复读

【服务体验规则】
- 先解决问题，再安抚情绪，避免空泛客套。
- 一轮只推进一个主任务，不要同时抛多个问题。
- 缺信息时只追问最少必要字段，不要一次盘问太多。
- 用户明确“要什么/不要什么”时，优先执行，不要反着来。
- 工具失败时必须给替代路径（如改用订单号、换查询方式），不要只说系统忙。
- 不要要求用户必须点击页面功能按钮；用户在普通对话中表达同样意图时，也要直接执行对应能力。

【用户信息】
性别：{user_info.get('gender', '未知')}
身高：{user_info.get('height', '未知')}cm
体重：{user_info.get('weight', '未知')}kg

【知识库参考】
{rag_context if rag_context else '暂无相关知识'}
"""

        if image_context:
            prompt += f"\n【图片分析结果】\n{image_context}\n"

        prompt += """
【可用工具】
- get_size_recommendation: 根据性别、身高和体重推荐尺码
- query_order: 查询订单状态（返回商品明细和商品图片链接）
- check_stock: 查询商品库存
- recommend_products: 推荐商品（返回图片链接 image_url 和商品链接 shop_url）
- get_coupon: 获取优惠券

【工具使用原则】
1) 需要精确信息时优先调用工具，不要编造事实。
2) 工具返回后必须用自然语言包装回复，不能直接返回原始JSON字段。
3) 商品推荐数量按用户意图动态决定：用户只问一款就给一款，用户要多款再给多款。
4) 说话风格要像真人导购：先说适合什么、为什么适合，再给商品名和价格，不要像说明书。
5) 只有当用户明确说“要图片”“发链接”“看图”“直接给链接”时，才把 image_url / shop_url 展示出来。
6) 如果用户说“发个图片看看”“给我看图”“推荐T恤”等商品导向诉求，应优先调用 recommend_products。
7) 仅在用户明确提供性别/身高/体重时才更新用户画像；若尺码问题缺信息，请先询问补充。
8) 对于“图片分析、穿搭推荐、尺码推荐、订单查询、售后、优惠券”等能力，统一支持自然语言触发，不依赖固定口令。

【意图理解要求（像真人客服）】
1) 先判断“用户现在要什么”，再决定怎么答：
    - 要看图片/链接：直接给对应图片或链接，不要再反问。
    - 要你评价：先给观点和理由，再补建议。
    - 要筛选（如预算、风格、尺码、场景）：先按条件收窄，再推荐。
    - 要换商品：先确认替换方向，再给新的候选。
    - 要结束话题：简短收尾，不要继续推销。
2) 要明确识别“用户不要什么”：
    - 用户说“不要图片/不要链接/别推荐太多”，就不要输出这些内容。
    - 用户说“直接点”，就减少铺垫、给结论和下一步。
3) 回复必须口语化、自然、连贯：
    - 先接住用户当前意图，再给信息。
    - 不机械复述，不自说自话，不忽略用户上一句的明确要求。
4) 若用户表达强意图（例如“我要图片”），优先执行该动作，不要继续停留在泛推荐。
"""
        return prompt

    def _topic_hint(self, topic: str) -> str:
        hints = {
            "size": "当前对话偏向尺码咨询，必要时先补齐性别、身高、体重。",
            "order": "当前对话偏向订单/物流，可优先引导提供订单号并查询。",
            "return": "当前对话偏向售后退换货，先说明规则再给处理步骤。",
            "stock": "当前对话偏向库存查询，先确认商品和尺码。",
            "recommend": "当前对话偏向商品推荐，优先调用推荐工具并给出图片和商品链接。"
        }
        return hints.get(topic, "")

    def _execute_tools(self, tool_calls: list) -> list[dict]:
        if not tool_calls:
            return []

        normalized_calls: list[dict] = []
        for tc in tool_calls:
            tool_name = tc.get('name', '')
            args = tc.get('args', {})
            if tool_name == "recommend_products" and self.last_user_gender and "gender" not in args:
                args = {**args, "gender": self.last_user_gender}
            normalized_calls.append({'name': tool_name, 'args': args})

        results: list[Optional[dict]] = [None] * len(normalized_calls)
        futures = []
        cache_hits = 0

        def _run_one(idx: int, tool_name: str, args: dict) -> tuple[int, dict]:
            try:
                result = self.tool_executor.execute(tool_name, args)
                return idx, {
                    "tool_name": tool_name,
                    "result": result,
                    "success": True,
                    "args": args,
                }
            except Exception as e:
                return idx, {
                    "tool_name": tool_name,
                    "result": str(e),
                    "success": False,
                    "args": args,
                }

        for idx, tc in enumerate(normalized_calls):
            tool_name = tc['name']
            args = tc['args']
            cache_key = self._build_cache_key({'tool_name': tool_name, 'args': args})
            cached_result = self._get_cached_value(self._tool_result_cache, cache_key)
            if cached_result is not None:
                cache_hits += 1
                results[idx] = {
                    "tool_name": tool_name,
                    "result": cached_result,
                    "success": True,
                    "args": args,
                }
                continue
            futures.append((cache_key, idx, tool_name, args))

        if futures:
            max_workers = min(4, len(futures))
            logger.info(f"⚙️ 工具并行执行: 并发 {max_workers} 个任务")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                submit_map = {
                    executor.submit(_run_one, idx, tool_name, args): (cache_key, idx)
                    for cache_key, idx, tool_name, args in futures
                }
                for future in submit_map:
                    cache_key, _ = submit_map[future]
                    idx, output = future.result()
                    results[idx] = output
                    if output.get('success'):
                        self._set_cached_value(self._tool_result_cache, cache_key, str(output.get('result', '')))

        if cache_hits:
            logger.info(f"⚡ 工具缓存命中: {cache_hits}/{len(normalized_calls)}")

        finalized: list[dict] = []
        for item in results:
            if not item:
                finalized.append({"tool_name": "unknown", "result": "工具执行失败", "success": False})
                continue

            if item.get("tool_name") == "recommend_products" and item.get("success"):
                try:
                    payload = json.loads(str(item.get("result", "")))
                    products = payload.get("products", []) if isinstance(payload, dict) else []
                    if isinstance(products, list) and products:
                        self.last_recommended_products = products
                        self.last_recommend_style = str((item.get("args") or {}).get("style", "") or "")
                except Exception:
                    pass

            finalized.append({
                "tool_name": str(item.get("tool_name", "unknown")),
                "result": item.get("result", ""),
                "success": bool(item.get("success", False)),
            })

        return finalized


    def _is_image_or_link_request(self, user_message: str) -> bool:
        """判断用户是否明确要求图片或链接"""
        text = (user_message or "").strip().lower()
        keywords = [
            "图片", "看图", "有图", "发图", "上图", "实物图", "链接", "购买链接", "商品链接",
            "url", "link", "image"
        ]
        return any(k in text for k in keywords)

    def _contains_valid_url(self, text: str) -> bool:
        if not text:
            return False
        if re.search(r"https?://[^\s)\]】}>]+", text, re.IGNORECASE):
            return True
        return bool(re.search(r"/images/[^\s)\]】}>]+", text, re.IGNORECASE))

    def _extract_product_index_request(self, user_message: str) -> Optional[int]:
        text = str(user_message or "").strip()
        if not text:
            return None
        mapping = {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10
        }
        m = re.search(r"第\s*([0-9一二两三四五六七八九十])\s*(款|个|件)", text)
        if m:
            token = m.group(1)
            if token.isdigit():
                idx = int(token)
            else:
                idx = mapping.get(token, 0)
            return idx if idx > 0 else None
        return None

    def _contains_multi_items_intent(self, user_message: str) -> bool:
        text = str(user_message or "")
        keywords = ["几款", "多款", "都发", "都给我", "全部", "三款", "再推荐", "更多"]
        return any(k in text for k in keywords)

    def _build_products_links_reply(
        self,
        products: Optional[list[dict]] = None,
        style: str = "",
        gender: str = "",
        max_items: int = 3,
        lead_text: str = ""
    ) -> Optional[str]:
        """构建包含图片和链接的商品回复；优先使用上下文商品，避免固定返回。"""
        try:
            chosen_products = products or []
            if not chosen_products:
                args = {}
                if style:
                    args["style"] = style
                if gender:
                    args["gender"] = gender
                raw = self.tool_executor.execute("recommend_products", args)
                data = json.loads(raw)
                if not data.get("success"):
                    return None
                chosen_products = data.get("products", [])
                if isinstance(chosen_products, list) and chosen_products:
                    self.last_recommended_products = chosen_products
                    self.last_recommend_style = str(style or "")

            if not chosen_products:
                return None

            items = chosen_products[:max(1, max_items)]
            header = lead_text or ("有的，给您发这几款可直接查看的图片和链接：" if len(items) > 1 else "有的，这款的图片和链接给您：")
            lines = [header, ""]
            for idx, p in enumerate(items, 1):
                name = p.get("name", f"商品{idx}")
                price = p.get("price", "")
                image_url = p.get("image_url", "")
                shop_url = p.get("shop_url", "")
                desc = p.get("desc", "")

                lines.append(f"{idx}. {name}{f'（¥{price}）' if price else ''}")
                if desc:
                    lines.append(f"   亮点：{desc}")
                if image_url:
                    lines.append(f"   图片：{image_url}")
                if shop_url:
                    lines.append(f"   链接：{shop_url}")
                lines.append("")

            lines.append("如果您告诉我想买的场景（通勤/休闲/运动）和版型偏好（修身/宽松），我可以继续帮您细化。")
            return "\n".join(lines).strip()
        except Exception:
            return None

    def _build_contextual_assets_reply(self, user_message: str) -> Optional[str]:
        """按用户当前意图给图片/链接：优先指代最近推荐项，其次再补充多款。"""
        if not self.last_recommended_products:
            return None

        index_request = self._extract_product_index_request(user_message)
        if index_request is not None:
            idx = index_request - 1
            if 0 <= idx < len(self.last_recommended_products):
                product = self.last_recommended_products[idx]
                name = str(product.get("name", "这款商品"))
                lead = f"有的，您说的第{index_request}款（{name}）图片和链接如下："
                return self._build_products_links_reply(products=[product], max_items=1, lead_text=lead)

        text = str(user_message or "")
        for product in self.last_recommended_products:
            name = str(product.get("name", "")).strip()
            if name and name in text:
                lead = f"有的，{name} 的图片和链接如下："
                return self._build_products_links_reply(products=[product], max_items=1, lead_text=lead)

        if self._contains_multi_items_intent(user_message):
            return self._build_products_links_reply(
                products=self.last_recommended_products,
                max_items=min(3, len(self.last_recommended_products)),
                lead_text="可以，这里给您发几款图和购买链接："
            )

        return None

    def _postprocess_response_for_assets(self, user_message: str, response_text: str) -> str:
        """当用户明确要图片/链接时，确保返回可用的真实地址，避免“(链接)”占位。"""
        text = (response_text or "").strip()
        if not self._is_image_or_link_request(user_message):
            return text

        contextual = self._build_contextual_assets_reply(user_message)
        if contextual:
            return contextual

        # 已含真实 URL，直接返回
        if self._contains_valid_url(text):
            return text

        # 出现占位词但无真实 URL，强制使用工具结果兜底
        placeholders = ["(链接)", "链接", "图片"]
        if any(p in text for p in placeholders):
            rebuilt = self._build_products_links_reply(
                products=self.last_recommended_products,
                style=self.last_recommend_style,
                gender=self.last_user_gender,
                max_items=1
            )
            if rebuilt:
                return rebuilt

        # 即便无占位词，只要用户明确要图/链接且结果不含 URL，也兜底
        rebuilt = self._build_products_links_reply(
            products=self.last_recommended_products,
            style=self.last_recommend_style,
            gender=self.last_user_gender,
            max_items=1
        )
        return rebuilt or text

    def _is_fashion_domain_query(self, user_message: str) -> bool:
        """判断是否属于服装电商客服领域问题。"""
        text = str(user_message or "").strip().lower()
        if not text:
            return False

        fashion_keywords = [
            "衣", "服装", "穿搭", "搭配", "款式", "版型", "面料", "材质", "颜色", "尺码", "试穿",
            "上衣", "裤", "裙", "外套", "衬衫", "卫衣", "t恤", "牛仔", "瑜伽裤", "紧身裤",
            "推荐", "买", "购买", "下单", "订单", "物流", "快递", "发货", "签收", "退货", "换货", "退款", "售后",
            "库存", "有货", "缺货", "优惠", "优惠券", "折扣", "满减", "链接", "图片"
        ]
        return any(keyword in text for keyword in fashion_keywords)

    def _build_domain_focus_reply(self, user_message: str) -> str:
        """非服装问题时，返回领域边界提示并引导用户回到服装客服场景。"""
        text = str(user_message or "")
        if any(k in text for k in ["天气", "温度", "热", "冷"]):
            return "我是服装电商客服，不能实时查询天气。你告诉我城市和体感温度，我可以马上给你今天的穿搭建议。"
        return "我是专门的服装电商客服，主要处理尺码建议、商品推荐、订单物流、退换货和优惠券问题。你可以直接告诉我你的服装需求。"

    def _fallback_message_category(self, user_message: str) -> str:
        """LLM 分类失败时的关键词降级分类。"""
        text = str(user_message or "").strip().lower()
        if not text:
            return "CHITCHAT"

        if any(k in text for k in ["你好", "您好", "hello", "hi", "哈喽", "在吗", "嗨"]):
            return "GREETING"
        if any(k in text for k in ["谢谢", "感谢", "辛苦", "thx", "thanks"]):
            return "THANKS"
        if any(k in text for k in ["好的", "行", "可以", "嗯", "ok", "好滴", "收到", "明白"]):
            return "CONFIRM"
        if any(k in text for k in ["今天天气", "天气", "温度", "炎热", "很热", "太热", "变冷", "很冷", "你是谁", "讲个笑话", "聊聊", "无聊", "在干嘛"]):
            return "CHITCHAT"
        if self._is_fashion_domain_query(user_message):
            return "BUSINESS"
        return "CHITCHAT"

    def _classify_message_category(self, user_message: str) -> str:
        """快速分类用户消息，优先使用 LLM，失败时降级关键词匹配。"""
        categories = {"GREETING", "THANKS", "CONFIRM", "CHITCHAT", "BUSINESS"}
        if not self.llm:
            return self._fallback_message_category(user_message)

        cache_key = self._build_cache_key({'message': str(user_message or '').strip().lower()})
        cached_label = self._get_cached_value(self._message_type_cache, cache_key)
        if cached_label in categories:
            logger.info(f"⚡ 快速分类缓存命中: {cached_label}")
            return cached_label

        system_prompt = (
            "你是消息分类器。请根据用户输入只输出一个词，且必须是以下之一："
            "GREETING、THANKS、CONFIRM、CHITCHAT、BUSINESS。"
            "不要输出任何解释、标点或其他文本。"
        )
        try:
            classifier_llm = self.classifier_llm or self.llm.bind(temperature=0, max_tokens=10)
            result = classifier_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=str(user_message or "")),
            ])
            raw = result.content if hasattr(result, 'content') else ""
            if isinstance(raw, list):
                raw = "".join([str(x) for x in raw])
            label = str(raw or "").strip().upper()
            label = re.sub(r"[^A-Z]", "", label)
            if label in categories:
                self._set_cached_value(self._message_type_cache, cache_key, label)
                return label
            return self._fallback_message_category(user_message)
        except Exception as e:
            logger.warning(f"⚠️ 快速分类 LLM 调用失败，降级关键词匹配: {e}")
            return self._fallback_message_category(user_message)

    def _invoke_with_tools(self, messages: list[BaseMessage], user_message: str, speed_mode: str = "balanced") -> str:
        response = self.llm_with_tools.invoke(messages)

        tool_rounds = 0
        max_tool_rounds = self._dynamic_tool_budget(user_message, speed_mode=speed_mode)
        last_tool_name = ""
        last_tool_signature = ""

        while hasattr(response, 'tool_calls') and response.tool_calls and tool_rounds < max_tool_rounds:
            tool_rounds += 1
            last_tool_name = response.tool_calls[0].get('name', '')

            try:
                current_signature = json.dumps(response.tool_calls, ensure_ascii=False, sort_keys=True)
            except Exception:
                current_signature = str(response.tool_calls)

            # 早停：连续两轮完全相同的工具调用，避免无效循环
            if current_signature == last_tool_signature:
                break
            last_tool_signature = current_signature

            messages.append(AIMessage(content=response.content if hasattr(response, 'content') else "", tool_calls=response.tool_calls))

            tool_results = self._execute_tools(response.tool_calls)

            # 本轮工具全部失败：交给 LLM 自然解释并提供下一步建议
            if tool_results and all(not item.get("success", False) for item in tool_results):
                fail_summary = "\n".join([f"- {item.get('tool_name', 'unknown')}: {item.get('result', '')}" for item in tool_results])
                messages.append(HumanMessage(content=f"工具调用失败，请用自然客服语气解释，并给用户下一步建议：\n{fail_summary}"))
                response = self.llm.invoke(messages)
                break

            for i, tc in enumerate(response.tool_calls):
                tool_call_id = tc.get('id', f"tool_call_{i}")
                tool_payload = tool_results[i].get('result', '')
                if not isinstance(tool_payload, str):
                    tool_payload = json.dumps(tool_payload, ensure_ascii=False)
                messages.append(ToolMessage(content=tool_payload, tool_call_id=tool_call_id))

            # 让模型先理解“要什么/不要什么”，再决定是否展示图片与链接
            messages.append(HumanMessage(content=f"请先准确理解用户这句话的核心诉求：{user_message}。先区分用户要什么和不要什么，再基于工具结果回复。若用户明确要图片或链接（如“我要图片”“发链接”），请直接给出对应 image_url/shop_url；若用户明确不要图片或不要链接，则不要输出。回复要像真人客服，口语化、自然、连贯，不要机械复述。"))

            response = self.llm_with_tools.invoke(messages)
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                break

        tool_to_topic = {
            "get_size_recommendation": "size", "query_order": "order",
            "check_stock": "stock",
            "recommend_products": "recommend", "get_coupon": "recommend"
        }
        if last_tool_name:
            self.last_topic = tool_to_topic.get(last_tool_name, "general")

        content = response.content if hasattr(response, 'content') else ""
        if isinstance(content, list):
            content = "".join([str(x) for x in content])
        content = (content or "").strip()

        if content:
            return self._postprocess_response_for_assets(user_message, content)

        fallback = self.llm.invoke(messages + [HumanMessage(content="请直接给出自然、简洁、可执行的客服回复。")])
        fallback_text = fallback.content if hasattr(fallback, 'content') else ""
        if isinstance(fallback_text, list):
            fallback_text = "".join([str(x) for x in fallback_text])
        final_text = (fallback_text or "抱歉，我再帮您确认一下，请稍等。").strip()
        return self._postprocess_response_for_assets(user_message, final_text)

    def run(self, user_message: str, user_info: dict, image_context: str = "", history: list = None,
            precomputed_rag_context: str = "", speed_mode: str = "balanced") -> str:
        self.reasoning_trace = []
        self.last_user_gender = str(user_info.get("gender", "") or "").strip()
        history = history or []
        user_message = str(user_message or "").strip()

        if not self.llm:
            logger.error("❌ 多智能体系统未能获取到可用 LLM")
            return "抱歉，系统暂时不可用，请稍后再试。"

        # 智能快速响应通道：避免简单消息进入多轮协作流程。
        if len(user_message) < 3:
            logger.info("⚡ 快速响应类型: SHORT_MESSAGE")
            return "请告诉我您想咨询的具体问题（如尺码、订单、退换货或商品推荐），我马上帮您处理。"

        quick_type = self._classify_message_category(user_message)
        logger.info(f"⚡ 快速响应类型: {quick_type}")
        quick_replies = {
            "GREETING": "您好！我是FashionAI智能客服，有什么可以帮您的吗？",
            "THANKS": "不客气！如果还有其他问题，随时问我哦~",
        }
        if quick_type in quick_replies:
            return self._humanize_reply(user_message, quick_replies[quick_type])

        if quick_type == "CHITCHAT":
            return self._humanize_reply(user_message, self._build_domain_focus_reply(user_message))

        if quick_type == "BUSINESS" and not self._is_fashion_domain_query(user_message):
            logger.info("🧭 检测到非服装领域问题，返回领域边界引导")
            return self._humanize_reply(user_message, self._build_domain_focus_reply(user_message))

        if self._is_fit_comparison_query(user_message):
            logger.info("🧭 命中版型对比咨询意图，优先返回选购建议")
            return self._humanize_reply(user_message, self._build_fit_comparison_reply(user_message))

        if precomputed_rag_context:
            rag_context = precomputed_rag_context
        else:
            rag_cache_key = self._build_cache_key({'user_message': user_message})
            cached_rag = self._get_cached_value(self._rag_context_cache, rag_cache_key)
            if cached_rag is not None:
                logger.info("⚡ RAG缓存命中")
                rag_context = cached_rag
            else:
                rag_context = self.knowledge_base.get_context(user_message)
                self._set_cached_value(self._rag_context_cache, rag_cache_key, rag_context or "")
        logger.info(
            f"🚀 多智能体开始 | user_gender:{self.last_user_gender or '未知'} | "
            f"history_len:{len(history)} | rag_len:{len(rag_context)} | speed:{speed_mode}"
        )
        if image_context:
            logger.info(f"🖼️ 图片上下文: {_shorten_text(image_context, 180)}")

        previous_critique = ""
        best_answer = ""
        last_round_signature = ""
        low_progress_rounds = 0
        speed_profile = str(speed_mode or "balanced").strip().lower()
        single_round_direct_mode = speed_profile != "quality"
        iteration_budget = 1 if single_round_direct_mode else self.max_iterations

        for iteration in range(1, iteration_budget + 1):
            logger.info(f"🧠 第 {iteration} 轮开始：Planner 生成计划")
            plan = self.planner_agent.create_plan(
                user_message=user_message,
                user_info=user_info,
                rag_context=rag_context,
                image_context=image_context,
                history=history,
                previous_critique=previous_critique,
                speed_mode=speed_mode,
            )
            logger.info(
                f"📋 Plan | intent:{_shorten_text(plan.get('intent_summary', ''), 140)} | "
                f"mode:{plan.get('response_mode', 'unknown')} | steps:{len(plan.get('steps', []))} | "
                f"confidence:{plan.get('confidence', 0)}"
            )
            if plan.get('clarification_questions'):
                logger.info(f"❓ Planner 补充问题: {plan.get('clarification_questions')}")

            # 不再进行“先追问”早停：即使检测到补充问题，也继续让智能体执行计划。
            response_mode = str(plan.get('response_mode', '') or '').strip().lower()
            planned_steps = plan.get('steps', []) if isinstance(plan, dict) else []
            tool_step_count = len([
                s for s in planned_steps
                if isinstance(s, dict) and str(s.get('tool_name', '') or '').strip()
            ])
            clarification_questions = plan.get('clarification_questions', []) if isinstance(plan, dict) else []
            if clarification_questions and tool_step_count == 0 and response_mode in {'clarify', 'direct'}:
                if self._should_skip_recommend_clarification(user_message, history) or self._should_prefer_recommend_first(user_message, history, plan):
                    plan = self._build_direct_recommend_plan(plan, user_message, user_info, history)
                    logger.info("✅ 跳过追问：命中推荐优先策略，先给商品推荐，再按需补问")
                else:
                    direct_question = self._build_clarification_reply(clarification_questions)
                    logger.info("⚡ 检测到补充问题，首轮直接返回追问，跳过执行与评审")
                    return self._humanize_reply(user_message, direct_question)

            planned_steps = plan.get('steps', []) if isinstance(plan, dict) else []
            clarification_questions = plan.get('clarification_questions', []) if isinstance(plan, dict) else []

            execution_result = self.executor_agent.execute_plan(
                user_message=user_message,
                user_info=user_info,
                rag_context=rag_context,
                image_context=image_context,
                history=history,
                plan=plan,
                speed_mode=speed_mode,
            )
            logger.info(
                f"⚙️ Execution | steps:{len(execution_result.get('step_results', []))} | "
                f"tools:{len(execution_result.get('tool_calls', []))} | summary:{_shorten_text(execution_result.get('execution_summary', ''), 180)}"
            )

            if single_round_direct_mode:
                final_answer = self._compact_repeated_text(
                    str(execution_result.get('draft_answer', '') or execution_result.get('execution_summary', '') or '').strip()
                )
                if not final_answer:
                    final_answer = "抱歉，我还需要更多信息才能给出准确回答。"
                logger.info(
                    f"⚡ 单轮直出模式 | speed:{speed_profile} | iteration:{iteration} | final_answer:{_shorten_text(final_answer, 220)}"
                )
                self.reasoning_trace.append({
                    'iteration': iteration,
                    'plan': plan,
                    'execution_result': execution_result,
                    'critic_result': {'skipped': True, 'reason': f'single_round:{speed_profile}'},
                })
                return self._humanize_reply(user_message, final_answer.strip())

            critic_result = self.critic_agent.evaluate(
                user_message=user_message,
                user_info=user_info,
                rag_context=rag_context,
                image_context=image_context,
                history=history,
                plan=plan,
                execution_result=execution_result,
                previous_critique=previous_critique,
                iteration=iteration,
                speed_mode=speed_mode,
            )
            logger.info(
                f"🔍 Critic | satisfied:{critic_result.get('satisfied')} | continue:{critic_result.get('should_continue')} | "
                f"confidence:{critic_result.get('confidence', 0)} | reason:{_shorten_text(critic_result.get('reason', ''), 180)}"
            )

            self.reasoning_trace.append({
                'iteration': iteration,
                'plan': plan,
                'execution_result': execution_result,
                'critic_result': critic_result,
            })

            candidate_answer = str(critic_result.get('final_answer', '') or execution_result.get('draft_answer', '') or '').strip()
            if candidate_answer:
                best_answer = self._compact_repeated_text(candidate_answer)

            execution_summary = self._compact_repeated_text(str(execution_result.get('execution_summary', '') or ''))
            current_signature = self._build_cache_key({
                'tool_count': len(execution_result.get('tool_calls', []) if isinstance(execution_result, dict) else []),
                'summary': execution_summary,
                'revision': str(critic_result.get('revision_instruction', '') or ''),
            })

            if current_signature == last_round_signature and self._is_low_progress_round(execution_result, plan):
                low_progress_rounds += 1
            else:
                low_progress_rounds = 0
            last_round_signature = current_signature

            if low_progress_rounds >= 1 and critic_result.get('should_continue', True):
                logger.info("ℹ️ 检测到连续低增益轮次，返回当前最佳答案，不触发追问早停")
                fallback_answer = best_answer or candidate_answer or execution_result.get('draft_answer', '') or "我先给您推荐3款热门商品，您看喜欢哪种风格我再继续细化。"
                return self._humanize_reply(user_message, fallback_answer)

            if critic_result.get('satisfied') or not critic_result.get('should_continue', True) or iteration >= iteration_budget:
                final_answer = self._compact_repeated_text(
                    best_answer or candidate_answer or execution_result.get('draft_answer', '') or "抱歉，我还需要更多信息才能给出准确回答。"
                )
                logger.info(
                    f"✅ 多智能体完成 | iteration:{iteration} | final_answer:{_shorten_text(final_answer, 220)}"
                )
                return self._humanize_reply(user_message, final_answer.strip())

            previous_critique = str(
                critic_result.get('revision_instruction', '')
                or critic_result.get('reason', '')
                or '请结合上一轮反馈继续优化回复。'
            ).strip()
            logger.info(f"🔁 进入下一轮，评审建议: {_shorten_text(previous_critique, 180)}")

        return self._humanize_reply(user_message, best_answer or "抱歉，我还需要更多信息才能给出准确回答。")

    def _run_with_routing(self, user_message: str, user_info: dict,
                          rag_context: str, image_context: str,
                          history: list = None, forced_topic: str = None,
                          speed_mode: str = "balanced") -> str:
        """兜底路由：只做话题提示，所有回复统一由 LLM 生成，不做硬编码业务返回。"""
        if forced_topic and forced_topic != "general":
            agent_type = forced_topic
        else:
            agent_type = self._route(user_message)

        if agent_type != "general":
            self.last_topic = agent_type

        system_prompt = self._build_system_prompt(user_info, rag_context, image_context)
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

        if history:
            for h in history[-8:]:
                role = h.get("role")
                content = h.get("content", "")
                if not content:
                    continue
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        hint = self._topic_hint(agent_type)
        if hint:
            messages.append(SystemMessage(content=f"话题提示：{hint}"))

        messages.append(HumanMessage(content=user_message))

        try:
            routed_reply = self._invoke_with_tools(messages, user_message, speed_mode=speed_mode)
            return self._humanize_reply(user_message, routed_reply)
        except Exception as e:
            final_messages: list[BaseMessage] = [
                SystemMessage(content="你是 FashionAI 客服，请用自然、礼貌、简洁的语气告知用户系统暂时繁忙，并给出下一步建议。"),
                HumanMessage(content=f"用户问题：{user_message}\n异常信息：{str(e)}")
            ]
            resp = self.llm.invoke(final_messages)
            text = resp.content if hasattr(resp, 'content') else ""
            if isinstance(text, list):
                text = "".join([str(x) for x in text])
            fallback_text = (text or "抱歉，我这边刚刚有点忙，您可以再说一次，我马上帮您处理。").strip()
            return self._humanize_reply(user_message, fallback_text)


class FashionAIAgent:
    """
    主智能体 - 完整版
                style_hint = style_alias_map.get(style_value, {})
                category_hint = str(style_hint.get("category", "") or "").lower()
                fit_tokens = [str(token).lower() for token in style_hint.get("fit", [])]

                scored_products = []
                for product in products:
                    name = str(product.get("name", "")).lower()
                    desc = str(product.get("desc", "")).lower()
                    tags = [str(tag or "").lower() for tag in product.get("style_tags", [])]
                    category = str(product.get("category", "")).lower()

                    score = 0
                    if style_value and style_value in name:
                        score += 6
                    if style_value and style_value in desc:
                        score += 4
                    if style_value and any(style_value in tag for tag in tags):
                        score += 4

                    if category_hint and category_hint in category:
                        score += 5
                    if category_hint and category_hint in name:
                        score += 3

                    if fit_tokens:
                        fit_hit = any(token in name or token in desc or any(token in tag for tag in tags) for token in fit_tokens)
                        if fit_hit:
                            score += 3

                    if style_value in {"裤子", "紧身裤", "打底裤", "瑜伽裤", "运动裤", "阔腿裤", "直筒裤", "修身裤"}:
                        if "裤" in category or "裤" in name:
                            score += 4

                    if score > 0:
                        scored_products.append((score, product))

                if scored_products:
                    scored_products.sort(key=lambda item: (-item[0], str(item[1].get("name", ""))))
                    products = [item[1] for item in scored_products]
    """
    
    # 全局会话存储（实际项目中应使用 Redis 等外部存储）
    _sessions: Dict[str, 'FashionAIAgent'] = {}
    _size_guide_table_cache: Optional[str] = None
    
    def __init__(self, mode: str = "auto", session_id: str = None):
        self.mode = mode
        self.session_id = session_id or f"session_{int(time.time() * 1000)}"
        
        # 使用 LLM 管理器（单例）
        self._llm_manager = LLMManager()
        
        # 加载 BGE 嵌入模型（带缓存）
        self._embeddings = get_embedding_model()
        if self._embeddings is None:
            logger.warning("⚠️ BGE 嵌入模型加载失败，将使用纯 TF-IDF 检索")
        
        # 初始化知识库
        self._knowledge_base = None  # 延迟初始化
        
        # 初始化智能体（延迟初始化）
        self._react_agent = None
        self._multi_agent = None
        
        # 情绪分析器
        self._emotion_analyzer = EmotionAnalyzer()
        
        # 对话历史
        self._conversation_history: list[dict] = []
        
        # 用户信息缓存
        self._user_info: dict = {}

        # 标记是否正在等待用户补充尺码推荐所需信息
        self._pending_size_recommendation = False

        # 最近一次上传图片的上下文，便于后续直接触发穿搭推荐/合成图
        self._last_image_context: dict = {}

        # 是否正在等待用户确认是否生成合成图
        self._pending_outfit_recommendation = False

        # 订单查询对话状态：支持自然对话补全手机号和姓名
        self._pending_order_query = False
        self._order_query_form = {
            "phone": "",
            "name": "",
        }

        # 轻量缓存：用于降低重复检索和重复问答的延迟
        self._rag_cache: dict[str, tuple[float, str]] = {}
        self._response_cache: dict[str, tuple[float, str]] = {}
        self._cache_ttl_seconds = 120
        self._max_cache_entries = 100

        # 运行时可调设置（由前端设置面板下发）
        self._runtime_settings = {
            "speed_mode": "balanced",
            "agent_mode": "auto",
            "reply_style": "friendly",
            "show_reasoning_summary": False,
        }
        
        logger.info(f"🚀 FashionAI 会话 {self.session_id} 初始化完成")
    
    @property
    def llm(self) -> Optional[ChatOpenAI]:
        """获取 LLM 实例"""
        return self._llm_manager.get_llm()
    
    @property
    def knowledge_base(self) -> VectorKnowledgeBase:
        """获取知识库实例（延迟初始化）"""
        if self._knowledge_base is None:
            self._knowledge_base = VectorKnowledgeBase(
                embeddings=self._embeddings,
                knowledge_dir=None
            )
            logger.info("✅ 知识库初始化完成")
        return self._knowledge_base
    
    @property
    def react_agent(self) -> ReActAgent:
        """获取 ReAct 智能体（延迟初始化）"""
        if self._react_agent is None:
            llm = self.llm
            if llm:
                self._react_agent = ReActAgent(llm, self.knowledge_base)
                logger.info("✅ ReAct 智能体初始化完成")
        return self._react_agent
    
    @property
    def multi_agent(self) -> MultiAgentSystem:
        """获取多智能体系统（延迟初始化）"""
        if self._multi_agent is None:
            llm = self.llm
            if llm:
                self._multi_agent = MultiAgentSystem(llm, self.knowledge_base)
                logger.info("✅ 多智能体系统初始化完成")
        return self._multi_agent
    
    def _auto_select_mode(self, message: str) -> str:
        """自动选择智能体模式"""
        text = str(message or "")

        if self._is_order_followup_message(text):
            return "multi"

        # 闲聊/天气类问题不走 ReAct：这是开放域问题，不是需要工具推理的业务任务。
        chit_chat_keywords = [
            "今天天气", "天气", "温度", "炎热", "很热", "太热", "变冷", "很冷",
            "你是谁", "讲个笑话", "聊聊", "无聊", "在干嘛", "最近怎么样", "吃了吗"
        ]
        if any(kw in text for kw in chit_chat_keywords):
            return "multi"

        # 复杂推理优先走 react：适合解释、多约束、多步骤分析
        reasoning_keywords = [
            "为什么", "原因", "分析", "推理", "请解释", "如何", "怎么", "对比", "利弊", "区别", "方案",
            "评估", "权衡", "优先级", "风险", "可行性", "根因", "排查", "诊断", "定位", "优化", "策略", "规划"
        ]
        if any(kw in text for kw in reasoning_keywords):
            return "react"

        # 多意图任务优先 react：同一问题中包含多个业务域时，使用深推理更稳
        domain_groups = {
            "order": ["订单", "物流", "快递"],
            "aftersales": ["退货", "换货", "退款", "售后"],
            "inventory": ["库存", "有货", "缺货"],
            "size": ["尺码", "身高", "体重"],
            "recommend": ["推荐", "搭配", "款式"],
            "promotion": ["优惠", "优惠券", "折扣"],
            "image": ["图片", "图", "链接"],
        }
        hit_domains = sum(
            1 for keywords in domain_groups.values() if any(kw in text for kw in keywords)
        )
        if hit_domains >= 2:
            return "react"

        # 长问题 + 多连接词/多问号，通常属于复杂需求
        complexity_connectors = ["并且", "同时", "另外", "然后", "再", "如果", "但是", "而且", "以及"]
        connector_hits = sum(1 for token in complexity_connectors if token in text)
        question_marks = text.count("?") + text.count("？")
        long_query = len(text) >= 36
        has_step_request = any(kw in text for kw in ["一步步", "分步骤", "详细", "完整地", "全面", "综合"])
        if (long_query and (connector_hits >= 2 or question_marks >= 2)) or has_step_request:
            return "react"

        # 结构化客服任务优先走 multi：更稳、更快
        structured_keywords = [
            "订单", "物流", "快递", "退货", "换货", "退款", "售后",
            "库存", "有货", "缺货", "优惠券", "尺码", "推荐", "图片", "链接"
        ]
        if any(kw in text for kw in structured_keywords):
            return "multi"

        # 最近多轮未解决时，自动切 react 做一次深推理兜底
        recent_assistant_msgs = [
            str(item.get("content", "")) for item in self._conversation_history[-6:]
            if item.get("role") == "assistant"
        ]
        unresolved_hits = 0
        unresolved_signals = ["未找到", "请重试", "暂时", "无法", "失败"]
        for msg in recent_assistant_msgs:
            if any(sig in msg for sig in unresolved_signals):
                unresolved_hits += 1
        if unresolved_hits >= 2:
            return "react"

        return "multi"
    
    def _trim_history(self, max_length: int = None) -> list[dict]:
        """修剪对话历史，保持上下文窗口"""
        config = Config()
        max_length = max_length or config.MAX_HISTORY_LENGTH
        
        if len(self._conversation_history) <= max_length:
            return self._conversation_history
        
        # 保留最近的消息
        self._conversation_history = self._conversation_history[-max_length:]
        return self._conversation_history
    
    def _format_history_for_context(self, max_messages: int = 10) -> str:
        """格式化对话历史为上下文字符串"""
        history = self._conversation_history[-max_messages:]
        formatted = []
        for msg in history:
            role = "用户" if msg.get("role") == "user" else "助手"
            formatted.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(formatted)

    def _cleanup_cache(self, cache: dict[str, tuple[float, str]]):
        """清理过期与超量缓存"""
        now = time.time()
        expired_keys = [k for k, (ts, _) in cache.items() if now - ts > self._cache_ttl_seconds]
        for key in expired_keys:
            cache.pop(key, None)

        if len(cache) > self._max_cache_entries:
            for key, _ in sorted(cache.items(), key=lambda item: item[1][0])[:len(cache) - self._max_cache_entries]:
                cache.pop(key, None)

    def _get_cached_value(self, cache: dict[str, tuple[float, str]], key: str) -> Optional[str]:
        self._cleanup_cache(cache)
        entry = cache.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > self._cache_ttl_seconds:
            cache.pop(key, None)
            return None
        return value

    def _set_cached_value(self, cache: dict[str, tuple[float, str]], key: str, value: str):
        cache[key] = (time.time(), value)
        self._cleanup_cache(cache)

    def _build_response_cache_key(self, user_message: str, image_context: str, history: list[dict]) -> str:
        """构建响应缓存键，包含用户画像与最近历史，避免串会话"""
        payload = {
            "message": user_message,
            "image_context": image_context,
            "user_info": {
                "gender": self._user_info.get("gender", ""),
                "height": self._user_info.get("height", ""),
                "weight": self._user_info.get("weight", ""),
            },
            "history": history,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def set_runtime_settings(self, settings: dict | None):
        """设置运行时参数，未传值时保持默认配置。"""
        if not isinstance(settings, dict):
            return

        speed_mode = str(settings.get("speed_mode", self._runtime_settings["speed_mode"]))
        agent_mode = str(settings.get("agent_mode", self._runtime_settings["agent_mode"]))
        reply_style = str(settings.get("reply_style", self._runtime_settings["reply_style"]))
        show_reasoning = bool(settings.get("show_reasoning_summary", self._runtime_settings["show_reasoning_summary"]))

        if speed_mode not in {"fast", "balanced", "quality"}:
            speed_mode = "balanced"
        if agent_mode not in {"auto", "multi", "react"}:
            agent_mode = "auto"
        if reply_style not in {"concise", "detailed", "friendly"}:
            reply_style = "friendly"

        self._runtime_settings.update({
            "speed_mode": speed_mode,
            "agent_mode": agent_mode,
            "reply_style": reply_style,
            "show_reasoning_summary": show_reasoning,
        })

    def _format_message_by_reply_style(self, user_message: str) -> str:
        style = self._runtime_settings.get("reply_style", "friendly")
        if style == "concise":
            return f"请用简洁风格回答，控制在3句以内：\n{user_message}"
        if style == "detailed":
            return f"请用详细风格回答，包含关键步骤与注意事项：\n{user_message}"
        return f"请用自然客服口语化风格回答：\n{user_message}"

    def _append_reasoning_summary_if_needed(self, response: str, selected_mode: str) -> str:
        if not self._runtime_settings.get("show_reasoning_summary", False):
            return response
        summary = f"\n\n[推理摘要] 模式: {selected_mode} | 速度: {self._runtime_settings.get('speed_mode', 'balanced')}"
        return (response or "") + summary

    def _build_rag_context_two_stage(self, retrieval_query: str, user_message: str) -> str:
        """两段式检索：先扩大召回，再按语义与关键词重排。"""
        results = self.knowledge_base.search(retrieval_query, top_k=6)
        if not results:
            return ""

        return_policy_keywords = ["退货", "换货", "退款", "退换", "退换货", "售后", "七天", "无理由"]
        is_return_policy_query = any(k in str(user_message or "") for k in return_policy_keywords)

        query_terms = {w for w in jieba.cut(user_message or "") if len(w.strip()) > 1}
        ranked = []
        for item in results:
            content = item.get("content", "")
            content_terms = {w for w in jieba.cut(content) if len(w.strip()) > 1}
            overlap = len(query_terms & content_terms)
            base_score = float(item.get("score", 0.0))
            rank_score = base_score + overlap * 0.2

            if is_return_policy_query:
                metadata = item.get("metadata", {}) or {}
                source = str(metadata.get("source", "") or "")
                category = str(metadata.get("category", "") or "")
                if source == "return_policy.txt" or category == "return_policy":
                    rank_score += 1.0

            ranked.append((rank_score, item))

        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = [item for _, item in ranked[:3]]
        return "\n".join([f"- {r.get('content', '')}" for r in selected if r.get('content')])

    def _calc_size(self, height: Optional[int], weight: Optional[int]) -> str:
        """根据身高体重估算常见尺码"""
        try:
            h = int(height) if height is not None else 0
            w = int(weight) if weight is not None else 0
        except (TypeError, ValueError):
            return "M"

        if h <= 0 or w <= 0:
            return "M"

        if h < 160:
            if w < 50:
                return "XS"
            if w < 60:
                return "S"
            return "M"
        elif h < 170:
            if w < 55:
                return "S"
            if w < 65:
                return "M"
            return "L"
        elif h < 180:
            if w < 65:
                return "M"
            if w < 75:
                return "L"
            return "XL"
        else:
            if w < 75:
                return "L"
            if w < 90:
                return "XL"
            return "XXL"

    def _extract_user_info_from_message(self, message: str) -> dict:
        """从用户对话中提取性别、身高和体重信息"""
        info = {}
        if not message:
            return info

        normalized_message = message.strip()

        # 性别识别
        gender_pattern = re.search(r"我(?:是|为|是个|是位|是名)?(男生|男|男士|先生|帅哥|男孩|女生|女|女性|女士|小姐|美女|女孩)", message)
        if gender_pattern:
            gender_token = gender_pattern.group(1)
            if gender_token in ["男生", "男", "先生", "帅哥", "男孩"]:
                info["gender"] = "男"
            else:
                info["gender"] = "女"
        elif re.fullmatch(r"(男生|男|男士|先生|帅哥|男孩)", normalized_message):
            info["gender"] = "男"
        elif re.fullmatch(r"(女生|女|女性|女士|小姐|美女|女孩)", normalized_message):
            info["gender"] = "女"
        else:
            loose_gender = re.search(r"(男生|男|男士|先生|帅哥|男孩|女生|女|女性|女士|小姐|美女|女孩)", normalized_message)
            if loose_gender:
                gender_token = loose_gender.group(1)
                if gender_token in ["男生", "男", "先生", "帅哥", "男孩"]:
                    info["gender"] = "男"
                else:
                    info["gender"] = "女"

        # 身高识别
        height_match = re.search(r"(?:身高[:：]?\s*|我(?:的)?身高(?:是|为)?\s*)(\d{2,3})\s*cm", message, re.IGNORECASE)
        if not height_match:
            height_match = re.search(r"我.*?(\d{2,3})\s*cm", message, re.IGNORECASE)
        if not height_match:
            # 兼容“女生163cm”“166 cm”等紧凑输入
            height_match = re.search(r"(?<!\d)(\d{2,3})\s*cm\b", message, re.IGNORECASE)
        if height_match:
            try:
                info["height"] = int(height_match.group(1))
            except ValueError:
                pass

        # 体重识别
        weight_match = re.search(r"(?:体重[:：]?\s*|我(?:的)?体重(?:是|为)?\s*)(\d{2,3})\s*kg", message, re.IGNORECASE)
        if not weight_match:
            weight_match = re.search(r"我.*?(\d{2,3})\s*kg", message, re.IGNORECASE)
        if not weight_match:
            # 兼容“66kg”“66 公斤”“120斤”等输入
            weight_match = re.search(r"(?<!\d)(\d{2,3})\s*(kg|公斤|斤)\b", message, re.IGNORECASE)
        if weight_match:
            try:
                weight_value = int(weight_match.group(1))
                weight_unit = weight_match.group(2).lower() if len(weight_match.groups()) > 1 else "kg"
                if weight_unit == "斤":
                    # 斤转kg（四舍五入）
                    weight_value = int(round(weight_value / 2))
                if 25 <= weight_value <= 200:
                    info["weight"] = weight_value
            except ValueError:
                pass

        # 兼容“女，166，65”这类紧凑输入，按顺序识别为身高/体重
        if "height" not in info or "weight" not in info:
            # 不依赖 \b，避免中文场景下“女生163cm”无法命中
            numbers = [int(n) for n in re.findall(r"(?<!\d)(\d{2,3})(?!\d)", normalized_message)]
            if len(numbers) >= 2:
                if "height" not in info and 120 <= numbers[0] <= 220:
                    info["height"] = numbers[0]
                if "weight" not in info and 25 <= numbers[1] <= 200:
                    info["weight"] = numbers[1]
            elif len(numbers) == 1:
                only_num = numbers[0]
                # 在尺码追问阶段，像“男，175”这类单数字通常是身高
                if self._pending_size_recommendation and "height" not in info and 120 <= only_num <= 220:
                    info["height"] = only_num
                # 兜底：明确携带体重单位时按体重处理
                elif "weight" not in info and re.search(r"(kg|公斤|斤)", normalized_message, re.IGNORECASE):
                    if 25 <= only_num <= 200:
                        info["weight"] = only_num

        return info

    def _get_loose_size(self, size: str) -> str:
        """计算宽松型号，通常比合身尺码大一号"""
        loose_mapping = {
            "XS": "S",
            "S": "M",
            "M": "L",
            "L": "XL",
            "XL": "XXL",
            "XXL": "XXXL"
        }
        return loose_mapping.get(size, "L")

    def _is_size_recommendation_request(self, message: str) -> bool:
        """判断用户是否在请求尺码推荐"""
        if not message:
            return False
        size_keywords = [
            "尺码推荐", "推荐尺码", "穿什么尺码", "穿什么码", "多大码", "选什么码",
            "尺码", "身高", "体重"
        ]
        return any(kw in message for kw in size_keywords)

    def _is_product_recommendation_request(self, message: str) -> bool:
        """判断用户是否在请求具体商品推荐或图片链接。"""
        if not message:
            return False

        text = str(message)
        product_keywords = [
            "推荐", "图片", "链接", "看图", "发图", "上衣", "t恤", "短袖", "衬衫", "外套",
            "卫衣", "裤子", "牛仔裤", "夹克", "风衣", "开衫", "男装", "女装", "款"
        ]
        return any(kw in text.lower() or kw in text for kw in product_keywords)

    def _should_skip_size_flow_for_product_request(self, message: str) -> bool:
        """商品推荐/图片链接请求应优先于尺码补全流程。"""
        if self._is_size_recommendation_request(message):
            return False
        return self._is_product_recommendation_request(message)

    def _extract_size_section_rows(self, raw_text: str, section_title: str) -> list[str]:
        """提取指定性别尺码段落中的行，保留尺码核心信息"""
        lines = [line.strip() for line in raw_text.splitlines()]
        in_section = False
        rows: list[str] = []

        for line in lines:
            if not line:
                continue

            if line.startswith("【") and line.endswith("】"):
                if section_title in line:
                    in_section = True
                    continue
                if in_section:
                    break

            if not in_section:
                continue

            row_match = re.match(r"^([A-Za-z]+)码[：:](.+)$", line)
            if not row_match:
                continue

            size = row_match.group(1).upper()
            details = row_match.group(2)
            height_match = re.search(r"身高\s*(\d{2,3}\s*-\s*\d{2,3}cm)", details)
            weight_match = re.search(r"体重\s*(\d{2,3}\s*-\s*\d{2,3}kg)", details)

            if height_match and weight_match:
                rows.append(f"- {size}码：身高{height_match.group(1)}，体重{weight_match.group(1)}")
            else:
                rows.append(f"- {size}码：{details}")

        return rows

    def _build_size_guide_table(self) -> str:
        """从 size_guide.txt 构建简洁清晰的男女装尺码表"""
        if FashionAIAgent._size_guide_table_cache:
            return FashionAIAgent._size_guide_table_cache

        size_file = Path(__file__).resolve().parent / "knowledge" / "size_guide.txt"

        if not size_file.exists():
            fallback = "尺码表暂不可用，请稍后重试。"
            FashionAIAgent._size_guide_table_cache = fallback
            return fallback

        try:
            raw_text = size_file.read_text(encoding="utf-8")
            women_rows = self._extract_size_section_rows(raw_text, "女装尺码对照表")
            men_rows = self._extract_size_section_rows(raw_text, "男装尺码对照表")

            if not women_rows and not men_rows:
                fallback = "尺码表暂不可用，请稍后重试。"
                FashionAIAgent._size_guide_table_cache = fallback
                return fallback

            table_lines = ["【尺码表】"]
            if women_rows:
                table_lines.append("女装：")
                table_lines.extend(women_rows)
            if men_rows:
                table_lines.append("")
                table_lines.append("男装：")
                table_lines.extend(men_rows)

            formatted_table = "\n".join(table_lines)
            FashionAIAgent._size_guide_table_cache = formatted_table
            return formatted_table
        except Exception as e:
            logger.error(f"❌ 读取尺码表失败: {e}")
            fallback = "尺码表暂不可用，请稍后重试。"
            FashionAIAgent._size_guide_table_cache = fallback
            return fallback

    def _build_size_chart_block(self) -> str:
        """返回前端可直接渲染的男女装尺码图。"""
        return (
            "请先参考尺码图：\n"
            "女装尺码表：/images/size_chart_women.png\n"
            "男装尺码表：/images/size_chart_men.png"
        )

    def _build_size_chart_followup_response(self, missing: list[str]) -> str:
        """缺少关键资料时，固定输出尺码图与补充提示，避免模型重复混发。"""
        chart_block = self._build_size_chart_block()
        ask = f"麻烦您再补充{'、'.join(missing)}，我就能直接给出更准确的尺码建议。比如：175cm，70kg。"
        tips = "小建议：如果喜欢宽松一点的穿着感，可以按推荐码再大一码参考。"
        return f"{chart_block}\n\n{ask}\n{tips}"

    def _detect_size_product_category(self, user_message: str = "") -> str:
        """从当前消息与最近上下文识别尺码咨询对应的品类。"""
        text = str(user_message or "").lower()

        pants_tokens = ["裤子", "牛仔裤", "运动裤", "休闲裤", "西裤", "短裤", "长裤", "裤"]
        top_tokens = ["t恤", "短袖", "上衣", "衬衫", "卫衣", "外套", "夹克", "毛衣", "针织衫"]
        skirt_tokens = ["裙子", "连衣裙", "半身裙", "短裙", "长裙", "裙"]

        if any(token in text for token in pants_tokens):
            return "pants"
        if any(token in text for token in top_tokens):
            return "top"
        if any(token in text for token in skirt_tokens):
            return "skirt"

        for item in reversed(self._conversation_history[-6:]):
            if item.get("role") != "user":
                continue
            content = str(item.get("content", "") or "").lower()
            if any(token in content for token in pants_tokens):
                return "pants"
            if any(token in content for token in top_tokens):
                return "top"
            if any(token in content for token in skirt_tokens):
                return "skirt"

        return "generic"

    def _build_size_analysis_recommendation_response(self, user_message: str = "") -> str:
        """已知身高体重时，直接输出分析+建议+推荐，避免模型随机发挥。"""
        height = self._user_info.get("height")
        weight = self._user_info.get("weight")
        gender = self._user_info.get("gender")
        category = self._detect_size_product_category(user_message)

        try:
            h = int(height)
            w = int(weight)
        except Exception:
            return "我已经收到您的尺码需求，请再提供身高和体重（如：175cm，70kg），我马上给您建议。"

        fitted_size = self._calc_size(h, w) or "M"
        loose_size = self._get_loose_size(fitted_size)

        if gender in {"男", "女"}:
            result = ToolExecutor.execute(
                "get_size_recommendation",
                {"gender": gender, "height": h, "weight": w}
            )
            try:
                data = json.loads(result)
            except Exception:
                data = {"success": False}

            if data.get("success"):
                fitted_size = data.get("size", fitted_size)
                loose_size = self._get_loose_size(fitted_size)
                advice = str(data.get("advice", "")).strip()
                low_value_advice = bool(re.fullmatch(r"推荐\s*[A-Za-z]+码", advice or "", flags=re.IGNORECASE))
                advice_line = advice if advice and not low_value_advice else "建议优先参考商品详情页的肩宽、胸围与衣长数据再确认。"

                if category == "pants":
                    category_tip = (
                        f"裤装建议先从 **{fitted_size}码** 试起；"
                        f"如果偏宽松或运动场景，可优先看 **{loose_size}码**。"
                    )
                    advice_line = "建议重点核对腰围、臀围和裤长三个数据，再结合面料弹性确认。"
                    risk_tip = "下单前把常穿裤子的腰围/臀围/裤长和平铺数据对比，控制在±2cm更稳妥。"
                    followup = "如果告诉我你更偏好修身、直筒还是宽松版型，我可以再给你主推码和备选码。"
                elif category == "skirt":
                    category_tip = (
                        f"裙装建议优先看 **{fitted_size}码**；"
                        f"如果喜欢更飘逸或活动空间更大，可看 **{loose_size}码**。"
                    )
                    advice_line = "建议重点核对腰围、臀围和裙长，避免腰部过紧或裙长不合适。"
                    risk_tip = "下单前将常穿裙装的腰围与裙长和商品页数据对比，控制在±2cm更稳。"
                    followup = "如果你告诉我想要的裙长（短/中/长）和版型（A字/直筒），我可以再细化推荐。"
                else:
                    category_tip = (
                        f"购买基础款上衣可优先选 **{fitted_size}码**；"
                        f"如果计划内搭或偏宽松风格，可选 **{loose_size}码**。"
                    )
                    advice_line = advice if advice and not low_value_advice else "建议优先参考商品详情页的肩宽、胸围与衣长数据再确认。"
                    risk_tip = "下单前建议把您的常穿单品平铺尺寸与商品的肩宽/胸围/衣长对比，控制在±2cm更稳妥。"
                    followup = "如果您告诉我具体想买的是T恤/衬衫/外套/裤子，我可以再给您该品类的更细推荐和备选码。"

                return (
                    f"根据您的信息（{gender}，{h}cm，{w}kg），推荐您选 **{fitted_size}码**。\n"
                    f"{category_tip}\n"
                    f"{advice_line}\n"
                    f"{risk_tip}\n"
                    f"{followup}"
                )

        return (
            f"根据您的信息（{h}cm，{w}kg），先给您通用推荐：**{fitted_size}码**。\n"
            f"如果喜欢宽松风格，可考虑 **{loose_size}码**。\n"
            "下单前建议优先核对商品页的肩宽、胸围、衣长三项数据，和您常穿款对比会更稳。\n"
            "再补充性别和想买的品类（T恤/衬衫/外套/裤子），我可以给您更精准的推荐码和备选码。"
        )

    def _build_size_recommendation_response(self, include_table: bool = True) -> str:
        """尺码推荐统一回复：先给尺码表，再推荐或追问缺失信息"""
        parts = []
        if include_table:
            parts.append(self._build_size_guide_table())

        missing = []
        gender = self._user_info.get("gender")
        height = self._user_info.get("height")
        weight = self._user_info.get("weight")

        if not gender:
            missing.append("性别")
        if not height:
            missing.append("身高")
        if not weight:
            missing.append("体重")

        if missing:
            self._pending_size_recommendation = True
            ask = f"麻烦您补充一下{'、'.join(missing)}，我好帮您看得更准一些。比如：男，175cm，70kg。"
            parts.append(ask)
            return "\n\n".join(parts)

        result = ToolExecutor.execute(
            "get_size_recommendation",
            {"gender": gender, "height": int(height), "weight": int(weight)}
        )

        try:
            data = json.loads(result)
        except Exception:
            data = {"success": False, "error": "尺码推荐结果解析失败"}

        if data.get("success"):
            self._pending_size_recommendation = False
            advice = data.get("advice", "")
            recommend_text = (
                f"根据您提供的信息（性别：{gender}，身高：{height}cm，体重：{weight}kg），"
                f"推荐您选择 **{data.get('size', 'M')}码**。{advice}"
            )
            parts.append(recommend_text)
        else:
            self._pending_size_recommendation = True
            parts.append(data.get("error", "暂时无法给出尺码推荐，请补充更完整信息。"))

        return "\n\n".join(parts)

    def _is_size_profile_complete(self) -> bool:
        """判断尺码推荐所需资料是否齐全"""
        return bool(self._user_info.get("gender") and self._user_info.get("height") and self._user_info.get("weight"))

    def _is_order_followup_message(self, message: str) -> bool:
        """识别基于已查询订单的物流追问。"""
        text = str(message or "").strip().lower()
        if not text:
            return False

        active_order = getattr(self, '_active_order_context', None)
        last_order = getattr(self, '_last_order_context', None)
        if not isinstance(active_order, dict) and not isinstance(last_order, dict):
            return False

        followup_keywords = [
            "什么时候到", "什么时候能到", "多久到", "几天到", "预计送达", "预计到达", "何时送达",
            "到哪了", "物流", "快递", "派送", "还没到", "没收到", "催一下", "催单", "催发货"
        ]
        return any(keyword in text for keyword in followup_keywords)

    def _build_order_followup_reply(self, message: str) -> str:
        """基于当前订单上下文生成物流追问回复。"""
        text = str(message or "")
        order = getattr(self, '_active_order_context', None)
        if not isinstance(order, dict):
            order = getattr(self, '_last_order_context', None)
        if not isinstance(order, dict):
            return ""

        shipping = order.get('shipping', {}) or {}
        status = str(shipping.get('status', '未知') or '未知')
        company = str(shipping.get('company', '未知') or '未知')
        tracking_number = str(shipping.get('tracking_number', '未知') or '未知')
        tracking_history = shipping.get('tracking_history', []) or []
        latest_track = tracking_history[-1] if tracking_history else {}
        latest_time = str(latest_track.get('time', '') or '').strip()
        latest_status = str(latest_track.get('status', '') or '').strip()

        eta_text = "预计 1-2 天内送达"
        if '已签收' in status:
            eta_text = "您的订单已签收"
        elif '派送中' in status or '配送中' in status:
            eta_text = "预计今天内送达（如遇站点拥堵可能顺延至明天）"
        elif '待发货' in status:
            eta_text = "商家发货后预计 1-3 天送达"
        elif '已发货' in status or '揽收' in status:
            eta_text = "预计 1-2 天内送达"
        elif '待付款' in status:
            eta_text = "当前订单还未支付，支付后商家才会安排发货"

        if any(k in text for k in ["什么时候", "多久", "几天", "预计", "到"]):
            head = f"这笔订单{eta_text}。"
        else:
            head = f"这笔订单当前物流状态：{status}。"

        tail_parts = [
            f"订单号：{order.get('order_id', '未知')}",
            f"快递公司：{company}",
            f"快递单号：{tracking_number}"
        ]
        if latest_time or latest_status:
            tail_parts.append(f"最新轨迹：{latest_time} {latest_status}".strip())

        tail = "；".join(tail_parts)
        return f"{head}\n{tail}。"
    
    def chat(self, user_message: str, user_info: dict, image_context: str = "", 
             stream: bool = False) -> str:
        """
        处理用户消息
        
        Args:
            user_message: 用户消息
            user_info: 用户信息（身高、体重等）
            image_context: 图片分析上下文
            stream: 是否使用流式输出（暂未实现）
        
        Returns:
            助手回复
        """
        start_time = time.time()
        
        # 更新用户信息
        self._user_info.update(user_info)
        self._user_info.update(self._extract_user_info_from_message(user_message))
        
        # 1. 情绪分析
        emotion_result = self._emotion_analyzer.analyze(user_message)
        empathy = self._emotion_analyzer.get_empathy(emotion_result["emotion"], user_message)

        # 订单追问直连：已查到订单后，优先复用上下文，避免进入 ReAct/多智能体绕路。
        if self._is_order_followup_message(user_message):
            direct_order_reply = self._build_order_followup_reply(user_message)
            if direct_order_reply:
                self._conversation_history.append({"role": "user", "content": user_message})
                self._conversation_history.append({"role": "assistant", "content": direct_order_reply})
                self._trim_history()
                duration = time.time() - start_time
                logger.info(f"✅ 对话完成（订单追问直连），模式: order_context，耗时: {duration:.2f}s")
                return direct_order_reply
        
        # 2. 自动选择模式
        selected_mode = self.mode
        runtime_agent_mode = self._runtime_settings.get("agent_mode", "auto")
        if runtime_agent_mode in {"multi", "react"}:
            selected_mode = runtime_agent_mode
        elif selected_mode == "auto":
            selected_mode = self._auto_select_mode(user_message)

        # 2.5 尺码场景：缺信息发尺码图，信息完整直接分析+建议+推荐
        size_request = self._is_size_recommendation_request(user_message)
        skip_size_flow = self._should_skip_size_flow_for_product_request(user_message)
        size_flow = (size_request or self._pending_size_recommendation) and not skip_size_flow
        if size_flow:
            missing = []
            if not self._user_info.get("height"):
                missing.append("身高")
            if not self._user_info.get("weight"):
                missing.append("体重")

            self._pending_size_recommendation = bool(missing)
            if missing:
                response = self._build_size_chart_followup_response(missing)
            else:
                response = self._build_size_analysis_recommendation_response(user_message)

            self._conversation_history.append({"role": "user", "content": user_message})
            self._conversation_history.append({"role": "assistant", "content": response})
            self._trim_history()
            duration = time.time() - start_time
            logger.info(f"✅ 对话完成（尺码场景直出），模式: {selected_mode}，耗时: {duration:.2f}s")
            return response

        if skip_size_flow and self._pending_size_recommendation:
            self._pending_size_recommendation = False
        
        # 3. 执行推理
        response = None
        error_message = None
        
        try:
            llm = self.llm
            if llm is None:
                raise Exception("LLM 服务不可用，请稍后重试")
            
            styled_message = self._format_message_by_reply_style(user_message)

            if selected_mode == "react" and self.react_agent:
                response = self.react_agent.run(styled_message, self._user_info, image_context)
            elif self.multi_agent:
                history = self._conversation_history[-6:]  # 最近3轮对话
                response = self.multi_agent.run(
                    styled_message,
                    self._user_info,
                    image_context,
                    history,
                    speed_mode=self._runtime_settings.get("speed_mode", "balanced")
                )
            else:
                raise Exception("智能体未初始化")
            
            # 记录成功
            self._llm_manager.record_success()
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"❌ 对话处理异常: {error_message}")
            
            # 记录失败
            self._llm_manager.record_failure()
            
            # 尝试降级处理
            response = self._fallback_response(user_message, error_message)
        
        # 4. 添加共情回复：放在主回复后，避免首句先被安抚话术占用
        if response and empathy:
            response = response.rstrip() + "\n" + empathy

        response = self._append_reasoning_summary_if_needed(response, selected_mode)
        
        # 5. 更新对话历史
        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append({"role": "assistant", "content": response or ""})
        
        # 6. 修剪历史
        self._trim_history()
        
        # 7. 记录日志
        duration = time.time() - start_time
        logger.info(f"✅ 对话完成，模式: {selected_mode}，耗时: {duration:.2f}s")
        
        return response or "抱歉，服务暂时不可用。"
    
    def _fallback_response(self, user_message: str, error: str) -> str:
        """
        降级响应策略
        
        当 LLM 不可用时，使用规则引擎提供基本回复
        """
        logger.warning(f"⚠️ 使用降级策略: {error}")
        
        return OfflineAssistant().reply(user_message, self._user_info)
    
    def chat_stream(self, user_message: str, user_info: dict, 
                    image_context: str = "", current_image=None, image_base64: str = ""):
        """增强版流式对话 - 带意图识别和行为预测"""
        start_time = time.time()
        preprocess_start = time.time()
        self._user_info.update(user_info)
        self._user_info.update(self._extract_user_info_from_message(user_message))
        retrieval_query = f"{user_message}\n{self._format_history_for_context(max_messages=4)}"

        selected_mode = self.mode
        runtime_agent_mode = self._runtime_settings.get("agent_mode", "auto")
        if runtime_agent_mode in {"multi", "react"}:
            selected_mode = runtime_agent_mode
        elif selected_mode == "auto":
            selected_mode = self._auto_select_mode(user_message)

        def analyze_emotion_and_intent():
            emotion = self._emotion_analyzer.analyze(user_message)
            empathy_text = self._emotion_analyzer.get_empathy(emotion["emotion"], user_message)
            intent_result = SizeIntentAnalyzer.classify(
                user_message,
                user_history=self._conversation_history
            )
            return emotion, empathy_text, intent_result.get("intent", "general")

        def analyze_image_context() -> str:
            if image_context == "[图片]" and current_image is not None:
                import io
                buffer = io.BytesIO()
                current_image.save(buffer, format='PNG')
                img_bytes = buffer.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode()
                return ImageAnalyzer.analyze(img_b64) or "图片分析失败"
            if image_base64:
                return ImageAnalyzer.analyze(image_base64) or "图片分析失败"
            return image_context

        def build_rag_context() -> str:
            rag_cache_key = hashlib.md5(retrieval_query.encode("utf-8")).hexdigest()
            cached = self._get_cached_value(self._rag_cache, rag_cache_key)
            if cached is not None:
                return cached
            context = self._build_rag_context_two_stage(retrieval_query, user_message)
            self._set_cached_value(self._rag_cache, rag_cache_key, context or "")
            return context

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_emotion_intent = executor.submit(analyze_emotion_and_intent)
            future_image_context = executor.submit(analyze_image_context)
            future_rag_context = executor.submit(build_rag_context)

            emotion_result, empathy, intent = future_emotion_intent.result()
            image_context = future_image_context.result()
            rag_context = future_rag_context.result()

        # 订单追问直连：优先走会话订单上下文回复，避免触发无效推理。
        if self._is_order_followup_message(user_message):
            direct_order_reply = self._build_order_followup_reply(user_message)
            if direct_order_reply:
                for i in range(0, len(direct_order_reply), 1):
                    yield direct_order_reply[i:i + 1]
                self._conversation_history.append({"role": "user", "content": user_message})
                self._conversation_history.append({"role": "assistant", "content": direct_order_reply})
                self._trim_history()
                duration = time.time() - start_time
                logger.info(f"✅ 流式对话完成（订单追问直连），耗时: {duration:.2f}s")
                return

        # 尺码场景：缺信息发尺码图，信息完整直接分析+建议+推荐
        size_request = self._is_size_recommendation_request(user_message)
        skip_size_flow = self._should_skip_size_flow_for_product_request(user_message)
        size_flow = (size_request or self._pending_size_recommendation) and not skip_size_flow
        if size_flow:
            missing = []
            if not self._user_info.get("height"):
                missing.append("身高")
            if not self._user_info.get("weight"):
                missing.append("体重")

            self._pending_size_recommendation = bool(missing)
            if missing:
                direct_response = self._build_size_chart_followup_response(missing)
            else:
                direct_response = self._build_size_analysis_recommendation_response(user_message)

            chunk_size = 1
            for i in range(0, len(direct_response), chunk_size):
                yield direct_response[i:i + chunk_size]
            self._conversation_history.append({"role": "user", "content": user_message})
            self._conversation_history.append({"role": "assistant", "content": direct_response})
            self._trim_history()
            duration = time.time() - start_time
            logger.info(f"✅ 流式对话完成（尺码场景直出），耗时: {duration:.2f}s")
            return

        if skip_size_flow and self._pending_size_recommendation:
            self._pending_size_recommendation = False

        size = self._calc_size(self._user_info.get('height'), self._user_info.get('weight')) or "M"
        loose_size = self._get_loose_size(size)

        full_response = ""
        fallback = ""
        history = self._conversation_history[-6:]

        # 真流式首字优化：先输出共情前缀，降低首字等待体感
        if empathy:
            full_response += empathy + "\n"
            yield empathy + "\n"

        preprocess_elapsed = time.time() - preprocess_start
        inference_start = time.time()

        try:
            if self.multi_agent is None:
                raise Exception("多智能体系统未初始化")

            styled_message = self._format_message_by_reply_style(user_message)
            response_cache_key = self._build_response_cache_key(styled_message, image_context, history)

            def _run_inference() -> tuple[str, bool]:
                cached_response_local = self._get_cached_value(self._response_cache, response_cache_key)
                if cached_response_local:
                    return cached_response_local, True

                if selected_mode == "react" and self.react_agent:
                    model_text = self.react_agent.run(styled_message, self._user_info, image_context)
                else:
                    model_text = self.multi_agent.run(
                        styled_message,
                        self._user_info,
                        image_context,
                        history,
                        precomputed_rag_context=rag_context,
                        speed_mode=self._runtime_settings.get("speed_mode", "balanced")
                    )

                self._set_cached_value(self._response_cache, response_cache_key, model_text)
                return model_text, False

            with ThreadPoolExecutor(max_workers=1) as infer_executor:
                infer_future = infer_executor.submit(_run_inference)
                model_response, from_cache = infer_future.result()

            if from_cache:
                logger.info("⚡ 响应缓存命中")

            if not model_response:
                raise Exception("多智能体未返回有效回复")

            # 已提前流式输出共情前缀时，避免模型回复再次重复同一句
            if empathy and model_response.startswith(empathy):
                model_response = model_response[len(empathy):].lstrip("\n ")

            self._llm_manager.record_success()

            inference_elapsed = time.time() - inference_start
            stream_start = time.time()

            # 逐字流式输出，提升“正在打字”体感
            chunk_size = 1
            model_response = self._append_reasoning_summary_if_needed(model_response, selected_mode)
            full_response += model_response
            for i in range(0, len(model_response), chunk_size):
                yield model_response[i:i + chunk_size]

            stream_elapsed = time.time() - stream_start
            logger.info(
                f"⏱️ 流式阶段耗时 - 预处理: {preprocess_elapsed:.2f}s, 推理: {inference_elapsed:.2f}s, 输出: {stream_elapsed:.2f}s"
            )

        except Exception as e:
            error_message = str(e)
            logger.error(f"❌ 对话处理异常: {error_message}")
            self._llm_manager.record_failure()
            fallback = self._fallback_response(user_message, error_message)
            full_response += fallback
            for i in range(0, len(fallback), 1):
                yield fallback[i:i + 1]

        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append({"role": "assistant", "content": full_response})
        self._trim_history()

        duration = time.time() - start_time
        logger.info(f"✅ 流式对话完成，耗时: {duration:.2f}s")
        return
    
    def clear_history(self):
        """清除当前会话的对话历史"""
        self._conversation_history = []
        self._user_info = {}
        self._pending_size_recommendation = False
        self._last_image_context = {}
        self._pending_outfit_recommendation = False
        self._pending_order_query = False
        self._order_query_form = {"phone": "", "name": ""}
        self._rag_cache = {}
        self._response_cache = {}
        if hasattr(self, '_last_order_context'):
            self._last_order_context = None
        if hasattr(self, '_active_order_context'):
            self._active_order_context = None
        logger.info(f"🗑️ 会话 {self.session_id} 历史已清除")

    def set_last_image_context(self, image_base64: str = "", mime_type: str = "image/jpeg", analysis: str = ""):
        """缓存最近一次上传的图片信息"""
        self._last_image_context = {
            "image_base64": image_base64,
            "mime_type": mime_type,
            "analysis": analysis,
        }

    def get_last_image_context(self) -> dict:
        """获取最近一次上传的图片信息"""
        return self._last_image_context.copy()

    def has_last_image_context(self) -> bool:
        """是否存在可用于后续穿搭推荐的图片上下文"""
        return bool(self._last_image_context.get("image_base64"))

    def clear_last_image_context(self):
        """清除最近图片缓存"""
        self._last_image_context = {}
    
    def set_user_info(self, **kwargs):
        """设置用户信息"""
        self._user_info.update(kwargs)
    
    def get_user_info(self) -> dict:
        """获取用户信息"""
        return self._user_info.copy()
    
    # ========== 会话管理（类方法）==========
    
    @classmethod
    def get_or_create_session(cls, session_id: str, mode: str = "auto") -> 'FashionAIAgent':
        """获取或创建会话"""
        if session_id not in cls._sessions:
            cls._sessions[session_id] = FashionAIAgent(mode=mode, session_id=session_id)
            logger.info(f"🆕 创建新会话: {session_id}")
        return cls._sessions[session_id]
    
    @classmethod
    def delete_session(cls, session_id: str):
        """删除会话"""
        if session_id in cls._sessions:
            del cls._sessions[session_id]
            logger.info(f"🗑️ 删除会话: {session_id}")
    
    @classmethod
    def list_sessions(cls) -> list[str]:
        """列出所有会话"""
        return list(cls._sessions.keys())
    
    @classmethod
    def clear_all_sessions(cls):
        """清除所有会话"""
        cls._sessions.clear()
        logger.info("🗑️ 所有会话已清除")


# ============================================================
# 穿搭推荐服务
# ============================================================

class OutfitRecommender:
    """穿搭推荐服务 - 基于豆包 Vision 和图像生成 API"""

    DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
    VISION_MODEL = "doubao-1-5-vision-pro-32k-250115"
    IMAGE_MODEL = "doubao-seedream-4-5-251128"
    VISION_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    IMAGE_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

    DEFAULT_RESULT = {
        "success": False,
        "analysis": None,
        "analysis_text": None,
        "recommendation": None,
        "image_url": None,
        "error": None
    }

    OUTFIT_RULES = {
        "牛仔裤": "白色T恤",
        "黑裤子": "白衬衫",
        "灰裤子": "浅色上衣",
        "西装裤": "衬衫",
        "运动裤": "运动上衣",
        "牛仔裙": "针织衫",
        "连衣裙": "外套",
        "短裙": "T恤",
        "长裙": "针织衫"
    }

    @staticmethod
    def _get_api_key() -> str:
        return os.getenv("DOUBAO_API_KEY", "")

    @staticmethod
    def _prepare_headers() -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OutfitRecommender._get_api_key()}"
        }

    @staticmethod
    def _normalize_analysis(raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "category": raw.get("category", "未知") or "未知",
            "color": raw.get("color", "未知") or "未知",
            "style": raw.get("style", "未知") or "未知",
            "fashion_style": raw.get("fashion_style", "未知") or "未知",
            "material": raw.get("material", "未知") or "未知",
            "scene": raw.get("scene", "未知") or "未知",
            "suggestion": raw.get("suggestion", "建议保持整体色彩协调，选择质感良好的上衣。"),
            "description": raw.get("description", "图片分析完成，但结果不充分。")
        }

    @staticmethod
    def _parse_text_analysis(text: str) -> Dict[str, Any]:
        analysis = {
            "category": "未知",
            "color": "未知",
            "style": "未知",
            "fashion_style": "未知",
            "material": "未知",
            "scene": "未知",
            "suggestion": "建议保持整体色彩协调，选择质感良好的上衣。",
            "description": text[:200]
        }

        if "裤" in text:
            analysis["category"] = "裤子"
        elif "裙" in text:
            analysis["category"] = "裙子"
        elif "上衣" in text or "衬衫" in text or "T恤" in text:
            analysis["category"] = "上衣"

        for color in ["黑", "白", "红", "蓝", "绿", "黄", "灰", "粉", "紫", "棕"]:
            if color in text:
                analysis["color"] = color + "色"
                break

        if "休闲" in text:
            analysis["fashion_style"] = "休闲"
        elif "正式" in text:
            analysis["fashion_style"] = "正式"
        elif "运动" in text:
            analysis["fashion_style"] = "运动"
        elif "时尚" in text:
            analysis["fashion_style"] = "时尚"

        if "约会" in text or "通勤" in text or "日常" in text or "休闲" in text:
            analysis["scene"] = "日常/通勤/约会"

        if "棉" in text or "麻" in text or "皮" in text or "丝" in text:
            analysis["material"] = "棉麻/皮革/丝绸"

        if "紧身" in text or "修身" in text or "宽松" in text or "直筒" in text:
            analysis["style"] = "紧身/修身/宽松/直筒"

        if "搭配" in text or "建议" in text:
            analysis["suggestion"] = text[:200]

        return analysis

    @staticmethod
    def _select_catalog_products(analysis: Dict[str, Any], recommended_top: str, limit: int = 2) -> list[dict]:
        """从商品库中选择与当前搭配最相关的商品，用于展示真实图片和链接。"""
        try:
            catalog = ToolExecutor.list_products(force_reload=False)
        except Exception:
            catalog = []

        if not catalog:
            return []

        category = str(analysis.get("category", "") or "").strip()
        color = str(analysis.get("color", "") or "").strip()
        style = str(analysis.get("fashion_style") if analysis.get("fashion_style") != "未知" else analysis.get("style", "") or "").strip()

        def contains_any(text: str, keywords: list[str]) -> bool:
            text_lower = text.lower()
            return any(k.lower() in text_lower for k in keywords if k)

        # 互补搭配优先：裤/裙 -> 上衣；上衣 -> 裤/裙
        target_keywords: list[str] = []
        same_group_keywords: list[str] = []
        if "裤" in category or "裙" in category:
            target_keywords = ["上衣", "衬衫", "t恤", "卫衣", "针织", "外套", "夹克"]
            same_group_keywords = ["裤", "裙"]
        elif "上衣" in category or "衬衫" in category or "t恤" in category or "卫衣" in category:
            target_keywords = ["裤", "裙", "牛仔裤", "西装裤", "半裙"]
            same_group_keywords = ["上衣", "衬衫", "t恤", "卫衣", "外套", "夹克"]

        recommended_top_keywords: list[str] = []
        for token in ["上衣", "衬衫", "T恤", "卫衣", "针织", "外套", "夹克", "背心"]:
            if token.lower() in str(recommended_top or "").lower():
                recommended_top_keywords.append(token)

        scored: list[tuple[int, dict]] = []
        for product in catalog:
            name = str(product.get("name", "") or "")
            desc = str(product.get("desc", "") or "")
            product_category = str(product.get("category", "") or "")
            tags = " ".join([str(tag or "") for tag in product.get("style_tags", [])])
            blob = f"{name} {desc} {tags} {product_category}"
            score = 0

            if recommended_top and recommended_top in name:
                score += 8
            if recommended_top and recommended_top in desc:
                score += 5
            if recommended_top_keywords and contains_any(blob, recommended_top_keywords):
                score += 6

            # 强化互补类目，抑制同类目误推
            if target_keywords and contains_any(blob, target_keywords):
                score += 10
            if same_group_keywords and contains_any(blob, same_group_keywords):
                score -= 6

            if category and category in product_category:
                score -= 2
            if color and color in blob:
                score += 2
            if style and style in blob:
                score += 1

            # 有图有链接的商品优先，避免给出不可落地推荐
            if str(product.get("image_url", "") or "").strip():
                score += 1
            if str(product.get("shop_url", "") or "").strip():
                score += 1

            if score > 0:
                scored.append((score, product))

        if not scored:
            return []

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[dict] = []
        seen_names: set[str] = set()
        for _, product in scored:
            name = str(product.get("name", "") or "")
            if name in seen_names:
                continue
            selected.append(product)
            seen_names.add(name)
            if len(selected) >= limit:
                break

        return selected

    @staticmethod
    def _analysis_prompt() -> str:
        return """请分析这张服装图片，提取以下信息：
1. 类型：上衣/裤子/裙子/鞋子/配饰等
2. 颜色：主要颜色
3. 款式/版型：紧身/宽松/修身/直筒等
4. 风格：休闲/正式/运动/时尚等
5. 场合：适合出席的场景
6. 材质：棉/麻/皮革/丝绸等（如果能判断）
7. 搭配建议：适合的上衣/外套搭配风格

请只返回纯 JSON，不要额外解释。如果无法判断，请用"未知"填充对应字段。
返回示例：
{
  "category": "裤子",
  "color": "蓝色",
  "style": "修身",
  "fashion_style": "休闲",
  "material": "牛仔",
  "scene": "日常/通勤/约会",
  "suggestion": "建议搭配浅色T恤和白色运动鞋",
  "description": "蓝色牛仔裤，修身款式，适合日常休闲搭配。"
}"""

    @staticmethod
    def analyze_clothing(image_base64: str, mime_type: str = "image/jpeg", retry_count: int = 3) -> Optional[Dict[str, Any]]:
        if not OutfitRecommender._get_api_key():
            logger.error("❌ DOUBAO_API_KEY 未配置")
            return None

        headers = OutfitRecommender._prepare_headers()
        data = {
            "model": OutfitRecommender.VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                    {"type": "text", "text": OutfitRecommender._analysis_prompt()}
                ]
            }],
            "max_tokens": 500,
            "temperature": 0.3
        }

        for attempt in range(retry_count):
            try:
                response = requests.post(OutfitRecommender.VISION_API_URL, headers=headers, json=data, timeout=30)
                result = response.json()

                if response.status_code != 200:
                    logger.error(f"❌ Vision API HTTP {response.status_code}: {result}")
                    continue

                choices = result.get("choices")
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    try:
                        raw_analysis = json.loads(content)
                        logger.info("✅ 服装分析成功(JSON)")
                    except json.JSONDecodeError:
                        logger.warning("⚠️ 分析结果不是标准JSON，尝试文本解析")
                        raw_analysis = OutfitRecommender._parse_text_analysis(content)
                    return OutfitRecommender._normalize_analysis(raw_analysis)

            except Exception as e:
                logger.error(f"❌ 服装分析异常 (尝试 {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2)

        return None

    @staticmethod
    def generate_outfit_image(clothing_analysis: Dict[str, Any], retry_count: int = 3) -> Optional[str]:
        """生成搭配图片 - 修复版"""
        if not OutfitRecommender._get_api_key():
            logger.error("❌ DOUBAO_API_KEY 未配置")
            return None

        headers = OutfitRecommender._prepare_headers()

        category = clothing_analysis.get("category", "服装")
        color = clothing_analysis.get("color", "白色")
        style = clothing_analysis.get("fashion_style") if clothing_analysis.get("fashion_style") != "未知" else "休闲"
        recommended_top = OutfitRecommender.OUTFIT_RULES.get(category, "白色T恤")

        prompt = f"""生成一张电商风格的服装搭配展示图：
- 主体：一件{recommended_top}，与{color}{category}搭配
- 整体风格：{style}风格
- 展示方式：全身照，正面视角
- 背景：干净简约的纯色背景
- 画质：高清，适合商品展示
- 构图：居中展示"""

        data = {
            "model": OutfitRecommender.IMAGE_MODEL,
            "prompt": prompt,
            "size": "1920x1920",
            "n": 1,
            "response_format": "url"
        }

        logger.info(f"🔄 调用豆包图片生成API...")
        logger.info(f"请求数据: {json.dumps(data, ensure_ascii=False)[:200]}")

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    OutfitRecommender.IMAGE_API_URL,
                    headers=headers,
                    json=data,
                    timeout=60
                )

                logger.info(f"图片生成API响应状态码: {response.status_code}")

                if response.status_code != 200:
                    error_text = response.text[:500]
                    logger.error(f"❌ API错误: {error_text}")
                    if attempt < retry_count - 1:
                        time.sleep(2)
                    continue

                result = response.json()
                logger.info(f"API响应: {json.dumps(result, ensure_ascii=False)[:300]}")

                image_url = None

                if result.get("data") and len(result["data"]) > 0:
                    item = result["data"][0]
                    image_url = item.get("url") or item.get("b64_json")
                    if image_url and not image_url.startswith("data:"):
                        image_url = item.get("url")

                if not image_url and result.get("url"):
                    image_url = result["url"]

                if not image_url and result.get("output") and result["output"].get("url"):
                    image_url = result["output"]["url"]

                if image_url:
                    logger.info(f"✅ 成功生成图片: {image_url[:100]}...")
                    return image_url
                else:
                    logger.warning(f"⚠️ 响应中没有找到图片URL: {result}")

            except requests.exceptions.Timeout:
                logger.error(f"❌ 图片生成超时 (尝试 {attempt + 1}/{retry_count})")
            except Exception as e:
                logger.error(f"❌ 图像生成异常 (尝试 {attempt + 1}/{retry_count}): {e}")
                import traceback
                traceback.print_exc()

            if attempt < retry_count - 1:
                time.sleep(2)

        return None

    @staticmethod
    def build_recommendation_text(analysis: Dict[str, Any], user_info: Dict[str, Any]) -> str:
        category = analysis.get("category", "服装")
        color = analysis.get("color", "未知")
        style = analysis.get("fashion_style") if analysis.get("fashion_style") != "未知" else analysis.get("style", "未知")
        scene = analysis.get("scene", "未知")
        material = analysis.get("material", "未知")
        suggestion = analysis.get("suggestion", "建议保持整体色彩协调，选择质感良好的上衣。")
        top = OutfitRecommender.OUTFIT_RULES.get(category, "白色T恤")
        has_image = bool(analysis.get("image_url"))
        related_products = analysis.get("related_products", []) if isinstance(analysis.get("related_products", []), list) else []

        image_line = "📸 **搭配效果图**：图片已生成，您可以直接查看下方图片" if has_image else "📸 **搭配效果图**：我可以继续帮您生成一版更直观的搭配效果图"

        product_lines = []
        if related_products:
            product_lines.append("🛍️ **对应商品链接**：")
            for idx, product in enumerate(related_products[:2], 1):
                name = product.get("name", f"商品{idx}")
                price = product.get("price", "")
                image_url = product.get("image_url", "")
                shop_url = product.get("shop_url", "")
                role = "推荐上衣" if idx == 1 else "参考单品"
                product_lines.append(f"- {role}：{name}{f'（¥{price}）' if price else ''}")
                if image_url:
                    product_lines.append(f"  图片：{image_url}")
                if shop_url:
                    product_lines.append(f"  链接：{shop_url}")

        product_block = "\n".join(product_lines)

        return f"""根据您上传的{color}{category}，我为您推荐了以下搭配：

🎨 **分析结果**：
- 类型：{category}
- 颜色：{color}
- 风格：{style}
- 场合：{scene}
- 材质：{material}

👕 **推荐上衣**：{top}

💡 **搭配建议**：
- {suggestion}
- 建议选择合身款式，以提升整体穿搭效果。
- 这个搭配适合{scene}场合，展现您的个性魅力。

{image_line}

{product_block if product_block else ''}

如果您喜欢这个搭配，可以直接下单哦~"""

    @staticmethod
    def recommend(image_base64: str, user_info: Dict[str, Any], mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """穿搭推荐主方法"""
        result = dict(OutfitRecommender.DEFAULT_RESULT)

        if not image_base64:
            result["error"] = "缺少图片数据"
            return result

        try:
            logger.info("🔍 开始分析服装图片...")
            analysis = OutfitRecommender.analyze_clothing(image_base64, mime_type=mime_type)
            if not analysis:
                result["error"] = "服装分析失败，请检查图片或稍后重试"
                return result

            result["analysis"] = analysis
            result["analysis_text"] = analysis.get("description", "已完成多角度分析。")
            logger.info(f"✅ 服装分析完成: {analysis}")

            logger.info("🎨 开始生成搭配图片...")
            image_url = OutfitRecommender.generate_outfit_image(analysis)
            related_products = OutfitRecommender._select_catalog_products(analysis, OutfitRecommender.OUTFIT_RULES.get(analysis.get("category", "服装"), "白色T恤"), limit=2)

            if image_url:
                result["image_url"] = image_url
                analysis["image_url"] = image_url
                logger.info("✅ 搭配图片生成成功")
            else:
                logger.warning("⚠️ 搭配图片生成失败，但文字推荐仍然可用")
                result["image_url"] = None

            analysis["related_products"] = related_products
            result["recommendation"] = OutfitRecommender.build_recommendation_text(analysis, user_info)

            if not image_url:
                result["recommendation"] += "\n\n⚠️ 提示：图片生成功能暂时不可用，以上为文字搭配建议。"

            result["success"] = True
            return result

        except Exception as e:
            logger.error(f"❌ 穿搭推荐异常: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
            return result

# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 创建智能体
    agent = FashionAIAgent(mode="auto")
    
    # 用户信息
    user_info = {"height": 175, "weight": 70, "level": "普通"}
    
    # 对话示例
    print("=== FashionAI 客服系统 ===\n")
    
    questions = [
        "我身高175cm体重70kg，应该穿什么尺码？",
        "推荐几款衣服",
        "查一下订单ORD123456"
    ]
    
    for q in questions:
        print(f"用户: {q}")
        response = agent.chat(q, user_info)
        print(f"助手: {response}\n")



