

import os
import logging
import psycopg2
from pandas import read_sql
from app.config import settings
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

# ✅ Setup Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VannaService(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None) -> None:
        logger.debug("📦 Inisialisasi VannaService...")
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        self.pg_conn = None
        logger.debug("✅ VannaService initialized with config: %s", config)

    def connect_to_pg(self, conn_str: str):
        """Koneksi ke PostgreSQL via psycopg2."""
        try:
            self.pg_conn = psycopg2.connect(conn_str)
            logger.info("✅ PostgreSQL connected!")
        except Exception as e:
            logger.error("❌ PostgreSQL connection failed: %s", e)
            raise


def get_vanna_instance() -> VannaService:
    """Get configured Vanna instance."""
    cdir = os.getcwd()
    chroma_path = os.path.join(cdir, settings.chroma_folder)

    logger.info(">> Current directory: %s", cdir)
    logger.info(">> Chroma path: %s", chroma_path)
    logger.info(">> Model: %s", settings.model_name)
    logger.info(">> PostgreSQL conn: %s", settings.postgres_conn)

    if not settings.postgres_conn:
        raise Exception("❌ POSTGRES_CONN tidak ditemukan di settings (.env)!")

    vn = VannaService(
        config={
            "model": settings.model_name,
            "path": chroma_path,
        }
    )

    # 🔌 Hubungkan ke PostgreSQL
    vn.connect_to_pg(settings.postgres_conn)

    # ✅ Inject run_sql() supaya bisa digunakan oleh Vanna
    def run_sql(sql: str):
        logger.info("📥 SQL received: %s", sql)
        try:
            df = read_sql(sql, con=vn.pg_conn)
            logger.debug("📊 SQL result: %s", df.head())
            return df
        except Exception as e:
            logger.error("❌ Error saat menjalankan SQL: %s", e)
            raise

    vn.run_sql = run_sql

    logger.info("✅ Vanna instance siap digunakan.")
    return vn


def get_pg_conn():
    """Dapatkan koneksi psycopg2 langsung."""
    if not settings.postgres_conn:
        raise Exception("❌ POSTGRES_CONN tidak ditemukan di settings (.env)!")
    return psycopg2.connect(settings.postgres_conn)


# ✅ Inisialisasi Vanna
logger.info("🚀 Inisialisasi global Vanna...")
vanna = get_vanna_instance()
