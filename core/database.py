
"""
قاعدة بيانات SQLite لتاريخ المعالجة
"""
import os
import json
import sqlite3
import datetime
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

from core.paths import data_dir
DEFAULT_DB_PATH = str(data_dir() / "history.db")


class Database:
    """إدارة قاعدة بيانات تاريخ المعالجة"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """إنشاء الجداول إذا لم تكن موجودة"""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    model TEXT,
                    device TEXT,
                    compute_type TEXT,
                    language TEXT,
                    task TEXT,
                    beam_size INTEGER DEFAULT 5,
                    temperature REAL DEFAULT 0.0,
                    vad_filter INTEGER DEFAULT 0,
                    total_files INTEGER DEFAULT 0,
                    total_duration REAL DEFAULT 0,
                    total_elapsed REAL DEFAULT 0,
                    total_segments INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'completed',
                    config_json TEXT
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT,
                    file_size_mb REAL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    model TEXT,
                    language TEXT,
                    language_probability REAL,
                    duration REAL DEFAULT 0,
                    segments INTEGER DEFAULT 0,
                    elapsed REAL DEFAULT 0,
                    speed_factor REAL DEFAULT 0,
                    output_dir TEXT,
                    output_formats TEXT,
                    status TEXT DEFAULT 'completed',
                    error_message TEXT,
                    text_preview TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
                CREATE INDEX IF NOT EXISTS idx_files_name ON files(file_name);
                CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
            """)
            conn.commit()
        finally:
            conn.close()

    # ==========================================
    # إدارة الجلسات
    # ==========================================

    def create_session(self, config: Dict = None, *, status: str = "completed") -> int:
        """إنشاء جلسة معالجة جديدة"""
        conn = self._get_conn()
        try:
            config = config or {}
            cursor = conn.execute("""
                INSERT INTO sessions (model, device, compute_type, language, task, 
                                      beam_size, temperature, vad_filter, config_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.get("selected_model"),
                config.get("device"),
                config.get("compute_type"),
                config.get("language"),
                config.get("task"),
                config.get("beam_size", 5),
                config.get("temperature", 0.0),
                int(config.get("vad_filter", False)),
                json.dumps(config, ensure_ascii=False) if config else None,
                status,
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_session(self, session_id: int, **kwargs):
        """تحديث بيانات الجلسة"""
        if not kwargs:
            return
        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [session_id]

        conn = self._get_conn()
        try:
            conn.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", values)
            conn.commit()
        finally:
            conn.close()

    def get_session(self, session_id: int) -> Optional[Dict]:
        """الحصول على بيانات جلسة"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_sessions(self, limit: int = 50, offset: int = 0, 
                     status: str = None, model: str = None) -> List[Dict]:
        """الحصول على قائمة الجلسات"""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM sessions WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status)
            if model:
                query += " AND model = ?"
                params.append(model)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_session(self, session_id: int):
        """حذف جلسة وكل ملفاتها"""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM files WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    # ==========================================
    # إدارة الملفات
    # ==========================================

    def add_file(self, session_id: int, file_data: Dict) -> int:
        """إضافة ملف معالج"""
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                INSERT INTO files (session_id, file_name, file_path, file_size_mb,
                                   model, language, language_probability, duration,
                                   segments, elapsed, speed_factor, output_dir,
                                   output_formats, status, error_message, text_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                file_data.get("file_name", ""),
                file_data.get("file_path", ""),
                file_data.get("file_size_mb", 0),
                file_data.get("model", ""),
                file_data.get("language", ""),
                file_data.get("language_probability", 0),
                file_data.get("duration", 0),
                file_data.get("segments", 0),
                file_data.get("elapsed", 0),
                file_data.get("speed_factor", 0),
                file_data.get("output_dir", ""),
                file_data.get("output_formats", ""),
                file_data.get("status", "completed"),
                file_data.get("error_message"),
                file_data.get("text_preview", "")[:500] if file_data.get("text_preview") else "",
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_files(self, session_id: int) -> List[Dict]:
        """الحصول على ملفات جلسة"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM files WHERE session_id = ? ORDER BY id", 
                (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_file_history(self) -> List[Dict]:
        """Return one row per file, including metadata stored on older sessions."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT f.file_name, f.elapsed, f.created_at,
                       COALESCE(NULLIF(f.model, ''), s.model) AS model, s.config_json
                FROM files f JOIN sessions s ON s.id = f.session_id
                ORDER BY f.created_at DESC, f.id DESC
            """).fetchall()
        finally:
            conn.close()
        history = []
        for row in rows:
            record = dict(row)
            try:
                config = json.loads(record.pop("config_json") or "{}")
            except (ValueError, TypeError):
                config = {}
            engine = config.get("engine") if isinstance(config, dict) else None
            record["engine"] = engine if isinstance(engine, str) else None
            history.append(record)
        return history

    def get_file(self, file_id: int) -> Optional[Dict]:
        """الحصول على ملف واحد"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def search_files(self, query: str, limit: int = 20) -> List[Dict]:
        """البحث في الملفات"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT f.*, s.model as session_model FROM files f "
                "LEFT JOIN sessions s ON f.session_id = s.id "
                "WHERE f.file_name LIKE ? OR f.text_preview LIKE ? "
                "ORDER BY f.created_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ==========================================
    # إحصائيات
    # ==========================================

    def get_stats(self) -> Dict:
        """الحصول على إحصائيات عامة"""
        conn = self._get_conn()
        try:
            stats = {}
            stats["total_sessions"] = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            stats["total_files"] = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

            dur_row = conn.execute("SELECT COALESCE(SUM(duration), 0) FROM files").fetchone()
            stats["total_duration"] = dur_row[0] if dur_row else 0

            elapsed_row = conn.execute("SELECT COALESCE(SUM(elapsed), 0) FROM files").fetchone()
            stats["total_elapsed"] = elapsed_row[0] if elapsed_row else 0

            seg_row = conn.execute("SELECT COALESCE(SUM(segments), 0) FROM files").fetchone()
            stats["total_segments"] = seg_row[0] if seg_row else 0

            # أكثر النماذج استخداماً
            model_rows = conn.execute(
                "SELECT model, COUNT(*) as cnt FROM sessions WHERE model IS NOT NULL "
                "GROUP BY model ORDER BY cnt DESC LIMIT 5"
            ).fetchall()
            stats["top_models"] = [dict(r) for r in model_rows]

            # أكثر اللغات
            lang_rows = conn.execute(
                "SELECT language, COUNT(*) as cnt FROM files WHERE language IS NOT NULL AND language != '' "
                "GROUP BY language ORDER BY cnt DESC LIMIT 5"
            ).fetchall()
            stats["top_languages"] = [dict(r) for r in lang_rows]

            return stats
        finally:
            conn.close()

    def cleanup_old_sessions(self, days: int = 90):
        """حذف الجلسات الأقدم من عدد أيام"""
        conn = self._get_conn()
        try:
            cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
            conn.execute("DELETE FROM files WHERE session_id IN "
                        "(SELECT id FROM sessions WHERE created_at < ?)", (cutoff,))
            conn.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()
