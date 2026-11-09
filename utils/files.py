
"""
أدوات إدارة الملفات - اختيار، معلومات، تحميل
"""
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


def get_file_info(file_path: str) -> Optional[Dict]:
    """الحصول على معلومات ملف"""
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        return {
            "name": os.path.basename(file_path),
            "size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "ext": Path(file_path).suffix.lower(),
            "path": file_path,
        }
    except OSError as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return None


def select_files_dialog() -> List[str]:
    """فتح نافذة اختيار الملفات"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.wm_attributes('-topmost', 1)
        root.withdraw()
        files = filedialog.askopenfilenames(
            filetypes=[("Media Files", "*.mp4 *.mkv *.avi *.mov *.wav *.mp3 *.flac *.m4a *.aac *.ogg *.wma *.opus *.webm")]
        )
        root.destroy()
        return list(files)
    except Exception as e:
        logger.error(f"Error opening file dialog: {e}")
        return []


def select_directory_dialog() -> Optional[str]:
    """فتح نافذة اختيار المجلد"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.wm_attributes('-topmost', 1)
        root.withdraw()
        directory = filedialog.askdirectory()
        root.destroy()
        return directory if directory else None
    except Exception as e:
        logger.error(f"Error opening directory dialog: {e}")
        return None
