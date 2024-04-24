from pickle import dump
from pathlib import Path
from openpyxl import load_workbook


ROOT = Path(__file__).parent.parent
CACHE = ROOT / 'cache'
if not CACHE.exists():
    CACHE.mkdir()

def save_cache(data, name):
    """保存数据到缓存文件"""
    with open(CACHE / name, 'wb') as f:
        dump(data, f)

def load_excel(excel_path: Path):
    """读取Excel文件"""
    wb = load_workbook(excel_path)
    ws = wb.active
    data = []
    for row in ws.iter_rows(values_only=True):
        data.append(row)
    return data
