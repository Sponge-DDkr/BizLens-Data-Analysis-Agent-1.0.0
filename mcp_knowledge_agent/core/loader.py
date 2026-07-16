"""文档加载器 — PDF/TXT/Markdown 文本提取 + 切片

从 AI Research Copilot 的 backend/api/knowledge.py 抽取，独立封装。
不依赖 FastAPI/UploadFile，直接接受文件路径。
"""

from pathlib import Path

# 支持的文件类型
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def extract_text(file_path: str) -> str:
    """根据文件类型提取文本

    Args:
        file_path: 文件路径（PDF/TXT/MD/Markdown）

    Returns:
        提取的纯文本内容

    Raises:
        ValueError: 文件类型不支持或文件不存在
        RuntimeError: 解析失败
    """
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"文件不存在: {file_path}")

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件类型「{ext}」。支持：{', '.join(ALLOWED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"不支持的文件类型: {ext}")


def _extract_pdf(path: Path) -> str:
    """从 PDF 文件提取文本"""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n\n".join(texts)
    except ImportError:
        raise RuntimeError("PDF 解析需要 PyPDF2 库。请运行：pip install PyPDF2")
    except Exception as e:
        raise RuntimeError(f"PDF 解析失败：{e}")


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """简单固定大小切片 + overlap

    按段落优先分割，超过 chunk_size 才强制截断。
    段落内部按句子分割（中文句号/英文句号）。

    Args:
        text: 原始文本
        chunk_size: 每个切片的字符数上限
        overlap: 相邻切片的重叠字符数（当前实现为段落级 overlap）

    Returns:
        文本切片列表
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    paragraphs = text.split("\n\n")

    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # 如果单个段落就超过 chunk_size，按句子强制截断
            if len(para) > chunk_size:
                sentences = _split_sentences(para)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) + 1 <= chunk_size:
                        sub = (sub + "。" + sent).strip("。") if sub else sent
                    else:
                        if sub:
                            chunks.append(sub)
                        # 单句超长，强制按字符截断
                        if len(sent) > chunk_size:
                            for i in range(0, len(sent), chunk_size - overlap):
                                chunks.append(sent[i:i + chunk_size])
                            sub = ""
                        else:
                            sub = sent
                current = sub if sub else ""
            else:
                current = para

    if current and current.strip():
        chunks.append(current)

    return chunks


def _split_sentences(text: str) -> list[str]:
    """按句子分割（中英文句号、问号、感叹号）"""
    import re
    # 按 。！？.!? 后跟空白或结尾分割
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    return [s.strip() for s in sentences if s.strip()]
