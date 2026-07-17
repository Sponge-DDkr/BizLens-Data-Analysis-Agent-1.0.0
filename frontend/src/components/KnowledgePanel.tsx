/**
 * KnowledgePanel — Knowledge base document management.
 *
 * Provides upload + list + delete for the MCP Knowledge Server,
 * proxied through the BizLens backend (/api/knowledge/*).
 */

import { useState, useEffect, useCallback, useRef, type DragEvent } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface KnowledgeDoc {
  doc_id?: string;
  id?: string;
  filename?: string;
  file_name?: string;
  name?: string;
  category?: string;
  chunks?: number;
  chunk_count?: number;
  file_size?: number;
  added_at?: string;
  created_at?: string;
}

interface KnowledgeStats {
  available?: boolean;
  total_files?: number;
  file_count?: number;
  document_count?: number;
  total_chunks?: number;
  chunk_count?: number;
  vector_dim?: number;
  embedding_dim?: number;
  storage_path?: string;
  storage_size?: string;
  storage_size_mb?: number;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function KnowledgePanel() {
  const [docs, setDocs] = useState<KnowledgeDoc[]>([]);
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [kbUnavailable, setKbUnavailable] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Fetch documents + stats ──
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [docsRes, statsRes] = await Promise.all([
        fetch('/api/knowledge/documents'),
        fetch('/api/knowledge/stats'),
      ]);

      if (docsRes.ok) {
        const data = await docsRes.json();
        // Backend distinguishes "empty" from "MCP server unreachable"
        if (data.available === false) {
          setKbUnavailable(true);
          setDocs([]);
        } else {
          setKbUnavailable(false);
          setDocs(data.documents ?? []);
        }
      } else {
        setKbUnavailable(true);
        setError(`知识库查询失败 (HTTP ${docsRes.status})`);
      }
      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data);
      }
    } catch (e) {
      // Backend itself unreachable
      setKbUnavailable(true);
      setError('无法连接后端服务');
      console.warn('KnowledgePanel: fetch failed', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (expanded) {
      fetchData();
    }
  }, [expanded, fetchData]);

  // ── Upload ──
  const handleUpload = useCallback(async (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!ext || !['pdf', 'txt', 'md', 'markdown'].includes(ext)) {
      setError('仅支持 PDF / TXT / MD 格式');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/api/knowledge/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const result = await res.json();
      console.log('Knowledge uploaded:', result);
      await fetchData(); // refresh list
    } catch (e) {
      const msg = e instanceof Error ? e.message : '上传失败';
      setError(msg);
    } finally {
      setUploading(false);
    }
  }, [fetchData]);

  // ── Delete ──
  const handleDelete = useCallback(async (docId: string) => {
    try {
      const res = await fetch(`/api/knowledge/${encodeURIComponent(docId)}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      await fetchData();
    } catch (e) {
      const msg = e instanceof Error ? e.message : '删除失败';
      setError(msg);
    }
  }, [fetchData]);

  // ── File input ──
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
    if (e.target) e.target.value = '';
  }, [handleUpload]);

  // ── Drag & Drop ──
  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);
  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);
  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleUpload(file);
  }, [handleUpload]);

  // ── Helpers ──
  const getDocId = (doc: KnowledgeDoc): string =>
    doc.doc_id || doc.id || '';

  const getDocName = (doc: KnowledgeDoc): string =>
    doc.filename || doc.file_name || doc.name || '未知文件';

  const getChunkCount = (doc: KnowledgeDoc): number =>
    doc.chunks ?? doc.chunk_count ?? 0;

  // ── Render ──
  return (
    <section className="step-section knowledge-panel">
      <div
        className="knowledge-header"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="step-number knowledge-step-num">KB</div>
        <h2>📚 知识库管理</h2>
        <span className="knowledge-toggle">{expanded ? '▾' : '▸'}</span>
      </div>

      {!expanded && (
        <p className="knowledge-hint">
          上传行业报告、竞品分析等文档，Insight Agent 将在分析中自动引用。
          {docs.length > 0 && (
            <span className="knowledge-badge">{docs.length} 份文档</span>
          )}
        </p>
      )}

      {expanded && (
        <div className="knowledge-body">
          {/* ── Stats bar ── */}
          {stats && stats.available !== false && (
            <div className="knowledge-stats">
              <span className="knowledge-stat">
                📄 {(stats.total_files ?? stats.file_count ?? stats.document_count ?? docs.length)} 份文档
              </span>
              <span className="knowledge-stat-sep">·</span>
              <span className="knowledge-stat">
                🧩 {(stats.total_chunks ?? stats.chunk_count ?? 0)} 个切片
              </span>
              {(stats.storage_size_mb != null || stats.storage_size) && (
                <>
                  <span className="knowledge-stat-sep">·</span>
                  <span className="knowledge-stat">
                    💾 {stats.storage_size ?? `${stats.storage_size_mb} MB`}
                  </span>
                </>
              )}
            </div>
          )}

          {/* ── Upload zone ── */}
          <div
            className={`knowledge-dropzone ${dragOver ? 'knowledge-dropzone--active' : ''} ${uploading ? 'knowledge-dropzone--uploading' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt,.md,.markdown"
              className="file-input-hidden"
              onChange={handleFileChange}
            />
            {uploading ? (
              <span className="knowledge-drop-text">
                <span className="loading-spinner" /> 正在导入知识库...
              </span>
            ) : (
              <span className="knowledge-drop-text">
                📤 拖拽或点击上传文档（PDF / TXT / MD）
              </span>
            )}
          </div>

          {/* ── Error ── */}
          {error && (
            <div className="knowledge-error">
              ⚠️ {error}
              <button className="knowledge-error-dismiss" onClick={() => setError(null)}>×</button>
            </div>
          )}

          {/* ── Document list ── */}
          {loading ? (
            <p className="knowledge-loading">加载中...</p>
          ) : kbUnavailable ? (
            <p className="knowledge-empty">
              ⚠️ 知识库服务暂不可用（MCP Server 可能正在启动或已离线）。
              <button className="knowledge-retry-btn" onClick={fetchData}>重试</button>
            </p>
          ) : docs.length === 0 ? (
            <p className="knowledge-empty">
              知识库为空 — 上传行业报告、竞品分析等文档，Insight Agent 将在分析中自动检索引用。
            </p>
          ) : (
            <div className="knowledge-doc-list">
              {docs.map((doc) => {
                const docId = getDocId(doc);
                return (
                  <div key={docId} className="knowledge-doc-item">
                    <div className="knowledge-doc-info">
                      <span className="knowledge-doc-icon">
                        {getDocName(doc).endsWith('.pdf') ? '📕' : '📝'}
                      </span>
                      <span className="knowledge-doc-name">{getDocName(doc)}</span>
                      {doc.category && doc.category !== 'general' && (
                        <span className="knowledge-doc-category">{doc.category}</span>
                      )}
                      <span className="knowledge-doc-chunks">
                        {getChunkCount(doc)} chunks
                      </span>
                    </div>
                    <button
                      className="knowledge-doc-delete"
                      onClick={() => handleDelete(docId)}
                      title="删除文档"
                    >
                      🗑️
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
