import { useState, useRef, type DragEvent, type ChangeEvent } from 'react';
import axios from 'axios';
import type { FilePreview, UploadState } from '../types';

const API_BASE = '/api';

export default function FileUpload({
  onUploadDone,
}: {
  onUploadDone: (preview: FilePreview) => void;
}) {
  const [state, setState] = useState<UploadState>({
    status: 'idle',
    progress: 0,
    preview: null,
    error: null,
  });
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (file: File) => {
    // Client-side validation
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!['.csv', '.xlsx'].includes(ext)) {
      setState({ status: 'error', progress: 0, preview: null, error: '仅支持 .csv 和 .xlsx 文件' });
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setState({ status: 'error', progress: 0, preview: null, error: '文件不能超过 10MB' });
      return;
    }

    setState({ status: 'uploading', progress: 0, preview: null, error: null });

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post<FilePreview>(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
          if (e.total) {
            const pct = Math.round((e.loaded * 100) / e.total);
            setState((s) => ({ ...s, progress: pct }));
          }
        },
      });

      setState({ status: 'done', progress: 100, preview: res.data, error: null });
      onUploadDone(res.data);
    } catch (err: unknown) {
      const msg =
        axios.isAxiosError(err) && err.response?.data?.detail
          ? err.response.data.detail
          : '上传失败，请重试';
      setState({ status: 'error', progress: 0, preview: null, error: msg });
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(true);
  };

  const onDragLeave = () => setDragOver(false);

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  const isUploading = state.status === 'uploading';

  return (
    <div className="file-upload-section">
      {/* Drop Zone */}
      <div
        className={`drop-zone ${dragOver ? 'drag-over' : ''} ${isUploading ? 'uploading' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !isUploading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx"
          className="file-input-hidden"
          onChange={onFileChange}
          disabled={isUploading}
        />

        {state.status === 'idle' && (
          <div className="drop-prompt">
            <span className="drop-icon">📤</span>
            <p>拖拽 CSV / Excel 文件到这里，或点击选择</p>
            <p className="drop-hint">支持 .csv / .xlsx，最大 10MB</p>
          </div>
        )}

        {state.status === 'uploading' && (
          <div className="upload-progress">
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: `${state.progress}%` }} />
            </div>
            <p>上传中 {state.progress}%</p>
          </div>
        )}

        {state.status === 'error' && (
          <div className="drop-prompt error">
            <span className="drop-icon">❌</span>
            <p className="error-text">{state.error}</p>
            <p className="drop-hint">点击重试</p>
          </div>
        )}
      </div>

      {/* Preview Panel */}
      {state.preview && (
        <div className="file-preview">
          <div className="preview-header">
            <span className="preview-icon">📄</span>
            <span className="preview-filename">{state.preview.filename}</span>
          </div>
          <div className="preview-stats">
            <span className="stat">{state.preview.column_count} 列</span>
            <span className="stat-separator">·</span>
            <span className="stat">{state.preview.row_count.toLocaleString()} 行</span>
            <span className="stat-separator">·</span>
            <span className="stat">{state.preview.file_size_display}</span>
          </div>
          <details className="preview-columns">
            <summary>列名与类型</summary>
            <div className="columns-grid">
              {state.preview.columns.map((col) => (
                <span key={col} className="column-tag">
                  <code>{col}</code>
                  <span className="dtype">{state.preview!.dtypes[col]}</span>
                </span>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
}
