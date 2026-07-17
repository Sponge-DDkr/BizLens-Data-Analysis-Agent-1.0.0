import { useState, useRef, useCallback } from 'react';
import FileUpload from './components/FileUpload';
import AnalysisInput from './components/AnalysisInput';
import ProgressBar from './components/ProgressBar';
import ReportView from './components/ReportView';
import KnowledgePanel from './components/KnowledgePanel';
import type {
  FilePreview,
  NodeName,
  StepUpdateEvent,
  DoneEvent,
  AnalysisStatus,
  NodeProgress,
} from './types';
import './App.css';

function makeInitialProgress(): Record<NodeName, NodeProgress> {
  return {
    planner: { status: 'pending' },
    code_interpreter: { status: 'pending' },
    visualization: { status: 'pending' },
    insight: { status: 'pending' },
  };
}

function App() {
  // ── Upload ──
  const [filePreview, setFilePreview] = useState<FilePreview | null>(null);

  // ── Analysis ──
  const [query, setQuery] = useState('');
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus>('idle');
  const [nodeProgress, setNodeProgress] = useState<Record<NodeName, NodeProgress>>(makeInitialProgress);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // ── Results ──
  const [plannerSteps, setPlannerSteps] = useState<NonNullable<StepUpdateEvent['steps']>>([]);
  const [chartJson, setChartJson] = useState<Record<string, unknown> | null>(null);
  const [report, setReport] = useState<string | null>(null);

  // ── Refinement tracking ──
  const [refineQuery, setRefineQuery] = useState('');
  const [analysisCount, setAnalysisCount] = useState(0);

  // AbortController ref for cancelling SSE
  const abortRef = useRef<AbortController | null>(null);

  // ── Upload handler ──
  const handleUploadDone = useCallback((preview: FilePreview) => {
    setFilePreview(preview);
    resetAnalysis();
  }, []);

  const resetAnalysis = useCallback(() => {
    setAnalysisStatus('idle');
    setAnalysisError(null);
    setNodeProgress(makeInitialProgress());
    setPlannerSteps([]);
    setChartJson(null);
    setReport(null);
    setRefineQuery('');
  }, []);

  // ── SSE Analysis Flow ──

  const handleAnalyze = useCallback(async (analysisQuery?: string) => {
    const q = (analysisQuery ?? query).trim();
    if (!q || !filePreview) return;

    // Reset for new analysis
    setAnalysisError(null);
    setNodeProgress(makeInitialProgress());
    setPlannerSteps([]);
    setChartJson(null);
    setReport(null);

    setAnalysisStatus('analyzing');
    abortRef.current = new AbortController();

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          session_id: filePreview.session_id,
        }),
        signal: abortRef.current.signal,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE frames from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // keep incomplete line in buffer

        let currentEvent = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            try {
              const data = JSON.parse(dataStr);
              handleSSEEvent(currentEvent, data);
            } catch {
              // skip malformed JSON
            }
          }
        }
      }

      // Flush remaining buffer
      if (buffer.trim()) {
        const lastLine = buffer.split('\n').find((l) => l.startsWith('data: '));
        if (lastLine) {
          try {
            const data = JSON.parse(lastLine.slice(6));
            handleSSEEvent('', data);
          } catch { /* skip */ }
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      const msg = err instanceof Error ? err.message : '未知错误';
      setAnalysisStatus('error');
      setAnalysisError(msg);
    }
  }, [query, filePreview]);

  const handleSSEEvent = useCallback((eventType: string, data: StepUpdateEvent | DoneEvent) => {
    if (eventType === 'step_update') {
      const evt = data as StepUpdateEvent;
      setNodeProgress((prev) => ({
        ...prev,
        [evt.step]: {
          status: evt.status === 'done' ? 'done' : evt.status === 'error' ? 'error' : 'running',
        },
      }));

      if (evt.status === 'done') {
        if (evt.steps) setPlannerSteps(evt.steps);
        if (evt.chart_json) setChartJson(evt.chart_json);
        if (evt.report) setReport(evt.report);
      }

      if (evt.status === 'error' && evt.error) {
        setAnalysisError(evt.error);
        setAnalysisStatus('error');
      }
    } else if (eventType === 'done') {
      const evt = data as DoneEvent;
      if (evt.status === 'completed') {
        setAnalysisStatus('done');
        setAnalysisCount((c) => c + 1);
      } else {
        setAnalysisStatus('error');
        setAnalysisError(evt.error || '分析流程异常终止');
      }
    }
  }, []);

  // ── Submit: main analysis ──
  const handleMainSubmit = useCallback(() => {
    handleAnalyze(query);
  }, [handleAnalyze, query]);

  // ── Refinement: append refinement query or use as override ──
  const handleRefine = useCallback((prompt: string) => {
    setRefineQuery(prompt);
    // Build a combined query: original context + refinement angle
    const combinedQuery = `${query}\n\n追加分析角度：${prompt}`;
    handleAnalyze(combinedQuery);
  }, [handleAnalyze, query]);

  const handleRefineSubmit = useCallback(() => {
    if (refineQuery.trim()) {
      handleRefine(refineQuery.trim());
    }
  }, [handleRefine, refineQuery]);

  // ── Derived state ──
  const isAnalyzing = analysisStatus === 'analyzing';
  const hasResults = Boolean(chartJson || report);

  return (
    <div className="app-container">
      {/* ── Header ── */}
      <header className="app-header">
        <h1 className="app-title">
          <span className="logo">🔍</span> BizLens
        </h1>
        <p className="app-subtitle">智能数据分析助手 — 上传数据，打字提问，拿报告</p>
      </header>

      {/* ── Main Content ── */}
      <main className="app-main">
        {/* Step 1: Upload — always visible */}
        <section className="step-section">
          <div className="step-number">1</div>
          <h2>上传数据文件</h2>
          <FileUpload onUploadDone={handleUploadDone} />
        </section>

        {/* Knowledge Base Management */}
        <KnowledgePanel />

        {/* Step 2: Analysis — visible after upload */}
        {filePreview && (
          <section className="step-section">
            <div className="step-number">2</div>
            <h2>输入分析问题</h2>

            {/* Main analysis input (hidden after successful analysis if report exists) */}
            {!hasResults && (
              <AnalysisInput
                query={query}
                onChange={setQuery}
                onSubmit={handleMainSubmit}
                disabled={isAnalyzing}
                mode="initial"
              />
            )}

            {/* Show query + "ask again" when results exist */}
            {hasResults && (
              <div className="current-query-display">
                <span className="current-query-label">当前分析：</span>
                <span className="current-query-text">
                  {query.length > 80 ? query.slice(0, 80) + '...' : query}
                </span>
                <button
                  className="new-analysis-btn"
                  onClick={() => {
                    resetAnalysis();
                    setQuery('');
                  }}
                >
                  ✨ 新分析
                </button>
              </div>
            )}

            {/* Progress bar — driven by SSE events */}
            <ProgressBar
              nodeProgress={nodeProgress}
              error={analysisError}
            />

            {/* Planner steps — shown after planner completes */}
            {plannerSteps.length > 0 && (
              <div className="planner-output">
                <h3>📋 分析步骤</h3>
                <ol className="step-list">
                  {plannerSteps.map((s) => (
                    <li key={s.step} className={`step-item step-type-${s.type}`}>
                      <span className="step-badge">
                        {s.type === 'code' ? '🐍' : s.type === 'chart' ? '📊' : '📝'}
                      </span>
                      {s.description}
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </section>
        )}

        {/* Empty state — no file uploaded yet */}
        {!filePreview && (
          <section className="step-section empty-state-section">
            <div className="empty-state">
              <div className="empty-state-icon">👆</div>
              <p className="empty-state-title">上传文件开始分析</p>
              <p className="empty-state-desc">
                支持 CSV / Excel 格式，最大 10MB。上传后输入分析问题，AI 将自动完成数据探索、可视化与报告生成。
              </p>
              <div className="empty-state-features">
                <div className="empty-feature">
                  <span className="empty-feature-icon">🔍</span>
                  <span>智能步骤拆解</span>
                </div>
                <div className="empty-feature">
                  <span className="empty-feature-icon">🐍</span>
                  <span>Python 代码自动执行</span>
                </div>
                <div className="empty-feature">
                  <span className="empty-feature-icon">📊</span>
                  <span>交互式可视化</span>
                </div>
                <div className="empty-feature">
                  <span className="empty-feature-icon">📝</span>
                  <span>四段式专业报告</span>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ── Results Section ── */}
        {hasResults && (
          <section className="step-section results-section">
            <div className="step-number">3</div>
            <h2>分析结果 {analysisCount > 1 ? `(第 ${analysisCount} 次)` : ''}</h2>

            <ReportView
              chartJson={chartJson}
              report={report}
              onRefine={handleRefine}
              disabled={isAnalyzing}
            />

            {/* Export button */}
            <button
              className="export-btn"
              onClick={() => window.print()}
            >
              📥 导出 PDF
            </button>

            {/* ── Refinement input ── */}
            <div className="refine-input-section">
              <AnalysisInput
                query={refineQuery}
                onChange={setRefineQuery}
                onSubmit={handleRefineSubmit}
                disabled={isAnalyzing}
                mode="refine"
              />
            </div>
          </section>
        )}

        {/* Error state — analysis failed without any partial results */}
        {analysisStatus === 'error' && !hasResults && (
          <section className="step-section error-section">
            <div className="error-state">
              <span className="error-state-icon">⚠️</span>
              <div>
                <p className="error-state-title">分析未完成</p>
                <p className="error-state-desc">{analysisError || '发生了未知错误，请重试'}</p>
                <button
                  className="retry-btn"
                  onClick={() => handleAnalyze(query)}
                >
                  🔄 重试分析
                </button>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <p>BizLens — Data Analysis Agent · v1.0.0</p>
      </footer>
    </div>
  );
}

export default App;
