/**
 * ReportView — Chart rendering + Markdown report with refinement quick-actions.
 *
 * Day 6 improvements:
 *   - Uses react-markdown (already in package.json) instead of hand-rolled regex
 *   - Custom renderers for tables, headings, code blocks
 *   - Refinement quick-action chips below the report
 *   - Better section partitioning with visual dividers
 */

import { useEffect, useRef, useCallback, type ReactNode } from 'react';
import Plotly from 'plotly.js-dist-min';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ReportViewProps {
  chartJson: Record<string, unknown> | null;
  report: string | null;
  /** Called when user clicks a refinement quick-action chip */
  onRefine?: (prompt: string) => void;
  /** Whether analysis is currently running (disables refine buttons) */
  disabled?: boolean;
}

/** Quick-action refinement chips shown below the report */
const REFINE_CHIPS = [
  { label: '📅 按月度拆分', prompt: '按月度拆分' },
  { label: '💰 加上利润率', prompt: '加上利润率分析' },
  { label: '🏆 只看Top3', prompt: '只看Top3' },
  { label: '📊 按地区分组', prompt: '按地区分组分析' },
];

// ---------------------------------------------------------------------------
// Custom react-markdown renderers
// ---------------------------------------------------------------------------

interface MarkdownComponentProps {
  children?: ReactNode;
}

function MarkdownH2({ children, ...props }: MarkdownComponentProps) {
  return (
    <h2 className="report-h2" {...props}>
      {children}
    </h2>
  );
}

function MarkdownH3({ children, ...props }: MarkdownComponentProps) {
  return (
    <h3 className="report-h3" {...props}>
      {children}
    </h3>
  );
}

function MarkdownH4({ children, ...props }: MarkdownComponentProps) {
  return (
    <h4 className="report-h4" {...props}>
      {children}
    </h4>
  );
}

function MarkdownTable({ children, ...props }: MarkdownComponentProps) {
  return (
    <div className="report-table-wrapper">
      <table className="report-table" {...props}>
        {children}
      </table>
    </div>
  );
}

function MarkdownStrong({ children, ...props }: MarkdownComponentProps) {
  return (
    <strong className="report-strong" {...props}>
      {children}
    </strong>
  );
}

function MarkdownParagraph({ children, ...props }: MarkdownComponentProps) {
  return (
    <p className="report-paragraph" {...props}>
      {children}
    </p>
  );
}

function MarkdownList({ children, ...props }: MarkdownComponentProps) {
  return (
    <ul className="report-list" {...props}>
      {children}
    </ul>
  );
}

function MarkdownOrderedList({ children, ...props }: MarkdownComponentProps) {
  return (
    <ol className="report-ordered-list" {...props}>
      {children}
    </ol>
  );
}

function MarkdownCode({ children, ...props }: MarkdownComponentProps) {
  return (
    <code className="report-code" {...props}>
      {children}
    </code>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ReportView({
  chartJson,
  report,
  onRefine,
  disabled = false,
}: ReportViewProps) {
  const chartRef = useRef<HTMLDivElement>(null);

  // ── Render chart with Plotly.react ──
  useEffect(() => {
    if (!chartJson || !chartRef.current) return;

    const data = (chartJson.data ?? []) as Plotly.Data[];
    const rawLayout = (chartJson.layout ?? {}) as Record<string, unknown>;

    // ── Top area protection: reserve space for title, prevent overlap ──
    const margin = { ...(rawLayout.margin as Record<string, number> | undefined) };
    margin.t = Math.max(margin.t ?? 60, 100);  // at least 100px top margin

    const title =
      typeof rawLayout.title === 'string'
        ? { text: rawLayout.title, y: 0.96 }
        : { ...(rawLayout.title as Record<string, unknown> | undefined), y: 0.96 };

    const layout = { ...rawLayout, margin, title } as Partial<Plotly.Layout>;
    const config: Partial<Plotly.Config> = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: 'bizlens_chart',
        height: 600,
        width: 900,
      },
      displayModeBar: true,
    };

    Plotly.react(chartRef.current, data, layout, config);

    return () => {
      if (chartRef.current) {
        Plotly.purge(chartRef.current);
      }
    };
  }, [chartJson]);

  // ── Download PNG ──
  const handleDownloadPng = useCallback(() => {
    if (!chartRef.current) return;
    Plotly.downloadImage(chartRef.current, {
      format: 'png',
      width: 900,
      height: 600,
      filename: 'bizlens_chart',
    });
  }, []);

  if (!chartJson && !report) return null;

  return (
    <div className="report-view">
      {/* ── Chart section ── */}
      {chartJson && (
        <div className="chart-section">
          <div className="chart-header">
            <h3>📊 数据可视化</h3>
            <div className="chart-actions">
              <button className="download-png-btn" onClick={handleDownloadPng}>
                📥 下载 PNG
              </button>
            </div>
          </div>
          <div className="chart-container">
            <div ref={chartRef} className="chart-plot" />
          </div>
        </div>
      )}

      {/* ── Report section with react-markdown ── */}
      {report && (
        <div className="report-section">
          <div className="report-section-header">
            <h3>📄 分析报告</h3>
          </div>

          <div className="report-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h2: MarkdownH2,
                h3: MarkdownH3,
                h4: MarkdownH4,
                table: MarkdownTable,
                strong: MarkdownStrong,
                p: MarkdownParagraph,
                ul: MarkdownList,
                ol: MarkdownOrderedList,
                code: MarkdownCode,
              }}
            >
              {report}
            </ReactMarkdown>
          </div>

          {/* ── Refinement quick-actions ── */}
          {onRefine && (
            <div className="refine-actions">
              <div className="refine-divider" />
              <p className="refine-actions-label">
                🔧 对结果不满意？快速微调：
              </p>
              <div className="refine-actions-row">
                {REFINE_CHIPS.map((chip) => (
                  <button
                    key={chip.prompt}
                    className="refine-action-chip"
                    onClick={() => onRefine(chip.prompt)}
                    disabled={disabled}
                  >
                    {chip.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
