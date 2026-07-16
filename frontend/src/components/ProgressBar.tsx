/**
 * ProgressBar — SSE event-driven component showing live Agent node status.
 *
 * Displays the 4-agent pipeline as a horizontal step indicator:
 *   planner → code_interpreter → visualization → insight
 *
 * Each step shows: ○ pending / ⏳ running / ✅ done / ❌ error
 * With human-readable labels and animated connectors.
 *
 * Day 6: Extracted from App.tsx inline progress code.
 */

import type { NodeName, NodeProgress } from '../types';

export interface ProgressBarProps {
  nodeProgress: Record<NodeName, NodeProgress>;
  error: string | null;
}

/** Human-readable labels while a node is running */
const NODE_RUNNING_LABELS: Record<NodeName, string> = {
  planner: '🔍 正在分析问题结构...',
  code_interpreter: '🐍 正在执行数据分析...',
  visualization: '📊 正在生成图表...',
  insight: '📝 正在生成分析报告...',
};

const NODE_DONE_LABELS: Record<NodeName, string> = {
  planner: '✅ 步骤拆解完成',
  code_interpreter: '✅ 数据分析完成',
  visualization: '✅ 图表生成完成',
  insight: '✅ 报告生成完成',
};

const NODE_ORDER: NodeName[] = ['planner', 'code_interpreter', 'visualization', 'insight'];

export default function ProgressBar({ nodeProgress, error }: ProgressBarProps) {
  const allDone = NODE_ORDER.every((n) => nodeProgress[n].status === 'done');
  const hasStarted = NODE_ORDER.some((n) => nodeProgress[n].status !== 'pending');

  if (!hasStarted) return null;

  return (
    <div className={`progress-panel ${allDone ? 'progress-panel--done' : ''}`}>
      <h3 className="progress-title">
        {allDone ? '🎉 分析完成' : '⚡ 分析进度'}
      </h3>

      <div className="progress-steps">
        {NODE_ORDER.map((node, idx) => {
          const prog = nodeProgress[node];
          const isActive = prog.status === 'running';
          const isDone = prog.status === 'done';
          const isError = prog.status === 'error';

          const label = isActive
            ? NODE_RUNNING_LABELS[node]
            : isDone
              ? NODE_DONE_LABELS[node]
              : isError
                ? `❌ ${node} 出错`
                : node;

          return (
            <div
              key={node}
              className={`progress-step ${isActive ? 'progress-step--active' : ''} ${isDone ? 'progress-step--done' : ''} ${isError ? 'progress-step--error' : ''}`}
            >
              <span className="progress-dot">
                {isDone ? '✅' : isError ? '❌' : isActive ? '⏳' : '○'}
              </span>
              <span className="progress-label">{label}</span>
              {idx < NODE_ORDER.length - 1 && (
                <span
                  className={`progress-connector ${isDone ? 'progress-connector--done' : ''}`}
                >
                  →
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Error display */}
      {error && (
        <div className="analysis-error">
          <strong>❌ 分析出错：</strong> {error}
        </div>
      )}
    </div>
  );
}
