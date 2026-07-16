/**
 * AnalysisInput — Query input with quick-fill examples and refinement mode.
 *
 * Two modes:
 *   'initial' — Full input with example prompts (shown before analysis starts)
 *   'refine'  — Compact input below the report for follow-up adjustments
 *
 * Day 6: Extracted from App.tsx, added refinement mode with quick-action chips.
 */

import { type KeyboardEvent } from 'react';

export interface AnalysisInputProps {
  query: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  /** 'initial' = main input with examples; 'refine' = compact follow-up below report */
  mode: 'initial' | 'refine';
  /** Only used in 'initial' mode */
  examplePrompts?: string[];
  /** Only used in 'refine' mode — quick-action refinement chips */
  refineActions?: string[];
  /** Placeholder override */
  placeholder?: string;
}

const DEFAULT_EXAMPLES = [
  'Q3各产品线营收趋势，哪个增长最快？',
  '各产品线利润率对比分析',
  '按月度拆分营收，找出增长异常点',
];

const DEFAULT_REFINE_ACTIONS = [
  '按月度拆分',
  '加上利润率',
  '只看Top3',
  '按地区分组',
];

export default function AnalysisInput({
  query,
  onChange,
  onSubmit,
  disabled,
  mode,
  examplePrompts = DEFAULT_EXAMPLES,
  refineActions = DEFAULT_REFINE_ACTIONS,
  placeholder,
}: AnalysisInputProps) {
  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && query.trim() && !disabled) {
      onSubmit();
    }
  };

  const defaultPlaceholder =
    mode === 'refine'
      ? '微调问题，例如："按月度拆分"、"只看Top3"...'
      : '例如："Q3各产品线营收趋势，哪个增长最快？"';

  if (mode === 'refine') {
    return (
      <div className="analysis-input analysis-input--refine">
        <div className="refine-header">
          <span className="refine-icon">🔧</span>
          <span className="refine-label">微调分析</span>
          <span className="refine-hint">— 输入新的分析角度，重新生成报告</span>
        </div>

        <div className="analysis-input-row">
          <input
            type="text"
            className="query-input query-input--refine"
            placeholder={placeholder ?? defaultPlaceholder}
            value={query}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
          />
          <button
            className="analyze-btn analyze-btn--refine"
            onClick={onSubmit}
            disabled={!query.trim() || disabled}
          >
            {disabled ? '分析中...' : '🔄 重新分析'}
          </button>
        </div>

        {/* Refinement quick-action chips */}
        <div className="refine-chips">
          <span className="refine-chips-label">快捷微调：</span>
          {refineActions.map((action) => (
            <button
              key={action}
              className="refine-chip"
              onClick={() => {
                onChange(action);
                // Auto-submit after a short delay for better UX
                setTimeout(() => onSubmit(), 150);
              }}
              disabled={disabled}
            >
              {action}
            </button>
          ))}
        </div>
      </div>
    );
  }

  // ── Initial mode ──
  return (
    <div className="analysis-input analysis-input--initial">
      <div className="analysis-input-row">
        <input
          type="text"
          className="query-input"
          placeholder={placeholder ?? defaultPlaceholder}
          value={query}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          className="analyze-btn"
          onClick={onSubmit}
          disabled={!query.trim() || disabled}
        >
          {disabled ? (
            <span className="btn-loading">
              <span className="loading-spinner" /> 分析中...
            </span>
          ) : (
            '🚀 开始分析'
          )}
        </button>
      </div>

      {/* Quick-fill example prompts */}
      <div className="quick-prompts">
        <span className="quick-prompts-label">💡 试试这些问题：</span>
        {examplePrompts.map((prompt) => (
          <button
            key={prompt}
            className="quick-prompt-btn"
            onClick={() => onChange(prompt)}
            disabled={disabled}
            title={`填入：「${prompt}」`}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
