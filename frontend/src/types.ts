/** BizLens shared type definitions */

export interface FilePreview {
  session_id: string;
  filename: string;
  columns: string[];
  column_count: number;
  row_count: number;
  file_size_bytes: number;
  file_size_display: string;
  dtypes: Record<string, string>;
  sample_data: Record<string, unknown>[];
  sample_rows: number;
}

export interface UploadState {
  status: 'idle' | 'uploading' | 'done' | 'error';
  progress: number;
  preview: FilePreview | null;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Analysis / SSE types (Day 2+)
// ---------------------------------------------------------------------------

export interface AnalysisStepInfo {
  step: number;
  description: string;
  type: 'code' | 'chart' | 'insight';
}

export type NodeName = 'planner' | 'code_interpreter' | 'visualization' | 'insight';

export interface StepUpdateEvent {
  step: NodeName;
  status: 'running' | 'done' | 'error';
  label: string;
  // Payload depends on the step:
  steps?: AnalysisStepInfo[];       // planner done
  expected_output?: string;         // planner done
  chart_json?: Record<string, unknown>; // visualization done
  report?: string;                  // insight done
  error?: string;                   // any step error
}

export interface DoneEvent {
  status: 'completed' | 'error';
  error?: string;
}

export type AnalysisStatus = 'idle' | 'analyzing' | 'done' | 'error';

export interface NodeProgress {
  status: 'pending' | 'running' | 'done' | 'error';
}
