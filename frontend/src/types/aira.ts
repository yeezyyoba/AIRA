/**
 * AIRA Frontend TypeScript Type Definitions
 * Accurately aligned with backend FastAPI Pydantic schemas (validation/schemas.py & app/schemas/research.py).
 */

export interface ObservableLog {
  timestamp: string;
  step: number;
  node: string;
  decision: string;
  tool: string | null;
  outcome: string;
  details?: Record<string, any>;
}

export interface ResearchPlanTask {
  task_id: string;
  name: string;
  description: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "ADAPTED";
}

export interface DiscoveredPaper {
  id?: string;
  paper_id?: string;
  title: string;
  authors?: string[];
  year?: number | null;
  doi?: string | null;
  abstract?: string | null;
  venue?: string | null;
  url?: string | null;
  source?: string;
}

export interface ValidatedPaper {
  status: "VERIFIED" | "PARTIALLY_VERIFIED" | "CONFLICTING" | "UNVERIFIED";
  paper_id: string;
  doi?: string | null;
  explanation?: string;
}

export interface EvidenceItem {
  paper_id: string;
  page_number?: number | null;
  section_heading?: string | null;
  passage: string;
  evidence_type: "DIRECT_EXTRACT" | "SUMMARY" | "INFERENCE" | string;
}

export interface MetricRankedModel {
  paper_id: string;
  model_name: string;
  metric_name?: string;
  raw_value: number | string;
  normalized_value?: number | null;
  rank?: number;
  dataset_name?: string | null;
  dataset_split?: string | null;
  page_number?: number | null;
  evidence_passage?: string | null;
}

export interface MetricComparison {
  metric_name: string;
  goal?: "HIGHER_IS_BETTER" | "LOWER_IS_BETTER" | string;
  compatibility: "COMPATIBLE" | "INCOMPATIBLE" | "UNKNOWN" | string;
  incompatibility_reason?: string | null;
  ranked_models?: MetricRankedModel[];
  summary_table?: MetricRankedModel[];
  missing_data_warnings?: string[];
  trade_off_notes?: string | null;
  notes?: string | null;
  // Legacy or flattened fallback fields
  dataset?: string;
  split?: string;
  is_comparable?: boolean;
  paper_values?: Record<string, any>;
}

export interface ResearchGap {
  gap_id?: string;
  title?: string;
  description: string;
  gap_type?: "AUTHOR_ACKNOWLEDGED_GAP" | "CROSS_PAPER_GAP" | "POTENTIAL_RESEARCH_DIRECTION" | string;
  supporting_papers?: string[];
  evidence?: EvidenceItem[];
  suggested_direction?: string | null;
  // Legacy / fallback fields
  category?: string;
  suggested_future_work?: string;
  evidence_backing?: string;
}

export interface ReportReference {
  paper_id: string;
  title: string;
  authors?: string[];
  page_count?: number;
  citation?: string;
  doi?: string;
}

export interface ResearchComparisonReport {
  report_id: string;
  title?: string;
  research_question: string;
  papers_analyzed: string[];
  objectives_and_methodology_comparison?: string;
  dataset_comparison?: string;
  model_architecture_comparison?: string;
  experimental_results_comparison?: string;
  strengths_and_limitations?: string;
  research_gaps?: ResearchGap[];
  evidence_backed_conclusions?: string;
  limitations_of_analysis?: string;
  executive_summary?: string;
  references?: ReportReference[];
  metric_tables?: MetricComparison[];
  is_degraded?: boolean;
  created_at?: string;
}

export interface ResearchResponse {
  session_id: string;
  status: string;
  research_goal: string;
  query?: string;
  clarification_question?: string | null;
  missing_information?: string[] | null;
  plan?: ResearchPlanTask[] | null;
  final_report?: ResearchComparisonReport | null;
  observable_logs?: ObservableLog[];
  discovered_papers?: DiscoveredPaper[];
  validated_papers?: ValidatedPaper[];
  extracted_papers?: string[];
  metric_comparisons?: MetricComparison[];
  identified_gaps?: ResearchGap[];
  citations?: Record<string, string>;
  is_degraded?: boolean;
}

export interface SystemHealthServices {
  orchestration?: string;
  llm_router?: string;
  llm_primary?: string;
  llm_fallback?: string;
  embedding_provider?: string;
  database?: string;
  vector_store?: string;
}

export interface SystemHealth {
  status: string;
  version: string;
  phase?: string;
  services?: SystemHealthServices;
  environment?: string;
  database?: string;
  vector_store?: string;
  primary_llm?: string;
  fallback_llm?: string;
  degraded_mode?: boolean;
}
