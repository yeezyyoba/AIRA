/**
 * AIRA API Service Client
 * Connects Next.js client/server to the FastAPI backend.
 */
import { ResearchResponse, SystemHealth } from "@/types/aira";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiClientError extends Error {
  constructor(
    message: string,
    public status?: number,
    public data?: any
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers || {}),
      },
    });

    if (!res.ok) {
      let errData = null;
      try {
        errData = await res.json();
      } catch {
        errData = await res.text();
      }
      throw new ApiClientError(
        errData?.detail || `API Request failed with status ${res.status}`,
        res.status,
        errData
      );
    }

    return await res.json();
  } catch (err: any) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(err.message || "Failed to connect to AIRA backend service.");
  }
}

export const airaApi = {
  /**
   * Health check
   */
  async getHealth(): Promise<SystemHealth> {
    return request<SystemHealth>("/api/v1/health");
  },

  /**
   * Start a new research session
   */
  async startSession(query: string): Promise<ResearchResponse> {
    return request<ResearchResponse>("/api/v1/research/sessions", {
      method: "POST",
      body: JSON.stringify({ query }),
    });
  },

  /**
   * Get state of an active, paused, or finished research session
   */
  async getSession(sessionId: string): Promise<ResearchResponse> {
    return request<ResearchResponse>(`/api/v1/research/sessions/${sessionId}`);
  },

  /**
   * Resume a paused session with clarification
   */
  async resumeSession(sessionId: string, clarificationResponse: string): Promise<ResearchResponse> {
    return request<ResearchResponse>(`/api/v1/research/sessions/${sessionId}/resume`, {
      method: "POST",
      body: JSON.stringify({ clarification_response: clarificationResponse }),
    });
  },

  /**
   * List papers in the repository
   */
  async listPapers(): Promise<{ papers: string[]; extracted_papers: string[]; total_extracted: number; total_indexed: number }> {
    return request("/api/v1/papers/");
  },

  /**
   * List generated reports
   */
  async listReports(): Promise<{ reports: any[]; total: number }> {
    return request("/api/v1/reports/");
  },

  /**
   * Get direct URL to export report as Markdown or JSON
   */
  getReportExportUrl(reportId: string, format: "markdown" | "json"): string {
    return `${API_BASE_URL}/api/v1/reports/${reportId}/export?format=${format}`;
  },
};
