// Typed client for the Neural Compressor backend.

export interface ToolResult {
  name: string;
  compressed_bytes: number;
  ratio: number;
  bits_per_byte: number;
  compress_seconds: number;
  decompress_seconds: number;
  ok: boolean;
}

export interface InputSummary {
  size_bytes: number;
  preview: string;
  preview_truncated: boolean;
}

export interface SurpriseSeries {
  bits: number[];
  bytes: number[];
}

export interface CompressionResponse {
  input: InputSummary;
  results: ToolResult[];
  surprise: SurpriseSeries;
}

export interface InfoResponse {
  model_loaded: boolean;
  n_parameters: number | null;
  context_length: number | null;
  d_model: number | null;
  n_layers: number | null;
  n_heads: number | null;
  fingerprint_hex: string | null;
  eval_bits_per_byte: number | null;
  max_upload_bytes: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) {
    throw new ApiError(`GET ${path} failed: ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

export async function fetchInfo(): Promise<InfoResponse> {
  return getJson<InfoResponse>('/api/info');
}

export async function compressFile(file: File | Blob): Promise<CompressionResponse> {
  const form = new FormData();
  const name = file instanceof File ? file.name : 'sample.txt';
  form.append('file', file, name);
  const res = await fetch('/api/compress', { method: 'POST', body: form });
  if (!res.ok) {
    let message: string;
    try {
      const body = await res.json();
      message = body?.detail ?? `compress failed: ${res.status}`;
    } catch {
      message = `compress failed: ${res.status}`;
    }
    throw new ApiError(message, res.status);
  }
  return (await res.json()) as CompressionResponse;
}

export const NCOMP_TOOL_NAME = 'ncomp';

export function findNcompResult(results: ToolResult[]): ToolResult | undefined {
  return results.find((r) => r.name === NCOMP_TOOL_NAME);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatSeconds(seconds: number): string {
  if (seconds < 0.001) return `${(seconds * 1_000_000).toFixed(0)} us`;
  if (seconds < 1) return `${(seconds * 1000).toFixed(1)} ms`;
  return `${seconds.toFixed(2)} s`;
}
