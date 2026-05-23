'use client';

import { useCallback, useEffect, useState } from 'react';
import { ComparisonChart } from '@/components/ComparisonChart';
import { ModelInfo } from '@/components/ModelInfo';
import { ResultsCard } from '@/components/ResultsCard';
import { SurpriseHeatmap } from '@/components/SurpriseHeatmap';
import { UploadArea } from '@/components/UploadArea';
import {
  ApiError,
  type CompressionResponse,
  type InfoResponse,
  compressFile,
  fetchInfo,
  findNcompResult,
} from '@/lib/api';
import { sampleBlob } from '@/lib/sample';

const DEFAULT_MAX_BYTES = 4096;

export default function HomePage() {
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [result, setResult] = useState<CompressionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchInfo()
      .then((i) => {
        if (!cancelled) setInfo(i);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const message =
          e instanceof Error ? e.message : 'failed to reach backend; is the API running?';
        setInfoError(message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleFile = useCallback(async (file: File | Blob) => {
    setBusy(true);
    setError(null);
    try {
      const response = await compressFile(file);
      setResult(response);
    } catch (e: unknown) {
      if (e instanceof ApiError) {
        setError(e.message);
      } else if (e instanceof Error) {
        setError(e.message);
      } else {
        setError('unknown error');
      }
    } finally {
      setBusy(false);
    }
  }, []);

  const handleSample = useCallback(() => handleFile(sampleBlob()), [handleFile]);

  const maxBytes = info?.max_upload_bytes ?? DEFAULT_MAX_BYTES;
  const ncompResult = result ? findNcompResult(result.results) : undefined;
  const gzipResult = result?.results.find((r) => r.name === 'gzip -9');

  return (
    <main className="mx-auto max-w-6xl px-4 py-10 sm:py-14">
      <header className="mb-8 flex flex-col gap-2">
        <p className="text-xs uppercase tracking-widest text-ncomp-600 dark:text-ncomp-400">
          Neural Compressor
        </p>
        <h1 className="text-3xl font-semibold text-slate-900 sm:text-4xl dark:text-slate-100">
          A lossless compressor that thinks before it writes
        </h1>
        <p className="max-w-2xl text-sm text-slate-600 sm:text-base dark:text-slate-300">
          A small autoregressive model predicts the next byte; an integer arithmetic coder turns its
          probabilities into the shortest possible bitstream. Upload a short passage of English text
          and see how it compares to gzip, bzip2, and xz.
        </p>
      </header>

      {infoError && (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-700 dark:bg-rose-950/40 dark:text-rose-200"
        >
          Could not reach the API: {infoError}. Start the backend with{' '}
          <code className="font-mono">python scripts/serve.py</code> and refresh.
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="flex flex-col gap-6">
          <UploadArea
            disabled={busy}
            maxBytes={maxBytes}
            onFile={handleFile}
            onUseSample={handleSample}
          />
          {error && (
            <div
              role="alert"
              className="rounded-2xl border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-700 dark:bg-rose-950/40 dark:text-rose-200"
            >
              {error}
            </div>
          )}
          {busy && <BusyBanner />}
          {ncompResult && result && (
            <ResultsCard
              ncomp={ncompResult}
              gzip={gzipResult}
              inputBytes={result.input.size_bytes}
            />
          )}
          {result && <ComparisonChart results={result.results} />}
          {result && (
            <section className="flex flex-col gap-2 rounded-2xl border border-slate-200 bg-white p-5 dark:border-slate-700 dark:bg-slate-900">
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-100">
                Per-byte surprise
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Each cell is one byte of the input. Darker cells cost the model more bits to
                predict. The first byte is always 8 bits (no prior context); look for pale runs
                where the model anticipated common letters cheaply.
              </p>
              <SurpriseHeatmap surprise={result.surprise} />
            </section>
          )}
        </div>
        <ModelInfo info={info} />
      </div>

      <footer className="mt-10 border-t border-slate-200 pt-4 text-center text-xs text-slate-400 dark:border-slate-800">
        Built with PyTorch, FastAPI, and Next.js. Source on{' '}
        <a className="underline" href="https://github.com/joshuacowellcay/neural-compressor">
          GitHub
        </a>
        .
      </footer>
    </main>
  );
}

function BusyBanner() {
  return (
    <div className="border-ncomp-200 flex items-center gap-3 rounded-2xl border bg-ncomp-50 p-4 text-sm text-ncomp-700 dark:border-ncomp-700/60 dark:bg-ncomp-700/10 dark:text-ncomp-100">
      <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-ncomp-500" />
      Compressing on the server; one forward pass per byte takes a moment.
    </div>
  );
}
