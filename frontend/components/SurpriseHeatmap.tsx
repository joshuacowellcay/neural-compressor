'use client';

import { useMemo, useState } from 'react';
import type { SurpriseSeries } from '@/lib/api';

interface Props {
  surprise: SurpriseSeries;
  maxChars?: number;
}

const COLUMNS = 64;

export function SurpriseHeatmap({ surprise, maxChars = 1024 }: Props) {
  const limit = Math.min(surprise.bits.length, maxChars);
  const bits = useMemo(() => surprise.bits.slice(0, limit), [surprise.bits, limit]);
  const bytes = useMemo(() => surprise.bytes.slice(0, limit), [surprise.bytes, limit]);

  const stats = useMemo(() => summarise(bits), [bits]);
  const [hover, setHover] = useState<number | null>(null);

  if (bits.length === 0) {
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">No surprise data to display.</p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-3 text-xs text-slate-500 dark:text-slate-400">
        <span>
          Showing first{' '}
          <span className="font-medium text-slate-700 dark:text-slate-200">{limit}</span> bytes;
          total {surprise.bits.length}.
        </span>
        <span>
          Mean: <Mono>{stats.mean.toFixed(2)}</Mono> bits/byte
        </span>
        <span>
          Median: <Mono>{stats.median.toFixed(2)}</Mono>
        </span>
        <span>
          Max: <Mono>{stats.max.toFixed(2)}</Mono>
        </span>
      </div>
      <div
        data-testid="surprise-grid"
        className="grid select-none rounded-xl border border-slate-200 bg-white p-1 font-mono text-[11px] leading-[1.05rem] dark:border-slate-700 dark:bg-slate-900"
        style={{ gridTemplateColumns: `repeat(${COLUMNS}, minmax(0, 1fr))` }}
      >
        {bits.map((b, idx) => {
          const byte = bytes[idx] ?? 0;
          const ch = renderChar(byte);
          const color = pickColor(b, stats.cap);
          const textColor = b > stats.cap * 0.55 ? 'white' : 'inherit';
          return (
            <span
              key={idx}
              onMouseEnter={() => setHover(idx)}
              onMouseLeave={() => setHover(null)}
              className="flex h-5 items-center justify-center"
              style={{ backgroundColor: color, color: textColor }}
              title={`byte ${idx}: ${b.toFixed(2)} bits  (char "${ch}")`}
            >
              {ch}
            </span>
          );
        })}
      </div>
      {hover !== null && (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Byte <Mono>{hover}</Mono>: code point <Mono>{bytes[hover]}</Mono>, rendered as{' '}
          <Mono>&quot;{renderChar(bytes[hover] ?? 0)}&quot;</Mono>, cost{' '}
          <Mono>{bits[hover].toFixed(3)}</Mono> bits.
        </p>
      )}
      <Legend cap={stats.cap} />
    </div>
  );
}

interface Stats {
  mean: number;
  median: number;
  max: number;
  cap: number;
}

function summarise(values: number[]): Stats {
  if (values.length === 0) return { mean: 0, median: 0, max: 0, cap: 8 };
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  const mean = sum / sorted.length;
  const median =
    sorted.length % 2
      ? sorted[(sorted.length - 1) / 2]
      : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  const max = sorted[sorted.length - 1];
  const p98 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))];
  return { mean, median, max, cap: Math.max(2, Math.min(8, p98)) };
}

function pickColor(bits: number, cap: number): string {
  if (!Number.isFinite(bits)) return '#1e293b';
  const t = Math.max(0, Math.min(1, bits / cap));
  // Interpolate between a pale yellow and a deep magenta-red.
  const palette: [number, number, number][] = [
    [255, 247, 230],
    [254, 209, 130],
    [240, 130, 76],
    [188, 60, 70],
    [78, 21, 64],
  ];
  const idx = t * (palette.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(palette.length - 1, lo + 1);
  const frac = idx - lo;
  const [r1, g1, b1] = palette[lo];
  const [r2, g2, b2] = palette[hi];
  const r = Math.round(r1 + (r2 - r1) * frac);
  const g = Math.round(g1 + (g2 - g1) * frac);
  const b = Math.round(b1 + (b2 - b1) * frac);
  return `rgb(${r}, ${g}, ${b})`;
}

function renderChar(byte: number): string {
  if (byte >= 32 && byte < 127) return String.fromCharCode(byte);
  if (byte === 10) return '⏎'; // return symbol
  if (byte === 9) return '→'; // tab arrow
  return '·'; // middle dot for other non-printables
}

function Mono({ children }: { children: React.ReactNode }) {
  return <span className="font-mono">{children}</span>;
}

function Legend({ cap }: { cap: number }) {
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
      <span>0</span>
      <div
        className="h-2 flex-1 rounded-full"
        style={{
          background:
            'linear-gradient(to right, rgb(255,247,230), rgb(254,209,130), rgb(240,130,76), rgb(188,60,70), rgb(78,21,64))',
        }}
      />
      <span>{cap.toFixed(1)} bits</span>
    </div>
  );
}
