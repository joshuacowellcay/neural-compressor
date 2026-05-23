'use client';

import type { ToolResult } from '@/lib/api';
import { formatBytes } from '@/lib/api';

interface Props {
  ncomp: ToolResult;
  gzip: ToolResult | undefined;
  inputBytes: number;
}

export function ResultsCard({ ncomp, gzip, inputBytes }: Props) {
  const ratioPct = ((1 - ncomp.ratio) * 100).toFixed(1);
  const versusGzip = gzip ? 1 - ncomp.compressed_bytes / gzip.compressed_bytes : null;

  return (
    <section className="grid grid-cols-1 gap-3 rounded-2xl border border-slate-200 bg-white p-5 sm:grid-cols-3 dark:border-slate-700 dark:bg-slate-900">
      <Stat label="Compressed size" value={formatBytes(ncomp.compressed_bytes)}>
        {`from ${formatBytes(inputBytes)} (${ratioPct}% smaller)`}
      </Stat>
      <Stat label="Bits per byte" value={ncomp.bits_per_byte.toFixed(3)}>
        {`raw = 8.000; lower is better`}
      </Stat>
      <Stat
        label="Versus gzip -9"
        value={
          versusGzip !== null ? `${(versusGzip * 100).toFixed(1)}% smaller` : 'gzip unavailable'
        }
      >
        {gzip ? `gzip produced ${formatBytes(gzip.compressed_bytes)}` : null}
      </Stat>
    </section>
  );
}

function Stat({
  label,
  value,
  children,
}: {
  label: string;
  value: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </span>
      <span className="text-2xl font-semibold tabular-nums text-slate-900 dark:text-slate-100">
        {value}
      </span>
      {children && <span className="text-xs text-slate-500 dark:text-slate-400">{children}</span>}
    </div>
  );
}
