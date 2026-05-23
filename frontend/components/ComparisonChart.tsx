'use client';

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { ToolResult } from '@/lib/api';
import { NCOMP_TOOL_NAME } from '@/lib/api';

const NCOMP_COLOR = '#0a9b6e';
const OTHER_COLOR = '#94a3b8';

interface Props {
  results: ToolResult[];
}

export function ComparisonChart({ results }: Props) {
  const data = results.map((r) => ({
    name: r.name,
    bpb: Number(r.bits_per_byte.toFixed(3)),
    compress_ms: Number((r.compress_seconds * 1000).toFixed(1)),
  }));

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <ChartCard title="Bits per byte (lower is better)" hint="raw bytes = 8.00 bits/byte">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#94a3b8" />
            <YAxis tick={{ fontSize: 12 }} stroke="#94a3b8" domain={[0, 8]} />
            <Tooltip
              cursor={{ fill: 'rgba(148, 163, 184, 0.1)' }}
              formatter={(value: number) => `${value.toFixed(3)} bits/byte`}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="bpb" name="bits/byte">
              {data.map((d) => (
                <Cell key={d.name} fill={d.name === NCOMP_TOOL_NAME ? NCOMP_COLOR : OTHER_COLOR} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard
        title="Compression time (log scale)"
        hint="standard tools run in milliseconds; ncomp runs a forward pass per byte"
      >
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#94a3b8" />
            <YAxis
              tick={{ fontSize: 12 }}
              stroke="#94a3b8"
              scale="log"
              domain={[0.1, 'auto']}
              allowDataOverflow
              tickFormatter={(v: number) =>
                v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${v.toFixed(0)}ms`
              }
            />
            <Tooltip
              cursor={{ fill: 'rgba(148, 163, 184, 0.1)' }}
              formatter={(value: number) =>
                value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(1)} ms`
              }
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="compress_ms" name="compress time">
              {data.map((d) => (
                <Cell key={d.name} fill={d.name === NCOMP_TOOL_NAME ? NCOMP_COLOR : OTHER_COLOR} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

function ChartCard({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2 rounded-2xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
      <div>
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</h3>
        {hint && <p className="text-xs text-slate-500 dark:text-slate-400">{hint}</p>}
      </div>
      {children}
    </div>
  );
}
