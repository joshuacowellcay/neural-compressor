import type { InfoResponse } from '@/lib/api';

export function ModelInfo({ info }: { info: InfoResponse | null }) {
  if (!info) {
    return null;
  }
  if (!info.model_loaded) {
    return (
      <div className="rounded-2xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200">
        Model is not loaded on the backend.
      </div>
    );
  }
  const rows: [string, string][] = [
    ['Architecture', 'Causal Transformer'],
    ['Parameters', formatInt(info.n_parameters)],
    ['Layers', String(info.n_layers ?? '-')],
    ['Heads', String(info.n_heads ?? '-')],
    ['d_model', String(info.d_model ?? '-')],
    ['Context length', String(info.context_length ?? '-')],
    [
      'Held-out bits/byte',
      info.eval_bits_per_byte !== null ? info.eval_bits_per_byte.toFixed(3) : '-',
    ],
    ['Fingerprint', info.fingerprint_hex ?? '-'],
  ];
  return (
    <aside
      data-testid="model-info"
      className="rounded-2xl border border-slate-200 bg-white p-5 dark:border-slate-700 dark:bg-slate-900"
    >
      <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Model</h3>
      <dl className="mt-3 grid grid-cols-1 gap-y-2 text-sm">
        {rows.map(([label, value]) => (
          <div key={label} className="flex items-baseline justify-between gap-3">
            <dt className="text-slate-500 dark:text-slate-400">{label}</dt>
            <dd className="font-mono text-slate-800 dark:text-slate-100">{value}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}

function formatInt(value: number | null): string {
  if (value === null || value === undefined) return '-';
  return value.toLocaleString();
}
