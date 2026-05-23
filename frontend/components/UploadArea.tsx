'use client';

import { useRef, useState, type DragEvent } from 'react';

interface Props {
  disabled?: boolean;
  maxBytes: number;
  onFile: (file: File | Blob) => void;
  onUseSample: () => void;
}

export function UploadArea({ disabled, maxBytes, onFile, onUseSample }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isOver, setIsOver] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);

  function handleFiles(files: FileList | null) {
    setWarning(null);
    const file = files?.[0];
    if (!file) return;
    if (file.size > maxBytes) {
      setWarning(`File is ${formatBytes(file.size)}; the demo cap is ${formatBytes(maxBytes)}.`);
      return;
    }
    onFile(file);
  }

  function onDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsOver(true);
  }

  function onDragLeave() {
    setIsOver(false);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsOver(false);
    if (disabled) return;
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div className="flex flex-col gap-3">
      <div
        data-testid="upload-area"
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`flex flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
          isOver
            ? 'border-ncomp-500 bg-ncomp-50/60 dark:bg-ncomp-700/20'
            : 'border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900'
        } ${disabled ? 'pointer-events-none opacity-60' : ''}`}
      >
        <p className="text-base font-medium text-slate-800 dark:text-slate-100">
          Drop a small text file here
        </p>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          or pick one to compare against gzip, bzip2, and xz
        </p>
        <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
          <button
            type="button"
            className="rounded-full bg-ncomp-500 px-4 py-2 text-sm font-medium text-white hover:bg-ncomp-600 disabled:opacity-60"
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
          >
            Choose file
          </button>
          <button
            type="button"
            className="rounded-full border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-60 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-800"
            onClick={onUseSample}
            disabled={disabled}
          >
            Use sample text
          </button>
        </div>
        <p className="mt-1 text-xs text-slate-400">Demo cap: {formatBytes(maxBytes)}</p>
        <input
          ref={inputRef}
          data-testid="file-input"
          type="file"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
      {warning && (
        <p role="alert" className="text-sm text-amber-700 dark:text-amber-300">
          {warning}
        </p>
      )}
    </div>
  );
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
