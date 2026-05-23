import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ResultsCard } from '@/components/ResultsCard';
import type { ToolResult } from '@/lib/api';

function makeResult(name: string, compressed: number): ToolResult {
  return {
    name,
    compressed_bytes: compressed,
    ratio: compressed / 1000,
    bits_per_byte: (compressed * 8) / 1000,
    compress_seconds: 0.01,
    decompress_seconds: 0.01,
    ok: true,
  };
}

describe('ResultsCard', () => {
  const ncomp = makeResult('ncomp', 250);
  const gzip = makeResult('gzip -9', 350);

  it('renders the headline numbers', () => {
    render(<ResultsCard ncomp={ncomp} gzip={gzip} inputBytes={1000} />);
    expect(screen.getByText('Compressed size')).toBeInTheDocument();
    expect(screen.getByText('250 B')).toBeInTheDocument();
    expect(screen.getByText('Bits per byte')).toBeInTheDocument();
    expect(screen.getByText('2.000')).toBeInTheDocument();
  });

  it('shows the percent saved relative to gzip', () => {
    render(<ResultsCard ncomp={ncomp} gzip={gzip} inputBytes={1000} />);
    // 1 - 250 / 350 = 0.2857 = 28.6% smaller
    expect(screen.getByText('28.6% smaller')).toBeInTheDocument();
  });

  it('handles a missing gzip baseline', () => {
    render(<ResultsCard ncomp={ncomp} gzip={undefined} inputBytes={1000} />);
    expect(screen.getByText('gzip unavailable')).toBeInTheDocument();
  });
});
