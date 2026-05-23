import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SurpriseHeatmap } from '@/components/SurpriseHeatmap';

describe('SurpriseHeatmap', () => {
  it('renders summary statistics for the provided series', () => {
    const surprise = {
      bits: [0.5, 1.0, 4.0, 2.0],
      bytes: [104, 105, 33, 10],
    };
    render(<SurpriseHeatmap surprise={surprise} />);
    expect(screen.getByText(/Mean:/)).toBeInTheDocument();
    expect(screen.getByText(/Median:/)).toBeInTheDocument();
    expect(screen.getByText(/Max:/)).toBeInTheDocument();
    expect(screen.getByTestId('surprise-grid')).toBeInTheDocument();
  });

  it('renders one cell per byte in the series', () => {
    const surprise = {
      bits: [1, 2, 3],
      bytes: [97, 98, 99],
    };
    render(<SurpriseHeatmap surprise={surprise} />);
    const grid = screen.getByTestId('surprise-grid');
    expect(grid.children).toHaveLength(3);
    expect(grid.textContent).toContain('abc');
  });

  it('shows a friendly message when there is no data', () => {
    render(<SurpriseHeatmap surprise={{ bits: [], bytes: [] }} />);
    expect(screen.getByText(/No surprise data/i)).toBeInTheDocument();
  });
});
