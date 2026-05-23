import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { UploadArea } from '@/components/UploadArea';

describe('UploadArea', () => {
  it('invokes the sample handler when the user clicks "Use sample text"', async () => {
    const onFile = vi.fn();
    const onUseSample = vi.fn();
    render(<UploadArea maxBytes={4096} onFile={onFile} onUseSample={onUseSample} />);

    await userEvent.click(screen.getByRole('button', { name: /Use sample text/i }));
    expect(onUseSample).toHaveBeenCalledTimes(1);
    expect(onFile).not.toHaveBeenCalled();
  });

  it('passes uploaded files smaller than the cap to onFile', async () => {
    const onFile = vi.fn();
    const onUseSample = vi.fn();
    render(<UploadArea maxBytes={4096} onFile={onFile} onUseSample={onUseSample} />);

    const file = new File(['hello'], 'hi.txt', { type: 'text/plain' });
    const input = screen.getByTestId('file-input') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [file] } });
    expect(onFile).toHaveBeenCalledTimes(1);
    expect(onFile.mock.calls[0][0]).toBe(file);
  });

  it('warns when a file is over the cap and does not invoke onFile', () => {
    const onFile = vi.fn();
    const onUseSample = vi.fn();
    render(<UploadArea maxBytes={4} onFile={onFile} onUseSample={onUseSample} />);

    const big = new File(['too-big-payload'], 'big.txt', { type: 'text/plain' });
    fireEvent.change(screen.getByTestId('file-input'), { target: { files: [big] } });
    expect(onFile).not.toHaveBeenCalled();
    expect(screen.getByRole('alert')).toHaveTextContent(/demo cap is/i);
  });
});
