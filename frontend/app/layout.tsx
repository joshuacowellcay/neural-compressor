import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Neural Compressor',
  description:
    'A lossless data compressor built from a small neural network and an integer-arithmetic entropy coder.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
