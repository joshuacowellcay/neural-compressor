# Handoff notes

A plain-English tour of the project, the design decisions that mattered, and what to do
next.

## What the components actually do

* **`src/ncomp/coder/arithmetic.py`** is a 32-bit integer arithmetic coder. It uses only
  Python integers and the classic E1/E2/E3 (underflow) scaling, so the encoder and
  decoder narrow the interval in exactly the same way. The `BitWriter` and `BitReader`
  in the same folder buffer individual bits into whole bytes; the reader returns zero
  past end-of-stream so the decoder can drain the last few bits cleanly.
* **`src/ncomp/model/`** is the next-byte prediction model. `transformer.py` is a small
  causal pre-norm Transformer (default: four layers, four heads, `d_model=128`,
  `context_length=64`, ~430k parameters). `probabilities.py` deterministically converts
  the float probability vector the model produces into integer counts that sum to
  `prob_total` with at least `min_count` per symbol; that integer vector is what the
  coder operates on.
* **`src/ncomp/pipeline/compressor.py`** is the glue. It maintains the context window,
  runs the model, calls the quantiser, hands the resulting CDF to the arithmetic coder,
  and prepends the 21-byte file header (`pipeline/file_format.py`). The header includes
  an 8-byte SHA-256-derived fingerprint of the model so the decompressor refuses to
  decode a payload from a different model.
* **`src/ncomp/training/`** loads the corpus, splits it at a chapter boundary near 80%,
  samples random windows, and runs the AdamW + warmup-cosine training loop. The best
  held-out checkpoint by bits-per-byte is the one saved.
* **`src/ncomp/benchmark/runner.py`** wraps Python's stdlib `gzip`, `bz2`, and `lzma`
  plus the neural pipeline, returns a comparable `Result` per tool, and asserts every
  round-trip is exact.
* **`backend/`** is the FastAPI service used by the demo. It loads the model once in
  the lifespan handler and serialises requests under a lock (PyTorch models are not
  thread-safe for concurrent inference).
* **`frontend/`** is the Next.js demo: drop zone, comparison chart, and per-byte surprise
  heatmap.

## Design decisions and why

1. **Integer arithmetic in the coding path.** The encoder and decoder narrow a shared
   interval one symbol at a time. With floating point, tiny rounding differences between
   the two sides would eventually push a code point across a boundary and silently
   corrupt the output. Using only Python integers makes every interval split
   reproducible bit-for-bit, on any machine, for any input. This is the most important
   correctness decision in the project.
2. **Deterministic probability quantisation.** The model itself works in floats, so the
   quantiser in `model/probabilities.py` converts that float vector into integer counts
   in a way that is stable: allocate `min_count` per symbol, floor the scaled
   probabilities, then distribute the residual to the largest fractional parts with
   tie-breaking by symbol index. Encoder and decoder feed the same float vector through
   the same routine and get byte-identical CDFs. Without this step the model would
   compute slightly different floats on the two sides and the coder would diverge.
3. **Static, not adaptive.** The model never updates during compression or
   decompression. It learned its statistics during training and uses the same weights
   every step. An adaptive coder (online-updating frequencies) would compress slightly
   better on out-of-distribution text, but it would require the encoder and decoder to
   stay perfectly synchronised on which updates were applied to which symbol and would
   add a new failure mode. Static is simpler, easier to test, and easier to reason
   about; it is the right pick for a portfolio piece, and it is what makes the model
   fingerprint in the file header sufficient on its own.
4. **Byte-level tokenisation.** No learned tokeniser, no subwords, no Unicode handling.
   The vocabulary is the 256 possible byte values, full stop. That means the compressor
   can handle arbitrary binary input (it cannot represent something out of vocabulary
   because there is nothing out of vocabulary), and that the round-trip cannot lose
   information at the tokenisation boundary. The downside is that the model has to learn
   intra-word and intra-character structure from scratch; that is acceptable here and
   the bits-per-byte numbers show it works.
5. **Sliding context window.** The model is trained at a fixed context length; for
   inputs longer than that, both encoder and decoder simply use the most recent
   `context_length` bytes as context. The contexts on both sides are guaranteed to match
   byte-for-byte (the decoder reconstructs each byte before predicting the next), so the
   probabilities match too.

## Recording the demo GIF

1. Start the backend: `python scripts/serve.py`.
2. Start the frontend in another shell: `cd frontend && npm run dev`.
3. Pick a screen recorder (macOS: `Cmd+Shift+5` then "Record Selected Portion";
   on Linux, `peek` or `byzanz-record` work).
4. Record the page at `http://localhost:3000`, choose the bundled sample, wait for
   the per-byte heatmap to render, and stop the recording when it does.
5. Convert to a roughly 1200x800 GIF (`ffmpeg -i in.mov -vf "fps=8,scale=1200:-1:flags=lanczos"
   assets/demo.gif`) and commit it. The README already references that path.

A static screenshot in the same folder is also fine if a GIF is too heavy.

## Deployment

The split is intentional: the frontend is static and trivial to host; the backend is
the heavy half because it carries the PyTorch checkpoint.

* **Frontend (Vercel).** From `frontend/`, run `vercel` once to link the project, then
  set `NEXT_PUBLIC_API_URL` to the public URL of the deployed backend. The `rewrites`
  block in `next.config.mjs` will forward `/api/*` to the backend.
* **Backend (Render / Railway / Fly).** Use the included `docker-compose.yml` as
  reference. The Dockerfile is one line: pick a slim Python base, install
  `requirements.txt`, copy `src/`, `backend/`, `models/`, and `corpus/`, expose 8000,
  run `uvicorn backend.app:create_app --factory --host 0.0.0.0 --port 8000`. Render
  reads that directly from `docker-compose.yml`; Fly needs an equivalent `fly.toml`.
  The checkpoint is 2.3 MB so cold starts stay small; budget ~3 seconds for the model
  to load on first request.
* **Cost.** Vercel's hobby tier covers the frontend at zero cost. A 256 MB free-tier
  worker is enough for the backend; CPU is the bottleneck, not memory.

## Publishing this folder as its own repo

```bash
cd neural-compressor
git remote -v             # confirm no remote yet
gh repo create joshuacowell/neural-compressor --public --source=. --remote=origin --push
# or the manual equivalent:
# gh repo create joshuacowell/neural-compressor --public
# git remote add origin git@github.com:joshuacowell/neural-compressor.git
# git push -u origin main
```

CI runs the Python and frontend test suites on every push; the badge in the README
turns green once the first action completes.

## Interview Q&A

**Q. Why does prediction equal compression?**
Shannon: the optimal code length for a symbol of probability `p` is `-log2(p)` bits.
Arithmetic coding achieves that bound to within a fraction of a bit, whatever
probability model you supply. So the better the model, the shorter the encoding. The
encoded length, summed over the file, is essentially the model's cross-entropy in bits.

**Q. How does arithmetic coding work in two sentences?**
It keeps a current interval (start at `[0, 1)`). Each symbol narrows the interval to
the slot proportional to that symbol's probability; emitting the final interval (or any
value inside it) at sufficient precision encodes the entire message.

**Q. How is the round-trip guaranteed lossless?**
Three things. The coder uses only integer arithmetic (no float drift). The probability
quantiser is deterministic, so the encoder and decoder compute the same integer CDF
from the same float probabilities. The model is fingerprinted in the file header and
the decoder rejects payloads with a different fingerprint, so you cannot accidentally
decode with a model that would produce different probabilities. The test suite
round-trips empty input, single bytes, long constant runs, arbitrary binary bytes, and
random streams.

**Q. Why is it so slow compared to gzip?**
gzip is a hand-tuned C loop doing dictionary matching; we run a Transformer forward
pass per byte in PyTorch. The asymmetry is fundamental to the approach (the model is
the compressor), not a Python-vs-C issue alone. The decoder is even harder to
parallelise because each byte depends on the previous one. KV caching plus batched
encoding would cut wall time by roughly the context length; the spec for this project
prioritises correctness and clarity over throughput.

**Q. What would you change if you had another week?**
1. KV-cached inference for both encoder and decoder.
2. A larger, multi-domain corpus (Wikipedia + a code corpus + a small log dump) and a
   bigger model; aim for sub-1.5 bits/byte on natural text and parity with gzip on
   non-text.
3. Quantised weights so cross-machine bit-exact decoding works without pinning a CPU
   kernel.
4. A streaming API endpoint so the frontend can show the bitstream growing as the
   model encodes.
