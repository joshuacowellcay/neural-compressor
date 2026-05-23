# Corpus

## Pride and Prejudice, by Jane Austen

`pride_and_prejudice.txt` is the full text of *Pride and Prejudice* by Jane Austen
(first published in 1813, in the public domain worldwide).

* Source: Project Gutenberg eBook #1342 (https://www.gutenberg.org/ebooks/1342).
* Licence: public domain (the work itself; Project Gutenberg's optional
  trademark licence applies only to the Project Gutenberg name and header text,
  which have been stripped from this copy).

## Processing

The file in this repository has been lightly normalised from the Project
Gutenberg release so that it is ASCII-only and easier to tokenise:

* Header and footer added by Project Gutenberg are removed.
* Unicode quotation marks (`U+2018`, `U+2019`, `U+201C`, `U+201D`) are
  converted to the ASCII equivalents `'` and `"`.
* The dash characters (`U+2014`, `U+2013`) are converted to `--` and `-`.
* The ligature `oe` (`U+0153`) is expanded to `oe`; the middle dot
  (`U+00B7`) is replaced with `.`.
* Line endings are LF; runs of three or more blank lines are collapsed to two.

The resulting file is treated as a stream of bytes by the byte-level
tokeniser. The first ~80% of the file (up to the start of Chapter LI) is used
for training and the remainder is held out for evaluation and benchmarking.
The split is computed at load time by `ncomp.training.data.load_corpus`; the
corpus file on disk is the unsplit normalised text.
