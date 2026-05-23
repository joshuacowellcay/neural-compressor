// A short public-domain passage shipped with the frontend so visitors can run
// the demo without needing to upload anything. Lifted from the opening of
// Pride and Prejudice by Jane Austen.

export const SAMPLE_TEXT = `It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.

However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?"`;

export function sampleBlob(): Blob {
  return new Blob([SAMPLE_TEXT], { type: 'text/plain' });
}
