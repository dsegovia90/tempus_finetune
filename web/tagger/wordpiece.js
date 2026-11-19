// BERT's uncased WordPiece tokenizer, as Hugging Face `tokenizers` runs it (BertNormalizer with
// clean_text, handle_chinese_chars and lowercase; BertPreTokenizer; WordPiece with "##" pieces).

const CONTROL = /\p{C}/u;
const WHITESPACE = /\s/u;
const PUNCTUATION = /[\p{P}!-\/:-@\[-`{-~]/u;
const CHINESE = /[\u{4E00}-\u{9FFF}\u{3400}-\u{4DBF}\u{20000}-\u{2A6DF}\u{2A700}-\u{2B73F}\u{2B740}-\u{2B81F}\u{2B820}-\u{2CEAF}\u{F900}-\u{FAFF}\u{2F800}-\u{2FA1F}]/u;
const MAX_WORD = 100; // characters; longer words become [UNK]

export class WordPiece {
  /** `vocab`: the tokens in id order (vocab.txt). */
  constructor(vocab) {
    this.ids = new Map(vocab.map((token, id) => [token, id]));
    [this.cls, this.sep, this.unk] = ['[CLS]', '[SEP]', '[UNK]'].map((token) => this.ids.get(token));
  }

  /**
   * Token ids with [CLS] and [SEP], and for each token the [start, end) of the message text it
   * came from (0, 0 for the two special tokens), as the Hugging Face tokenizer's offsets.
   */
  encode(text) {
    const ids = [this.cls];
    const offsets = [[0, 0]];
    for (const word of words(normalize(text))) {
      for (const [id, start, end] of this.#pieces(word)) {
        ids.push(id);
        offsets.push([start, end]);
      }
    }
    ids.push(this.sep);
    offsets.push([0, 0]);
    return { ids, offsets };
  }

  /** A word's pieces as [id, start, end]: greedy longest match, "##" on all but the first. */
  #pieces(word) {
    const whole = [[this.unk, word[0][1], word.at(-1)[2]]];
    if (word.length > MAX_WORD) return whole;
    const pieces = [];
    for (let start = 0; start < word.length; ) {
      let end = word.length;
      let id;
      for (; end > start; end--) {
        id = this.ids.get((start ? '##' : '') + word.slice(start, end).map(([char]) => char).join(''));
        if (id !== undefined) break;
      }
      if (end === start) return whole;
      pieces.push([id, word[start][1], word[end - 1][2]]);
      start = end;
    }
    return pieces;
  }
}

/** The normalized characters, each as [char, start, end] with the range of the original character. */
function normalize(text) {
  const chars = [];
  let index = 0;
  for (const char of text) {
    const start = index;
    const end = (index += char.length);
    if (char === '\0' || char === '\uFFFD' || (CONTROL.test(char) && !'\t\n\r'.includes(char))) continue;
    if (WHITESPACE.test(char)) chars.push([' ', start, end]);
    else if (CHINESE.test(char)) chars.push([' ', start, start], [char, start, end], [' ', end, end]);
    // Accents stripped, then lowercased one character at a time, as Rust does: no final-sigma rule.
    else for (const plain of char.normalize('NFD').replace(/\p{Mn}/gu, '')) for (const lower of plain.toLowerCase()) chars.push([lower, start, end]);
  }
  return chars;
}

/** Split on whitespace, and every punctuation character becomes a word of its own. */
function words(chars) {
  const found = [];
  let word = [];
  const flush = () => {
    if (word.length) found.push(word);
    word = [];
  };
  for (const entry of chars) {
    if (entry[0] === ' ') flush();
    else if (PUNCTUATION.test(entry[0])) {
      flush();
      found.push([entry]);
    } else word.push(entry);
  }
  flush();
  return found;
}
