// Mirrors vocab.py exactly: 30 tokens — 4 specials + a-z
(function () {
  var SPECIAL = ['<PAD>', '<BOS>', '<SEP>', '<EOS>'];
  var CHARS   = 'abcdefghijklmnopqrstuvwxyz'.split('');
  var ITOS    = SPECIAL.concat(CHARS);           // length 30
  var STOI    = {};
  ITOS.forEach(function (t, i) { STOI[t] = i; });

  window.Vocab = {
    ITOS: ITOS,
    STOI: STOI,
    VOCAB_SIZE: 30,
    PAD_ID: 0,
    BOS_ID: 1,
    SEP_ID: 2,
    EOS_ID: 3,

    encode: function (tokens) {
      return tokens.map(function (t) { return STOI[t]; });
    },

    decode: function (ids) {
      return ids.map(function (i) { return ITOS[i]; });
    },

    // tokenizeWord('hello') → ['h','e','l','l','o']
    tokenizeWord: function (word) {
      return word.toLowerCase().split('').filter(function (c) {
        return c in STOI;
      });
    },

    isSpecial: function (token) {
      return SPECIAL.indexOf(token) !== -1;
    },

    // Hue 0-270 for letter a(0)..z(25) — used for coloring token boxes
    letterHue: function (token) {
      var idx = CHARS.indexOf(token);
      return idx >= 0 ? Math.round((idx / 25) * 270) : 0;
    },
  };
})();
