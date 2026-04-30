// Pure JS port of compute_reward() from mini_gpt_reverse_skip_b_rl.py
// No model needed — deterministic given (inputSeq, predTokens).
(function () {

  // targetSkipB(['t','e','s','b','t']) → ['t','s','e','t']
  function targetSkipB(seq) {
    return seq.slice().reverse().filter(function (c) { return c !== 'b'; });
  }

  function countTokens(tokens) {
    var counts = {};
    tokens.forEach(function (t) { counts[t] = (counts[t] || 0) + 1; });
    return counts;
  }

  function intersectCounts(a, b) {
    var result = {};
    Object.keys(a).forEach(function (k) {
      if (k in b) result[k] = Math.min(a[k], b[k]);
    });
    return result;
  }

  function sumValues(obj) {
    return Object.values(obj).reduce(function (s, v) { return s + v; }, 0);
  }

  /**
   * computeReward(inputSeq, predTokens)
   *
   * inputSeq  — array of single-char strings, e.g. ['b','o','m','b']
   * predTokens — array of token strings, may include '<EOS>', '<PAD>', etc.
   *
   * Returns { terms: {exact_match, positional, coverage, b_penalty,
   *                    length_mismatch, no_eos, pad_penalty, special_leak},
   *           total: number,
   *           target: string[] (skip-b target without <EOS>),
   *           targetWithEos: string[] }
   */
  function computeReward(inputSeq, predTokens) {
    var targetBase = targetSkipB(inputSeq);
    var target     = targetBase.concat(['<EOS>']);

    // Trim pred to up-to-and-including the first <EOS>
    var pred = predTokens.slice();
    var eosIdx = pred.indexOf('<EOS>');
    if (eosIdx !== -1) pred = pred.slice(0, eosIdx + 1);

    var reward = 0;
    var terms  = {};

    // 1. Exact match
    var exactMatch = (JSON.stringify(pred) === JSON.stringify(target));
    terms.exact_match = exactMatch ? 5.0 : 0;
    reward += terms.exact_match;

    // 2. Positional match (per-position, normalised by target length)
    var minLen = Math.min(pred.length, target.length);
    var posMat = 0;
    for (var i = 0; i < minLen; i++) {
      if (pred[i] === target[i]) posMat++;
    }
    terms.positional = 0.2 * posMat / Math.max(target.length, 1);
    reward += terms.positional;

    // 3. Character coverage — multiset intersection (exclude <EOS>)
    var predChars   = countTokens(pred.filter(function (t) { return t !== '<EOS>'; }));
    var targetChars = countTokens(target.filter(function (t) { return t !== '<EOS>'; }));
    var inter       = intersectCounts(predChars, targetChars);
    var coverage    = sumValues(inter);
    terms.coverage  = 0.1 * coverage / Math.max(target.length - 1, 1);
    reward += terms.coverage;

    // 4. b count penalty
    var numB = pred.filter(function (t) { return t === 'b'; }).length;
    terms.b_penalty = -1.0 * numB;
    reward += terms.b_penalty;

    // 5. Length mismatch
    terms.length_mismatch = -0.1 * Math.abs(pred.length - target.length);
    reward += terms.length_mismatch;

    // 6. No EOS in generated output
    terms.no_eos = predTokens.indexOf('<EOS>') !== -1 ? 0 : -0.5;
    reward += terms.no_eos;

    // 7. PAD tokens in output
    var numPad = pred.filter(function (t) { return t === '<PAD>'; }).length;
    terms.pad_penalty = -2.0 * numPad;
    reward += terms.pad_penalty;

    // 8. Special token leak (<SEP> or <BOS> in output)
    var numSep = pred.filter(function (t) { return t === '<SEP>'; }).length;
    var numBos = pred.filter(function (t) { return t === '<BOS>'; }).length;
    terms.special_leak = -2.0 * (numSep + numBos);
    reward += terms.special_leak;

    return {
      terms:         terms,
      total:         reward,
      target:        targetBase,
      targetWithEos: target,
      pred:          pred,
      exactMatch:    exactMatch,
    };
  }

  /**
   * parsePredString(str, advancedMode)
   *
   * In basic mode:  "mo"  → ['m','o','<EOS>']
   * In advanced mode: "mo<EOS>" → ['m','o','<EOS>']
   *                   "<PAD>mo" → ['<PAD>','m','o']
   */
  function parsePredString(str, advancedMode) {
    if (!advancedMode) {
      // basic: just lowercase letters, auto-append <EOS>
      var chars = str.toLowerCase().split('').filter(function (c) {
        return /[a-z]/.test(c);
      });
      return chars.concat(['<EOS>']);
    }
    // advanced: split on special token markers
    var tokens = [];
    var remaining = str;
    var specialRe = /^(<PAD>|<BOS>|<SEP>|<EOS>)/;
    while (remaining.length > 0) {
      var m = remaining.match(specialRe);
      if (m) {
        tokens.push(m[1]);
        remaining = remaining.slice(m[1].length);
      } else {
        var ch = remaining[0].toLowerCase();
        if (/[a-z]/.test(ch)) tokens.push(ch);
        remaining = remaining.slice(1);
      }
    }
    return tokens;
  }

  window.Reward = {
    targetSkipB:    targetSkipB,
    computeReward:  computeReward,
    parsePredString: parsePredString,
  };
})();
