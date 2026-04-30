(function () {
  var R = window.Reward;

  var TERM_LABELS = {
    exact_match:    'Exact match (+5.0)',
    positional:     'Positional match (+0.2×)',
    coverage:       'Character coverage (+0.1×)',
    b_penalty:      'ตัว b penalty (−1.0×count)',
    length_mismatch:'ความยาวต่างจาก target (−0.1×)',
    no_eos:         'ไม่มี <EOS> (−0.5)',
    pad_penalty:    '<PAD> ใน output (−2.0×count)',
    special_leak:   '<SEP>/<BOS> ใน output (−2.0×)',
  };

  var TERM_ORDER = [
    'exact_match', 'positional', 'coverage',
    'b_penalty', 'length_mismatch', 'no_eos', 'pad_penalty', 'special_leak',
  ];

  // Max absolute value across all terms (for bar scaling)
  var BAR_MAX = 5.0;

  function renderBars(result) {
    var cont = document.getElementById('reward-bars');
    cont.innerHTML = '';

    TERM_ORDER.forEach(function (key) {
      var val   = result.terms[key] || 0;
      if (val === 0 && key !== 'exact_match') {
        // still show row, just as zero
      }
      var row   = document.createElement('div');
      row.className = 'reward-bar-row';
      var frac  = Math.min(Math.abs(val) / BAR_MAX, 1) * 50;  // max half
      var fillHtml = val >= 0
        ? '<span class="reward-bar-fill-pos anim-grow" style="width:' + frac + '%"></span>'
        : '<span class="reward-bar-fill-neg anim-grow" style="width:' + frac + '%"></span>';

      row.innerHTML =
        '<span class="reward-bar-label">' + TERM_LABELS[key] + '</span>' +
        '<span class="reward-bar-center">' +
          '<span class="reward-bar-track-pos"></span>' +
          '<span class="reward-bar-track-neg"></span>' +
          '<span class="reward-bar-zero"></span>' +
          fillHtml +
        '</span>' +
        '<span class="reward-bar-val ' + (val >= 0 ? 'text-green-700' : 'text-red-700') + '">' +
          (val >= 0 ? '+' : '') + val.toFixed(2) + '</span>';
      cont.appendChild(row);
    });

    var total    = result.total;
    var totalBox = document.getElementById('reward-total-box');
    var totalEl  = document.getElementById('reward-total');
    totalEl.textContent = (total >= 0 ? '+' : '') + total.toFixed(3);
    totalEl.className   = 'text-2xl font-mono font-bold ' +
      (total >= 4 ? 'text-green-700' : total >= 0 ? 'text-amber-700' : 'text-red-700');
    totalBox.style.background = total >= 4 ? '#f0fdf4' : total >= 0 ? '#fffbeb' : '#fff1f2';
  }

  function updateTargetDisplay(inputStr) {
    var seq    = inputStr.toLowerCase().split('').filter(function (c) { return /[a-z]/.test(c); });
    var target = R.targetSkipB(seq);
    document.getElementById('reward-target-display').textContent =
      'target (skip-b): "' + target.join('') + '"' +
      (target.length === 0 ? ' (ว่าง — input ล้วนเป็น b)' : '');
  }

  function calculate() {
    var inputStr  = document.getElementById('reward-input').value.trim();
    var predStr   = document.getElementById('reward-pred').value.trim();
    var advanced  = document.getElementById('reward-advanced').checked;

    var inputSeq  = inputStr.toLowerCase().split('').filter(function (c) { return /[a-z]/.test(c); });
    var predTokens = R.parsePredString(predStr, advanced);

    updateTargetDisplay(inputStr);
    var result = R.computeReward(inputSeq, predTokens);
    renderBars(result);
  }

  function setPreset(inp, pred) {
    document.getElementById('reward-input').value = inp;
    document.getElementById('reward-pred').value  = pred;
    document.getElementById('reward-advanced').checked = false;
    calculate();
  }

  function init() {
    document.getElementById('reward-input').addEventListener('input', calculate);
    document.getElementById('reward-pred').addEventListener('input',  calculate);
    document.getElementById('reward-advanced').addEventListener('change', calculate);
    document.getElementById('reward-reset').addEventListener('click', function () {
      setPreset('bomb', 'bmob');
    });

    // Preset buttons
    document.querySelectorAll('#section-reward .btn-preset[data-ri]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        setPreset(this.dataset.ri, this.dataset.rp);
      });
    });

    setPreset('bomb', 'bmob');
  }

  window.SectionReward = { init: init };
})();
