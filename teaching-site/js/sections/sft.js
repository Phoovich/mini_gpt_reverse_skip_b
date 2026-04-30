(function () {
  // Example: hello → reversed hello
  var SEQ    = ['h', 'e', 'l', 'l', 'o'];
  var REV    = ['o', 'l', 'l', 'e', 'h'];
  var FULL   = ['<BOS>'].concat(SEQ).concat(['<SEP>']).concat(REV).concat(['<EOS>']);
  // Stepper shows positions 0..FULL.length-2 (model predicts FULL[1..])
  var INPUT  = FULL.slice(0, FULL.length - 1);
  var TARGET = FULL.slice(1);

  var stepIdx   = 0;
  var autoTimer = null;

  function isSpecial(t) { return ['<BOS>', '<SEP>', '<EOS>', '<PAD>'].indexOf(t) !== -1; }

  function renderStepper() {
    var seqEl  = document.getElementById('sft-stepper-seq');
    var predEl = document.getElementById('sft-stepper-predict');
    seqEl.innerHTML = '';

    INPUT.forEach(function (tok, i) {
      var span = document.createElement('span');
      span.className = 'stepper-token ' +
        (i < stepIdx ? 'past' : i === stepIdx ? 'current' : 'future');
      span.textContent = tok;
      seqEl.appendChild(span);
    });

    var nextTok = TARGET[stepIdx];
    predEl.innerHTML =
      'โมเดลเห็น: <strong class="font-mono">' + INPUT.slice(0, stepIdx + 1).join(' ') + '</strong>' +
      ' &nbsp;→&nbsp; ต้องทำนาย: <strong class="font-mono text-amber-700">' + nextTok + '</strong>' +
      ' (ID=' + (window.Vocab ? window.Vocab.STOI[nextTok] : '?') + ')';
  }

  function stepForward() {
    if (stepIdx < INPUT.length - 1) { stepIdx++; renderStepper(); }
    else stopAutoplay();
  }

  function stepBack() {
    if (stepIdx > 0) { stepIdx--; renderStepper(); }
  }

  function stopAutoplay() {
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
    document.getElementById('sft-autoplay').textContent = '▶▶ Auto-play';
  }

  function toggleAutoplay() {
    if (autoTimer) {
      stopAutoplay();
    } else {
      document.getElementById('sft-autoplay').textContent = '⏸ Pause';
      autoTimer = setInterval(function () {
        if (stepIdx >= INPUT.length - 1) { stopAutoplay(); return; }
        stepForward();
      }, 900);
    }
  }

  // ── CE loss visualizer ────────────────────────────────────────────────────
  function updateCE(pCorrect) {
    var canvas  = document.getElementById('ce-canvas');
    var ctx     = canvas.getContext('2d');
    var W       = canvas.width;
    var H       = canvas.height;
    var nTokens = 30;
    var pOther  = (1 - pCorrect) / (nTokens - 1);

    ctx.clearRect(0, 0, W, H);
    var barW = W / nTokens;
    for (var i = 0; i < nTokens; i++) {
      var p = (i === 0) ? pCorrect : pOther;
      var h = Math.max(2, p * (H - 4));
      ctx.fillStyle = (i === 0) ? '#16a34a' : '#93c5fd';
      ctx.fillRect(i * barW + 1, H - h, barW - 2, h);
    }
    // label correct token
    ctx.fillStyle = '#15803d';
    ctx.font = '9px monospace';
    ctx.fillText('correct', 2, H - 2);

    var loss = -Math.log(Math.max(pCorrect, 1e-9));
    document.getElementById('ce-formula').innerHTML =
      'loss = −log(' + pCorrect.toFixed(2) + ') = <span class="' +
      (loss < 0.5 ? 'text-green-700' : loss < 2 ? 'text-amber-700' : 'text-red-700') +
      '">' + loss.toFixed(3) + '</span>';

    document.getElementById('ce-p-val').textContent = pCorrect.toFixed(2);
    var verdict = document.getElementById('ce-verdict');
    if (pCorrect >= 0.8) {
      verdict.textContent = 'โมเดลมั่นใจมาก → loss ต่ำ → gradient update เล็กน้อย';
    } else if (pCorrect >= 0.4) {
      verdict.textContent = 'โมเดลไม่แน่ใจ → loss ปานกลาง → gradient update ปานกลาง';
    } else {
      verdict.textContent = 'โมเดลผิดมาก → loss สูง → gradient update แรง';
    }
  }

  function init() {
    renderStepper();

    document.getElementById('sft-next').addEventListener('click', stepForward);
    document.getElementById('sft-prev').addEventListener('click', stepBack);
    document.getElementById('sft-autoplay').addEventListener('click', toggleAutoplay);
    document.getElementById('sft-reset').addEventListener('click', function () {
      stopAutoplay(); stepIdx = 0; renderStepper();
    });

    var slider = document.getElementById('ce-slider');
    slider.addEventListener('input', function () {
      updateCE(parseInt(this.value) / 100);
    });
    updateCE(0.80);
  }

  window.SectionSFT = { init: init };
})();
