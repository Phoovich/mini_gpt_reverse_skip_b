(function () {
  // ── Decoder animation ────────────────────────────────────────────────────
  var decData    = null;
  var decStep    = -1;
  var decTimer   = null;
  var decPlaying = false;

  function decTokenEl(tok, isB, isEos, isCurrent) {
    var el = document.createElement('span');
    el.className = 'decoder-token';
    if (isCurrent) el.classList.add('current-step');
    else if (isB)  el.classList.add('b-token');
    else if (isEos) el.classList.add('eos');
    el.textContent = tok;
    return el;
  }

  function renderProbBars(containerId, top5, model) {
    var cont = document.getElementById(containerId);
    cont.innerHTML = '';
    if (!top5 || !top5.length) return;
    var maxP = top5[0][1] || 0.01;
    top5.forEach(function (pair) {
      var tok = pair[0], p = pair[1];
      var row = document.createElement('div');
      row.className = 'prob-bar-row';
      row.innerHTML =
        '<span class="prob-bar-label">' + tok + '</span>' +
        '<span class="prob-bar-track"><span class="prob-bar-fill ' +
        (model === 'sft' ? 'sft-fill' : '') +
        ' anim-grow" style="width:' + Math.round((p / maxP) * 100) + '%"></span></span>' +
        '<span class="prob-bar-val">' + (p * 100).toFixed(1) + '%</span>';
      cont.appendChild(row);
    });
  }

  function renderDecStep(stepIdx) {
    if (!decData) return;
    var sftSteps = decData.sft;
    var rlSteps  = decData.rl;

    // SFT tokens
    var sftCont = document.getElementById('dec-sft-tokens');
    sftCont.innerHTML = '';
    for (var i = 0; i <= stepIdx && i < sftSteps.length; i++) {
      var tok = sftSteps[i].token;
      var isCur = (i === stepIdx);
      sftCont.appendChild(decTokenEl(tok, tok === 'b', tok === '<EOS>', isCur));
    }

    // RL tokens
    var rlCont = document.getElementById('dec-rl-tokens');
    rlCont.innerHTML = '';
    for (var j = 0; j <= stepIdx && j < rlSteps.length; j++) {
      var rtok = rlSteps[j].token;
      var isRCur = (j === stepIdx);
      rlCont.appendChild(decTokenEl(rtok, rtok === 'b', rtok === '<EOS>', isRCur));
    }

    // Prob bars for current step
    var sStep = sftSteps[Math.min(stepIdx, sftSteps.length - 1)];
    var rStep = rlSteps[Math.min(stepIdx, rlSteps.length - 1)];
    renderProbBars('dec-sft-bars', sStep ? sStep.top5 : [], 'sft');
    renderProbBars('dec-rl-bars',  rStep ? rStep.top5  : [], 'rl');

    var maxSteps = Math.max(sftSteps.length, rlSteps.length);
    var status = document.getElementById('dec-status');
    status.textContent = 'step ' + (stepIdx + 1) + ' / ' + maxSteps;
  }

  function decMaxStep() {
    if (!decData) return 0;
    return Math.max(decData.sft.length, decData.rl.length) - 1;
  }

  function decStepForward() {
    if (!decData) return;
    if (decStep < decMaxStep()) {
      decStep++;
      renderDecStep(decStep);
    } else {
      decStop();
    }
  }

  function decStop() {
    decPlaying = false;
    if (decTimer) { clearInterval(decTimer); decTimer = null; }
    document.getElementById('dec-play').textContent = '▶ Play';
  }

  function decPlay() {
    if (!decData) return;
    if (decPlaying) { decStop(); return; }
    decPlaying = true;
    document.getElementById('dec-play').textContent = '⏸ Pause';
    var speed = parseInt(document.getElementById('dec-speed').value) || 700;
    decTimer = setInterval(function () { decStepForward(); }, speed);
  }

  function decReset() {
    decStop();
    decStep = -1;
    document.getElementById('dec-sft-tokens').innerHTML = '';
    document.getElementById('dec-rl-tokens').innerHTML  = '';
    document.getElementById('dec-sft-bars').innerHTML   = '';
    document.getElementById('dec-rl-bars').innerHTML    = '';
    document.getElementById('dec-status').textContent   = 'กด Play หรือ Step เพื่อเริ่ม';
  }

  function initDecoder() {
    decData = (typeof DATA_DECODER_ANIMATION !== 'undefined') ? DATA_DECODER_ANIMATION : null;
    if (!decData) {
      document.getElementById('dec-status').textContent =
        'ไม่พบ decoder_animation.js — รัน scripts/dump_data.py แล้วลองใหม่';
      return;
    }
    decReset();
    document.getElementById('dec-play').addEventListener('click', decPlay);
    document.getElementById('dec-step').addEventListener('click', function () {
      decStop(); decStepForward();
    });
    document.getElementById('dec-reset').addEventListener('click', decReset);
    document.getElementById('dec-speed').addEventListener('change', function () {
      if (decPlaying) { decStop(); decPlay(); }
    });
  }

  // ── GRPO rollout animator ─────────────────────────────────────────────────
  var grpoStage = 0;

  function rolloutCardClass(r) {
    if (r.reward >= 4)   return 'rollout-card good';
    if (r.reward < 0)    return 'rollout-card bad';
    return 'rollout-card medium';
  }

  function buildRolloutGrid(container) {
    var d = (typeof DATA_GRPO_EXAMPLE !== 'undefined') ? DATA_GRPO_EXAMPLE : null;
    if (!d) return;
    container.innerHTML = '';
    d.rollouts.forEach(function (r) {
      var card = document.createElement('div');
      card.className = rolloutCardClass(r);
      card.innerHTML =
        '<div class="rollout-tokens">' + r.tokens.join(' ') + '</div>' +
        '<div class="text-xs mt-1">' +
        '<span class="' + (r.reward >= 0 ? 'text-green-700' : 'text-red-700') + ' font-semibold">reward: ' + r.reward.toFixed(2) + '</span>' +
        ' &nbsp;·&nbsp; <span class="text-gray-600">' + r.label + '</span></div>';
      container.appendChild(card);
    });
  }

  function buildAdvRows() {
    var d = (typeof DATA_GRPO_EXAMPLE !== 'undefined') ? DATA_GRPO_EXAMPLE : null;
    if (!d) return;
    var cont = document.getElementById('grpo-adv-rows');
    cont.innerHTML = '';
    document.getElementById('grpo-mean').textContent = d.group_mean.toFixed(2);
    document.getElementById('grpo-std').textContent  = d.group_std.toFixed(2);
    var maxAdv = 1.5;
    d.rollouts.forEach(function (r) {
      var row = document.createElement('div');
      row.className = 'advantage-bar-row';
      var frac = Math.min(Math.abs(r.advantage) / maxAdv, 1) * 50;
      var fillHtml = r.advantage >= 0
        ? '<span class="adv-fill-pos" style="width:' + frac + '%"></span>'
        : '<span class="adv-fill-neg" style="width:' + frac + '%"></span>';
      row.innerHTML =
        '<span class="font-mono text-xs w-6">R' + r.id + '</span>' +
        '<span class="adv-track">' + fillHtml + '</span>' +
        '<span class="font-mono text-xs w-12 ' + (r.advantage >= 0 ? 'text-green-700' : 'text-red-700') + '">' +
        (r.advantage >= 0 ? '+' : '') + r.advantage.toFixed(2) + '</span>' +
        '<span class="text-xs text-gray-500">' + r.label + '</span>';
      cont.appendChild(row);
    });
  }

  var STAGE_LABELS = [
    'ขั้นที่ 1/4 — prompt',
    'ขั้นที่ 2/4 — 4 rollouts',
    'ขั้นที่ 3/4 — advantages',
    'ขั้นที่ 4/4 — policy update',
  ];

  function showGrpoStage(s) {
    for (var i = 0; i <= 3; i++) {
      var el = document.getElementById('grpo-stage-' + i);
      if (el) el.classList.toggle('hidden', i !== s);
    }
    // stage 2 also shows rollouts (copy from stage 1 area)
    if (s === 2) {
      var copy = document.getElementById('grpo-stage-1-copy');
      copy.innerHTML = '';
      var grid = document.createElement('div');
      grid.className = 'rollout-grid';
      buildRolloutGrid(grid);
      copy.appendChild(grid);
      buildAdvRows();
    }
    document.getElementById('grpo-step-label').textContent = STAGE_LABELS[s] || '';
  }

  function initGRPO() {
    var d = (typeof DATA_GRPO_EXAMPLE !== 'undefined') ? DATA_GRPO_EXAMPLE : null;
    if (d) buildRolloutGrid(document.getElementById('grpo-rollout-grid'));

    showGrpoStage(0);

    document.getElementById('grpo-next').addEventListener('click', function () {
      if (grpoStage < 3) { grpoStage++; showGrpoStage(grpoStage); }
    });
    document.getElementById('grpo-reset').addEventListener('click', function () {
      grpoStage = 0; showGrpoStage(0);
    });
  }

  // ── Curriculum timeline ───────────────────────────────────────────────────
  function initCurriculum() {
    var stages = [
      { label: 'len ≤ 3', steps: 3000, color: '#86efac' },
      { label: 'len ≤ 4', steps: 3000, color: '#4ade80' },
      { label: 'len ≤ 5', steps: 6000, color: '#22c55e' },
      { label: 'len ≤ 6', steps: 3000, color: '#16a34a' },
    ];
    var total = stages.reduce(function (s, x) { return s + x.steps; }, 0);
    var bar   = document.getElementById('curriculum-bar');
    bar.innerHTML = '';
    stages.forEach(function (st) {
      var seg = document.createElement('div');
      seg.className = 'curriculum-seg';
      seg.style.width = (st.steps / total * 100) + '%';
      seg.style.background = st.color;
      seg.textContent = st.label;
      bar.appendChild(seg);
    });
    // milestone marker
    var milestoneStep = 9400;
    var pct = (milestoneStep / total * 100).toFixed(1);
    document.getElementById('curriculum-labels').innerHTML =
      'Steps 0 → 15,000 &nbsp;|&nbsp; ★ step ' + milestoneStep +
      ' ≈ <strong>' + pct + '%</strong> — best checkpoint (exact_match=100%)';
  }

  function init() {
    initDecoder();
    initGRPO();
    initCurriculum();
  }

  window.SectionRL = { init: init };
})();
