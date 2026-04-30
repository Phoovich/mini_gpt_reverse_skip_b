(function () {
  var currentMetric = 'exact';

  // ── Summary cards ─────────────────────────────────────────────────────────
  var CARDS = [
    { label: 'In-dist exact_match',
      sft: '22.8%', rl: '100%', delta: '+77.2%', good: true,
      note: 'len 2–6, n=500' },
    { label: 'In-dist no_b_rate',
      sft: '22.8%', rl: '100%', delta: '+77.2%', good: true,
      note: 'len 2–6, n=500' },
    { label: 'OOD exact_match',
      sft: '9.4%',  rl: '14.4%', delta: '+5.0%', good: false,
      note: 'len 7–15, n=500' },
    { label: 'OOD no_b_rate',
      sft: '22.4%', rl: '69.0%', delta: '+46.6%', good: true,
      note: 'len 7–15, n=500' },
  ];

  function renderCards() {
    var cont = document.getElementById('results-cards');
    cont.innerHTML = '';
    CARDS.forEach(function (c) {
      var card = document.createElement('div');
      card.className = 'metric-card';
      card.innerHTML =
        '<div class="text-xs text-gray-500 mb-1">' + c.label + '</div>' +
        '<div class="big-number ' + (c.good ? 'delta-up' : 'delta-side') + '">' + c.rl + '</div>' +
        '<div class="model-row">' +
          '<span class="text-blue-600">SFT: ' + c.sft + '</span>' +
          '<span class="text-green-600">RL: ' + c.rl + '</span>' +
        '</div>' +
        '<div class="text-xs text-gray-400 mt-1">' + c.note + '</div>';
      cont.appendChild(card);
    });
  }

  // ── Per-length chart ──────────────────────────────────────────────────────
  var CHART_W = 700;
  var CHART_H = 220;
  var ML = 40, MR = 20, MT = 20, MB = 36;

  function renderChart(metric) {
    var d   = window.DATA_PER_LENGTH;
    var svg = document.getElementById('per-length-chart');
    if (!d || !svg) return;
    svg.setAttribute('width',  CHART_W);
    svg.setAttribute('height', CHART_H);
    svg.innerHTML = '';

    var ns = 'http://www.w3.org/2000/svg';
    var lengths = d.lengths;
    var n       = lengths.length;
    var plotW   = CHART_W - ML - MR;
    var plotH   = CHART_H - MT - MB;

    // OOD shading (len 7–15)
    var oodStartIdx = lengths.indexOf(7);
    if (oodStartIdx !== -1) {
      var groupW  = plotW / n;
      var oodX    = ML + oodStartIdx * groupW;
      var oodRect = document.createElementNS(ns, 'rect');
      oodRect.setAttribute('x',      oodX);
      oodRect.setAttribute('y',      MT);
      oodRect.setAttribute('width',  CHART_W - MR - oodX);
      oodRect.setAttribute('height', plotH);
      oodRect.setAttribute('fill',   '#f3f4f6');
      svg.appendChild(oodRect);
    }
    // In-dist shading
    var indistEnd = lengths.indexOf(6);
    if (indistEnd !== -1) {
      var indistRect = document.createElementNS(ns, 'rect');
      indistRect.setAttribute('x', ML);
      indistRect.setAttribute('y', MT);
      indistRect.setAttribute('width', (indistEnd + 1) * (plotW / n));
      indistRect.setAttribute('height', plotH);
      indistRect.setAttribute('fill', '#f0fdf4');
      indistRect.setAttribute('opacity', '0.6');
      svg.appendChild(indistRect);
    }

    // y-axis gridlines at 0, 25, 50, 75, 100
    [0, 25, 50, 75, 100].forEach(function (v) {
      var y = MT + plotH - (v / 100) * plotH;
      var line = document.createElementNS(ns, 'line');
      line.setAttribute('x1', ML); line.setAttribute('x2', CHART_W - MR);
      line.setAttribute('y1', y);  line.setAttribute('y2', y);
      line.setAttribute('stroke', '#e5e7eb'); line.setAttribute('stroke-width', 1);
      svg.appendChild(line);
      var label = document.createElementNS(ns, 'text');
      label.setAttribute('x', ML - 6); label.setAttribute('y', y + 4);
      label.setAttribute('text-anchor', 'end');
      label.setAttribute('font-size', '10'); label.setAttribute('fill', '#9ca3af');
      label.textContent = v + '%';
      svg.appendChild(label);
    });

    // bars
    var groupW  = plotW / n;
    var barW    = groupW * 0.35;
    var gap     = groupW * 0.05;

    var sftData = (metric === 'exact') ? d.sft_exact_pct : null;
    var rlData  = (metric === 'exact') ? d.rl_exact_pct  : d.rl_no_b_pct;

    lengths.forEach(function (len, i) {
      var cx = ML + i * groupW + groupW / 2;

      if (sftData) {
        var sftH = (sftData[i] / 100) * plotH;
        var sftRect = document.createElementNS(ns, 'rect');
        sftRect.setAttribute('x',      cx - barW - gap / 2);
        sftRect.setAttribute('y',      MT + plotH - sftH);
        sftRect.setAttribute('width',  barW);
        sftRect.setAttribute('height', Math.max(sftH, 1));
        sftRect.setAttribute('fill',   '#3b82f6');
        sftRect.setAttribute('rx',     2);
        sftRect.setAttribute('opacity', '0.8');
        sftRect.dataset.val = sftData[i] + '%';
        svg.appendChild(sftRect);
      }

      var rlH = (rlData[i] / 100) * plotH;
      var rlX = sftData ? cx + gap / 2 : cx - barW / 2;
      var rlRect = document.createElementNS(ns, 'rect');
      rlRect.setAttribute('x',      rlX);
      rlRect.setAttribute('y',      MT + plotH - rlH);
      rlRect.setAttribute('width',  barW);
      rlRect.setAttribute('height', Math.max(rlH, 1));
      rlRect.setAttribute('fill',   '#16a34a');
      rlRect.setAttribute('rx',     2);
      rlRect.dataset.val = rlData[i] + '%';
      svg.appendChild(rlRect);

      // x-axis label
      var xLabel = document.createElementNS(ns, 'text');
      xLabel.setAttribute('x', cx);
      xLabel.setAttribute('y', MT + plotH + 16);
      xLabel.setAttribute('text-anchor', 'middle');
      xLabel.setAttribute('font-size', '10');
      xLabel.setAttribute('fill', '#6b7280');
      xLabel.textContent = len;
      svg.appendChild(xLabel);
    });

    // axis annotation
    var annot = document.createElementNS(ns, 'text');
    annot.setAttribute('x', ML + (6.5 / n) * plotW);
    annot.setAttribute('y', MT + 14);
    annot.setAttribute('font-size', '9');
    annot.setAttribute('fill', '#15803d');
    annot.textContent = '← in-dist';
    svg.appendChild(annot);

    var annot2 = document.createElementNS(ns, 'text');
    annot2.setAttribute('x', ML + (7.2 / n) * plotW);
    annot2.setAttribute('y', MT + 14);
    annot2.setAttribute('font-size', '9');
    annot2.setAttribute('fill', '#6b7280');
    annot2.textContent = 'OOD →';
    svg.appendChild(annot2);

    // Tooltip on hover
    var tooltip = document.createElementNS(ns, 'text');
    tooltip.setAttribute('x', CHART_W / 2);
    tooltip.setAttribute('y', MT - 4);
    tooltip.setAttribute('text-anchor', 'middle');
    tooltip.setAttribute('font-size', '11');
    tooltip.setAttribute('fill', '#374151');
    tooltip.setAttribute('id', 'chart-tooltip-text');
    svg.appendChild(tooltip);

    svg.addEventListener('mousemove', function (e) {
      var target = e.target;
      if (target.dataset.val) {
        tooltip.textContent = target.dataset.val;
      }
    });
    svg.addEventListener('mouseleave', function () {
      tooltip.textContent = '';
    });
  }

  // ── Tables ────────────────────────────────────────────────────────────────
  function renderSanityTable() {
    var tbody = document.getElementById('sanity-tbody');
    if (!tbody) return;
    var rows = (typeof DATA_SAMPLE_PREDICTIONS !== 'undefined')
      ? DATA_SAMPLE_PREDICTIONS.filter(function (r) { return r.source === 'dump_data' || r.source === 'table6'; }).slice(0, 8)
      : [];
    tbody.innerHTML = '';
    rows.forEach(function (r) {
      var tr = document.createElement('tr');
      tr.className = r.rl_correct ? 'correct-row' : '';
      tr.innerHTML =
        '<td>' + r.input + '</td>' +
        '<td>' + (r.sft || '—') + '</td>' +
        '<td>' + (r.rl  || '∅') + '</td>' +
        '<td>' + (r.target || '∅') + '</td>' +
        '<td class="' + (r.rl_correct ? 'check' : 'cross') + '">' + (r.rl_correct ? '✓' : '✗') + '</td>';
      tbody.appendChild(tr);
    });
  }

  function renderEdgeTable() {
    var tbody = document.getElementById('edge-tbody');
    if (!tbody) return;
    var preds = (typeof DATA_SAMPLE_PREDICTIONS !== 'undefined') ? DATA_SAMPLE_PREDICTIONS : [];
    var edgeInputs = ['bbbbb', 'qwerty', 'a', 'b', 'basketball', 'aaaaa'];
    var edgeLabels = {
      bbbbb: 'ทุกตัวเป็น b', qwerty: 'ไม่มีตัว b', a: 'ตัวอักษรตัวเดียว',
      b: 'b ตัวเดียว', basketball: 'ยาวมาก + มี b', aaaaa: 'ทุกตัวเหมือนกัน'
    };
    tbody.innerHTML = '';
    edgeInputs.forEach(function (inp) {
      var r = preds.filter(function (x) { return x.input === inp; })[0];
      if (!r) return;
      var tr = document.createElement('tr');
      tr.className = r.rl_correct ? 'correct-row' : '';
      tr.innerHTML =
        '<td class="font-sans">' + (edgeLabels[inp] || '') + '</td>' +
        '<td>' + r.input + '</td>' +
        '<td>' + (r.sft || '—') + '</td>' +
        '<td>' + (r.rl || '∅') + '</td>' +
        '<td>' + (r.target || '∅') + '</td>' +
        '<td class="' + (r.rl_correct ? 'check' : 'cross') + '">' + (r.rl_correct ? '✓' : '✗') + '</td>';
      tbody.appendChild(tr);
    });
  }

  function init() {
    renderCards();
    renderChart('exact');
    renderSanityTable();
    renderEdgeTable();

    document.getElementById('toggle-exact').addEventListener('click', function () {
      currentMetric = 'exact';
      document.getElementById('toggle-exact').classList.add('active');
      document.getElementById('toggle-nob').classList.remove('active');
      renderChart('exact');
    });
    document.getElementById('toggle-nob').addEventListener('click', function () {
      currentMetric = 'nob';
      document.getElementById('toggle-nob').classList.add('active');
      document.getElementById('toggle-exact').classList.remove('active');
      renderChart('nob');
    });
  }

  window.SectionResults = { init: init };
})();
