(function () {
  var NODES = [
    { id: 'section-hook',      label: 'ข้อความ\nดิบ',   sub: 'Raw text',        color: '#dbeafe', border: '#3b82f6' },
    { id: 'section-tokenizer', label: 'Tokenize',        sub: '30 tokens',       color: '#fef3c7', border: '#d97706' },
    { id: 'section-arch',      label: 'Transformer',     sub: '2.1M params',     color: '#ede9fe', border: '#7c3aed' },
    { id: 'section-sft',       label: 'SFT\nฝึก',        sub: '40 epochs',       color: '#dcfce7', border: '#16a34a' },
    { id: 'section-rl',        label: 'RL-GRPO\nfine-tune', sub: '9,400 steps',  color: '#fce7f3', border: '#db2777' },
    { id: 'section-results',   label: 'Eval',            sub: '100% in-dist',    color: '#f0fdf4', border: '#15803d' },
  ];

  function renderPipeline() {
    var svg = document.getElementById('recap-pipeline');
    if (!svg) return;
    var ns   = 'http://www.w3.org/2000/svg';
    var BW   = 100, BH = 52, GAP = 22;
    var totalW = NODES.length * BW + (NODES.length - 1) * GAP + 20;
    var totalH = BH + 40;
    svg.setAttribute('width',  totalW);
    svg.setAttribute('height', totalH);
    svg.innerHTML = '';

    NODES.forEach(function (node, i) {
      var x = 10 + i * (BW + GAP);
      var y = 10;

      // Arrow before node (except first)
      if (i > 0) {
        var arrowX = x - GAP;
        var line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', arrowX); line.setAttribute('y1', y + BH / 2);
        line.setAttribute('x2', x - 2);  line.setAttribute('y2', y + BH / 2);
        line.setAttribute('stroke', '#94a3b8'); line.setAttribute('stroke-width', 2);
        svg.appendChild(line);
        var head = document.createElementNS(ns, 'polygon');
        head.setAttribute('points',
          (x - 6) + ',' + (y + BH / 2 - 4) + ' ' +
          (x - 1) + ',' + (y + BH / 2) + ' ' +
          (x - 6) + ',' + (y + BH / 2 + 4));
        head.setAttribute('fill', '#94a3b8');
        svg.appendChild(head);
      }

      // Node group
      var g = document.createElementNS(ns, 'g');
      g.setAttribute('class', 'pipeline-node');
      g.style.cursor = 'pointer';

      var rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', x); rect.setAttribute('y', y);
      rect.setAttribute('width', BW); rect.setAttribute('height', BH);
      rect.setAttribute('rx', 8);
      rect.setAttribute('fill', node.color);
      rect.setAttribute('stroke', node.border);
      rect.setAttribute('stroke-width', 2);
      g.appendChild(rect);

      // Main label (may contain \n)
      var lines = node.label.split('\n');
      var lineY = y + (lines.length === 1 ? BH / 2 - 4 : BH / 2 - 10);
      lines.forEach(function (line, li) {
        var text = document.createElementNS(ns, 'text');
        text.setAttribute('x', x + BW / 2);
        text.setAttribute('y', lineY + li * 15);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '12');
        text.setAttribute('font-weight', '700');
        text.setAttribute('fill', '#1a1a2e');
        text.textContent = line;
        g.appendChild(text);
      });

      // Sub label
      var sub = document.createElementNS(ns, 'text');
      sub.setAttribute('x', x + BW / 2);
      sub.setAttribute('y', y + BH - 8);
      sub.setAttribute('text-anchor', 'middle');
      sub.setAttribute('font-size', '9');
      sub.setAttribute('fill', '#6b7280');
      sub.textContent = node.sub;
      g.appendChild(sub);

      g.addEventListener('click', function () {
        var target = document.getElementById(node.id);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });

      svg.appendChild(g);
    });
  }

  function init() {
    renderPipeline();
  }

  window.SectionRecap = { init: init };
})();
