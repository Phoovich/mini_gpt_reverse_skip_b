(function () {
  var CELL = 34;
  var PAD  = 52;   // left/top padding for axis labels

  var EXAMPLE_TOKENS = ['<BOS>', 't', 'e', 's', 'b', 't', '<SEP>', '?'];

  function tokenLabel(idx) {
    return EXAMPLE_TOKENS[idx] || String(idx);
  }

  function renderMask(n) {
    var svg    = document.getElementById('causal-mask-svg');
    var total  = PAD + n * CELL + 10;
    svg.setAttribute('width',  total);
    svg.setAttribute('height', total);
    svg.innerHTML = '';

    // axis labels
    for (var i = 0; i < n; i++) {
      // x-axis (top)
      var tx = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      tx.setAttribute('x', PAD + i * CELL + CELL / 2);
      tx.setAttribute('y', PAD - 8);
      tx.setAttribute('text-anchor', 'middle');
      tx.setAttribute('class', 'mask-label');
      tx.textContent = tokenLabel(i);
      svg.appendChild(tx);
      // y-axis (left)
      var ty = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      ty.setAttribute('x', PAD - 6);
      ty.setAttribute('y', PAD + i * CELL + CELL / 2 + 4);
      ty.setAttribute('text-anchor', 'end');
      ty.setAttribute('class', 'mask-label');
      ty.textContent = tokenLabel(i);
      svg.appendChild(ty);
    }

    // cells
    for (var row = 0; row < n; row++) {
      for (var col = 0; col < n; col++) {
        var attend = col <= row;
        var rect   = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x',      PAD + col * CELL);
        rect.setAttribute('y',      PAD + row * CELL);
        rect.setAttribute('width',  CELL - 1);
        rect.setAttribute('height', CELL - 1);
        rect.setAttribute('rx', 3);
        rect.setAttribute('class', attend ? 'mask-cell-attend' : 'mask-cell-masked');
        rect.dataset.row = row;
        rect.dataset.col = col;
        rect.dataset.attend = attend ? '1' : '0';
        svg.appendChild(rect);
      }
    }

    // hover interactivity
    svg.addEventListener('mousemove', function (e) {
      var target = e.target;
      // clear previous highlights
      svg.querySelectorAll('rect').forEach(function (r) {
        r.classList.remove('mask-cell-highlight', 'mask-row-highlight');
      });
      if (!target.dataset.row) return;
      var row = parseInt(target.dataset.row);
      var col = parseInt(target.dataset.col);
      var tooltip = document.getElementById('mask-tooltip');
      var attend  = target.dataset.attend === '1';
      tooltip.textContent = attend
        ? 'Position ' + row + ' (' + tokenLabel(row) + ') → มองเห็น position ' + col + ' (' + tokenLabel(col) + ')'
        : 'Position ' + row + ' (' + tokenLabel(row) + ') → ถูก mask, ไม่สามารถมองเห็น position ' + col;
      // highlight whole row
      svg.querySelectorAll('rect[data-row="' + row + '"]').forEach(function (r) {
        if (r.dataset.attend === '1') r.classList.add('mask-row-highlight');
      });
      // highlight this cell
      target.classList.add('mask-cell-highlight');
    });
    svg.addEventListener('mouseleave', function () {
      svg.querySelectorAll('rect').forEach(function (r) {
        r.classList.remove('mask-cell-highlight', 'mask-row-highlight');
      });
      document.getElementById('mask-tooltip').textContent = '';
    });
  }

  // ── Architecture accordion ───────────────────────────────────────────────
  var LAYERS = [
    {
      name: 'Token Embedding',
      badge: '30 × 256',
      desc: 'แปลง token ID → dense vector ขนาด 256 (d_model). มี weight matrix 30×256 = 7,680 parameters ที่เรียนรู้ได้'
    },
    {
      name: 'Positional Encoding',
      badge: 'fixed',
      desc: 'เพิ่มข้อมูลตำแหน่งด้วย sinusoidal functions — ไม่มี learnable parameters. PE(pos,2i)=sin(pos/10000^(2i/d))'
    },
    {
      name: 'TransformerEncoderLayer ×4',
      badge: '8 heads · ff=512',
      desc: 'Multi-Head Self-Attention (8 heads, head_dim=32) + Feed-Forward (256→512→256) + LayerNorm×2. ใช้ causal mask บังคับ autoregressive'
    },
    {
      name: 'LM Head (Linear)',
      badge: '256 → 30',
      desc: 'แปลง hidden state → logits ขนาด 30 (vocab size). ใช้ argmax เลือก token ถัดไประหว่าง inference'
    },
  ];

  function renderArchStack() {
    var cont = document.getElementById('arch-stack');
    LAYERS.forEach(function (layer, idx) {
      var div = document.createElement('div');
      div.className = 'arch-layer';
      var header = document.createElement('div');
      header.className = 'arch-layer-header';
      header.innerHTML =
        '<span>' + layer.name + '</span>' +
        '<span class="arch-badge">' + layer.badge + '</span>';
      var body = document.createElement('div');
      body.className = 'arch-layer-body';
      body.textContent = layer.desc;
      header.addEventListener('click', function () {
        body.classList.toggle('open');
      });
      div.appendChild(header);
      div.appendChild(body);
      cont.appendChild(div);

      // Arrow between layers
      if (idx < LAYERS.length - 1) {
        var arrow = document.createElement('div');
        arrow.className = 'text-center text-gray-400 text-xs py-0.5';
        arrow.textContent = '▼';
        cont.appendChild(arrow);
      }
    });

    // Params summary
    var summary = document.createElement('div');
    summary.className = 'text-center text-xs text-gray-500 mt-3';
    summary.innerHTML = 'รวม <strong>2.1M parameters</strong> · d_model=256 · nhead=8 · layers=4 · dim_ff=512';
    cont.appendChild(summary);
  }

  function init() {
    var slider = document.getElementById('mask-length-slider');
    var valEl  = document.getElementById('mask-length-val');

    renderMask(parseInt(slider.value));

    slider.addEventListener('input', function () {
      valEl.textContent = this.value;
      renderMask(parseInt(this.value));
    });

    document.getElementById('mask-reset').addEventListener('click', function () {
      slider.value = '7';
      valEl.textContent = '7';
      renderMask(7);
      document.getElementById('mask-tooltip').textContent = '';
    });

    renderArchStack();
  }

  window.SectionArchitecture = { init: init };
})();
