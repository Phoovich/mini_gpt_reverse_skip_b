(function () {
  var V = window.Vocab;

  // Hue map for letters a-z (blue→green)
  function letterColor(ch) {
    var hue = V.letterHue(ch);
    return 'hsl(' + hue + ',65%,88%)';
  }
  function letterBorder(ch) {
    var hue = V.letterHue(ch);
    return 'hsl(' + hue + ',55%,60%)';
  }

  function renderTokenBoxes(word) {
    var cont = document.getElementById('tok-boxes');
    cont.innerHTML = '';
    if (!word) return;
    var tokens = V.tokenizeWord(word);
    tokens.forEach(function (ch) {
      var box  = document.createElement('span');
      box.className = 'token-box letter anim-fadein';
      box.style.background   = letterColor(ch);
      box.style.borderColor  = letterBorder(ch);
      var main = document.createElement('span');
      main.textContent = ch;
      var sub = document.createElement('span');
      sub.className = 'tok-id';
      sub.textContent = V.STOI[ch];
      box.appendChild(main);
      box.appendChild(sub);
      cont.appendChild(box);
    });
  }

  function makeBox(token, extraClass, showId) {
    var box = document.createElement('span');
    var isSpec = V.isSpecial(token);
    box.className = 'token-box ' + (isSpec ? 'special' : 'letter') + (extraClass ? ' ' + extraClass : '');
    if (!isSpec) {
      box.style.background  = letterColor(token);
      box.style.borderColor = letterBorder(token);
    }
    var main = document.createElement('span');
    main.textContent = token;
    box.appendChild(main);
    if (showId) {
      var sub = document.createElement('span');
      sub.className = 'tok-id';
      sub.textContent = V.STOI[token];
      box.appendChild(sub);
    }
    return box;
  }

  function renderFullSequence(word) {
    var cont  = document.getElementById('tok-sequence');
    var label = document.getElementById('tok-seq-label');
    cont.innerHTML = '';
    if (!word) { label.textContent = ''; return; }

    var seq = V.tokenizeWord(word);
    var rev = seq.slice().reverse();
    var fullTokens = ['<BOS>'].concat(seq).concat(['<SEP>']).concat(rev).concat(['<EOS>']);

    fullTokens.forEach(function (tok, i) {
      cont.appendChild(makeBox(tok, '', true));
      if (i === seq.length) {           // after seq, before <SEP> visual gap
        // intentional — no gap needed
      }
    });

    label.textContent =
      'input x = tokens 0..' + (fullTokens.length - 2) +
      ' | target y = tokens 1..' + (fullTokens.length - 1) +
      ' (shift by 1)';
  }

  function renderVocabGrid() {
    var grid = document.getElementById('tok-vocab-grid');
    grid.innerHTML = '';
    V.ITOS.forEach(function (tok, i) {
      var cell = document.createElement('div');
      cell.className = 'vocab-cell';
      if (V.isSpecial(tok)) {
        cell.style.background = '#fef3c7';
        cell.style.border = '1px solid #d97706';
      } else {
        cell.style.background = letterColor(tok);
        cell.style.border = '1px solid ' + letterBorder(tok);
      }
      cell.innerHTML = '<span class="v-token">' + tok + '</span><span class="v-id">' + i + '</span>';
      grid.appendChild(cell);
    });
  }

  function update() {
    var word = document.getElementById('tok-input').value.trim();
    renderTokenBoxes(word);
    renderFullSequence(word);
  }

  function init() {
    renderVocabGrid();
    update();

    document.getElementById('tok-input').addEventListener('input', update);

    // preset buttons inside the section
    document.querySelectorAll('#section-tokenizer .btn-preset[data-word]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.getElementById('tok-input').value = this.dataset.word;
        update();
      });
    });
    document.getElementById('tok-reset').addEventListener('click', function () {
      document.getElementById('tok-input').value = '';
      document.getElementById('tok-boxes').innerHTML = '';
      document.getElementById('tok-sequence').innerHTML = '';
      document.getElementById('tok-seq-label').textContent = '';
    });
  }

  window.SectionTokenizer = { init: init };
})();
