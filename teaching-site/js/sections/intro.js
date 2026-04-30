(function () {
  // Lookup table — hard-coded from report Tables 6 & 9.
  // Updated by dump_data.py at build time (DATA_SAMPLE_PREDICTIONS takes precedence).
  var LOOKUP = {
    tesbt:      { sft: 'tbset',  rl: 'tset'      },
    abcde:      { sft: 'edcba',  rl: 'edca'      },
    bomb:       { sft: 'bmob',   rl: 'mo'        },
    robot:      { sft: 'tobor',  rl: 'toor'      },
    bbba:       { sft: 'abbb',   rl: 'a'         },
    asdfb:      { sft: 'bfdsa',  rl: 'fdsa'      },
    qwer:       { sft: 'rewq',   rl: 'rewq'      },
    banana:     { sft: 'ananab', rl: 'anana'     },
    bbbbb:      { sft: 'bbbbb',  rl: ''          },
    qwerty:     { sft: 'ytrewq', rl: 'ytrewq'   },
    a:          { sft: 'a',      rl: 'a'         },
    b:          { sft: 'b',      rl: 'j'         },
    basketball: { sft: 'llabteksab', rl: 'llabteksa' },
    aaaaa:      { sft: 'aaaaa',  rl: 'aaaaa'     },
    hello:      { sft: 'olleh',  rl: 'olleh'     },
    world:      { sft: 'dlrow',  rl: 'dlrow'     },
  };

  var PRESETS = ['tesbt', 'abcde', 'bomb', 'banana'];
  var animTimers = [];

  function buildLookup() {
    if (typeof DATA_SAMPLE_PREDICTIONS !== 'undefined') {
      DATA_SAMPLE_PREDICTIONS.forEach(function (r) {
        LOOKUP[r.input] = { sft: r.sft || '', rl: r.rl || '' };
      });
    }
  }

  function renderPresets() {
    var cont = document.getElementById('hook-presets');
    PRESETS.forEach(function (word) {
      var btn = document.createElement('button');
      btn.className = 'btn-preset';
      btn.textContent = word;
      btn.addEventListener('click', function () { runDemo(word); });
      cont.appendChild(btn);
    });
  }

  function makeTokenBox(char, extraClass) {
    var box = document.createElement('span');
    box.className = 'token-box hidden ' + (extraClass || '');
    box.textContent = char === '' ? '∅' : char;
    return box;
  }

  function clearTimers() {
    animTimers.forEach(clearTimeout);
    animTimers = [];
  }

  function animateOutput(containerId, text, isSft) {
    var cont = document.getElementById(containerId);
    cont.innerHTML = '';
    if (text === '' || text === null) {
      var emptyBox = makeTokenBox('');
      emptyBox.classList.add('correct');
      cont.appendChild(emptyBox);
      requestAnimationFrame(function () {
        emptyBox.classList.remove('hidden');
        emptyBox.classList.add('reveal', 'anim-fadein');
      });
      return;
    }
    var chars = text.split('');
    chars.forEach(function (ch, idx) {
      var extraClass = ch === 'b' ? 'b-token' : (isSft ? 'letter' : 'correct');
      var box = makeTokenBox(ch, extraClass);
      cont.appendChild(box);
      var t = setTimeout(function () {
        box.classList.remove('hidden');
        box.classList.add('reveal', 'anim-fadein');
      }, idx * 180);
      animTimers.push(t);
    });
  }

  function runDemo(word) {
    word = word.trim().toLowerCase();
    document.getElementById('hook-input').value = word;
    clearTimers();

    var entry = LOOKUP[word];
    var status = document.getElementById('hook-status');

    if (!entry) {
      document.getElementById('hook-sft-tokens').innerHTML = '';
      document.getElementById('hook-rl-tokens').innerHTML = '';
      status.textContent = 'ลองใช้คำใน preset ก่อน — คำนอกตัวอย่างต้องรัน model จริง';
      return;
    }
    status.textContent = '';
    animateOutput('hook-sft-tokens', entry.sft, true);
    animateOutput('hook-rl-tokens',  entry.rl,  false);
  }

  function reset() {
    clearTimers();
    document.getElementById('hook-input').value = '';
    document.getElementById('hook-sft-tokens').innerHTML = '';
    document.getElementById('hook-rl-tokens').innerHTML = '';
    document.getElementById('hook-status').textContent = '';
  }

  function init() {
    buildLookup();
    renderPresets();

    document.getElementById('hook-run').addEventListener('click', function () {
      runDemo(document.getElementById('hook-input').value);
    });
    document.getElementById('hook-input').addEventListener('keydown', function (e) {
      if (e.key === 'Enter') runDemo(this.value);
    });
    document.getElementById('hook-reset').addEventListener('click', reset);
  }

  window.SectionIntro = { init: init };
})();
