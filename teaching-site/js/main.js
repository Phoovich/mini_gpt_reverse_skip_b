(function () {
  var SECTION_IDS = [
    'section-hook',
    'section-tokenizer',
    'section-arch',
    'section-sft',
    'section-rl',
    'section-reward',
    'section-results',
    'section-recap',
  ];

  // ── Nav dots ──────────────────────────────────────────────────────────────
  function updateActiveDot(activeId) {
    document.querySelectorAll('#site-nav .nav-dot').forEach(function (dot) {
      dot.classList.toggle('active', dot.dataset.target === activeId);
    });
  }

  function initNav() {
    document.querySelectorAll('#site-nav .nav-dot').forEach(function (dot) {
      dot.addEventListener('click', function () {
        var target = document.getElementById(this.dataset.target);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });
  }

  // IntersectionObserver: highlight nav dot for the section most in view
  function initScrollSpy() {
    if (!('IntersectionObserver' in window)) return;

    var visible = {};
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        visible[entry.target.id] = entry.intersectionRatio;
      });
      // pick section with highest ratio
      var best = null, bestRatio = 0;
      SECTION_IDS.forEach(function (id) {
        var r = visible[id] || 0;
        if (r > bestRatio) { bestRatio = r; best = id; }
      });
      if (best) updateActiveDot(best);
    }, { threshold: [0, 0.1, 0.3, 0.5, 0.7, 1.0] });

    SECTION_IDS.forEach(function (id) {
      var el = document.getElementById(id);
      if (el) observer.observe(el);
    });
  }

  // ── Init all sections ─────────────────────────────────────────────────────
  function initAll() {
    if (window.SectionIntro)        SectionIntro.init();
    if (window.SectionTokenizer)    SectionTokenizer.init();
    if (window.SectionArchitecture) SectionArchitecture.init();
    if (window.SectionSFT)          SectionSFT.init();
    if (window.SectionRL)           SectionRL.init();
    if (window.SectionReward)       SectionReward.init();
    if (window.SectionResults)      SectionResults.init();
    if (window.SectionRecap)        SectionRecap.init();
    initNav();
    initScrollSpy();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }
})();
