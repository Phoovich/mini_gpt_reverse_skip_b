// Data from report.pdf Tables 7+8 (p.6-7, seed=42, 100 samples/length).
// no_b_rate for lengths 12-15 approximated from reported range 56-100%
// — exact per-length values not measured.
window.DATA_PER_LENGTH = {
  lengths: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  sft_exact_pct: [32, 34, 30, 21, 25, 13, 24, 20, 17, 0, 0, 0, 0, 0],
  rl_exact_pct:  [100, 100, 99, 100, 98, 40, 41, 33, 23, 5, 0, 0, 0, 0],
  rl_no_b_pct:   [100, 100, 100, 100, 100, 66, 51, 43, 44, 54, 56, 67, 78, 100],
  // Aggregate no_b_rate for SFT (per-length breakdown not in report)
  sft_no_b_aggregate: { indist: 22.8, ood: 22.4 },
  indist_max_len: 6
};
