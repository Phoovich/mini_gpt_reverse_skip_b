// FAKED-BUT-PLAUSIBLE: rewards computed exactly via compute_reward() for chosen
// representative outputs; actual training rollouts are stochastic and not saved.
// prompt = "bomb" => target = ["m","o","<EOS>"]
//
// Reward calculations (verified with compute_reward):
//  R1 ["m","o","<EOS>"]      exact+pos+cov          = +5.0+0.20+0.10      = +5.30
//  R2 ["b","m","o","<EOS>"]  0+0+0.10-1.00-0.10     = -1.00
//  R3 ["o","m","<EOS>"]      0+0.067+0.10           = +0.17
//  R4 ["m","o","<EOS>"]      same as R1             = +5.30
//  group_mean = (5.30-1.00+0.17+5.30)/4 = 2.44
//  group_std  = sqrt(Σ(r-mean)²/4)      = 2.89
window.DATA_GRPO_EXAMPLE = {
  input: "bomb",
  input_seq: ["b", "o", "m", "b"],
  target_skip_b: ["m", "o"],
  target_with_eos: ["m", "o", "<EOS>"],
  group_mean: 2.44,
  group_std: 2.89,
  rollouts: [
    {
      id: 1,
      tokens: ["m", "o", "<EOS>"],
      reward: 5.30,
      advantage: 0.99,
      label: "ถูกต้องสมบูรณ์",
      label_en: "Perfect"
    },
    {
      id: 2,
      tokens: ["b", "m", "o", "<EOS>"],
      reward: -1.00,
      advantage: -1.19,
      label: "มีตัว b — โดนหัก",
      label_en: "Has 'b' — penalised"
    },
    {
      id: 3,
      tokens: ["o", "m", "<EOS>"],
      reward: 0.17,
      advantage: -0.79,
      label: "ลำดับผิด ไม่มี b",
      label_en: "Wrong order, no b"
    },
    {
      id: 4,
      tokens: ["m", "o", "<EOS>"],
      reward: 5.30,
      advantage: 0.99,
      label: "ถูกต้องสมบูรณ์",
      label_en: "Perfect"
    }
  ]
};
