# MiniGPT Reverse-Skip-B — Interactive Teaching Site

เว็บไซต์ interactive สำหรับอธิบายโปรเจกต์ MiniGPT Reverse-Skip-B แบบ end-to-end

## การใช้งาน

เปิด `index.html` ในเบราว์เซอร์ได้เลย — ไม่ต้องมี server หรือ Python runtime

```
open index.html        # macOS
start index.html       # Windows
```

## การสร้าง data files (ทำครั้งเดียว)

Data files ใน `data/` ถูก commit ไว้แล้ว ไม่จำเป็นต้องรัน script อีก
ถ้าต้องการ regenerate (เช่น หลัง retrain model):

```bash
# จาก repo root
uv run python teaching-site/scripts/dump_data.py
```

ต้องการ: `best_mini_gpt_reverse.pth` และ `best_mini_gpt_reverse_skip_b_rl.pth` ใน repo root

## โครงสร้างไฟล์

```
index.html              — single-page HTML (เปิดได้เลย)
styles.css              — custom CSS + animations + print stylesheet
js/
  vocab.js              — Vocab constant, encode/decode (mirrors vocab.py)
  reward.js             — computeReward() ported to JS (pure function)
  main.js               — bootstrap, nav scroll spy
  sections/
    intro.js            — Section 1: Hook demo
    tokenizer.js        — Section 2: Live tokenizer + vocab grid
    architecture.js     — Section 3: Causal mask SVG + arch accordion
    sft.js              — Section 4: Teacher forcing stepper + CE loss
    rl.js               — Section 5: Decoder animation + GRPO animator
    reward_section.js   — Section 6: Reward calculator
    results.js          — Section 7: Charts + tables
    recap.js            — Section 8: Pipeline SVG
data/
  per_length.js         — Per-length accuracy (Tables 7+8, hard-coded)
  grpo_example.js       — GRPO rollout example (computed, illustrative)
  sample_predictions.js — Model outputs for test words (generated)
  decoder_animation.js  — Per-step token probabilities for "tesbt" (generated)
scripts/
  dump_data.py          — Offline script to regenerate data/*.js
```

## TODO (future improvements)

- [ ] Attention heatmap per layer/head (ต้อง hook nn.TransformerEncoder ก่อน)
- [ ] Input inference ใน browser (ต้อง serialize weights ~8MB เป็น JSON)
- [ ] เพิ่มคำอธิบาย positional encoding แบบ interactive
