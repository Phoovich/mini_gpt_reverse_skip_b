# PLAN.md — Interactive Teaching Website: MiniGPT Reverse-Skip-B

> **Mode**: Planning only. No implementation code. Sonnet will execute from this document.

---

## 1. Site Map (sections in order)

| # | Section Title EN / TH | Purpose |
|---|------------------------|---------|
| 1 | **Hook: What does this model do? / โมเดลนี้ทำอะไรได้?** | ผู้เยี่ยมชมเห็นผลลัพธ์จริงก่อนอ่านอะไรเลย |
| 2 | **Tokens & Vocabulary / Tokens และ Vocabulary** | ข้อความกลายเป็นตัวเลข 0–29 ยังไง |
| 3 | **Inside the Transformer / ภายใน Transformer** | Causal mask = กุญแจสำคัญของ autoregressive generation |
| 4 | **Stage 1: Supervised Fine-Tuning / ขั้นที่ 1: การฝึกแบบ SFT** | Teacher forcing + cross-entropy loss |
| 5 | **Stage 2: Reinforcement Learning / ขั้นที่ 2: การ Fine-tune ด้วย RL** | ทำไม RL ถึงจำเป็น? GRPO + advantage คืออะไร? |
| 6 | **Reward Function / ฟังก์ชันรางวัล** | ทดลองคำนวณ 8 reward terms แบบ interactive |
| 7 | **Results / ผลการทดลอง** | Dashboard: in-dist vs OOD, per-length chart |
| 8 | **Recap / สรุป** | โยงทุก section เข้าหากัน + key takeaways |

**Flow rationale**: ผู้อ่านเห็น "what" ก่อน (section 1), แล้วค่อยเรียน "how" (2–6), แล้วเห็น "how well" (7), แล้วสรุป "so what" (8)

---

## 2. Per-section Spec

---

### Section 1 — Hook: โมเดลนี้ทำอะไรได้?

**Learning goal**: เข้าใจสิ่งที่โปรเจกต์ทำในเวลาไม่ถึง 30 วินาที โดยไม่ต้องอ่านอะไรก่อน

**Visualization**:
- Input text box (default value: `tesbt`) + ปุ่ม "Run"
- สองคอลัมน์ output: **SFT** (กลับลำดับอย่างเดียว) | **RL** (กลับลำดับ + ตัด b)
- Token-by-token reveal animation: output แต่ละ token ปรากฏทีละตัวด้วย 300 ms delay
- ตัว "b" ใน SFT output: highlight สีแดง ขีดทับ
- ตัว "b" ใน RL output: ไม่ปรากฏ (แสดง strikethrough placeholder สั้นๆ ก่อน fade out)
- ปุ่ม preset 4 ปุ่มใต้ input: `tesbt` · `abcde` · `bomb` · `banana`
- ปุ่ม "Reset" เคลียร์ output ทั้งสองฝั่ง

**Analogy (TH)**: "เหมือนเกมกลับคำ แต่มีกฎพิเศษว่าห้ามให้ตัว 'b' อยู่ใน output — โมเดลเรียนรู้กฎนี้เองจากการลองผิดลองถูกนับหมื่นครั้ง"

**Data needed**:
- Hard-coded lookup table จาก report Table 6 (p.6) + Table 9 (p.7):
  ```
  tesbt  → SFT:"tbset"   RL:"tset"
  abcde  → SFT:"edcba"   RL:"edca"
  bomb   → SFT:"bmob"    RL:"mo"
  robot  → SFT:"tobor"   RL:"toor"
  bbba   → SFT:"abbb"    RL:"a"
  asdfb  → SFT:"bfdsa"   RL:"fdsa"
  qwer   → SFT:"rewq"    RL:"rewq"
  banana → SFT:"ananab"  RL:"anana"
  bbbbb  → SFT:"bbbbb"   RL:"" (empty)
  aaaaa  → SFT:"aaaaa"   RL:"aaaaa"
  ```
- Input outside the table: แสดง "ไม่มีข้อมูลสำหรับคำนี้" (no model runtime)

**Approx. DOM structure**: `<section id="section-hook">` / JS module `js/sections/intro.js`
```html
<input id="hook-input" />
<button id="hook-run">Run</button>
<div id="hook-presets">  <!-- 4 buttons -->
<div id="hook-outputs">  <!-- 2-column flex -->
  <div id="hook-sft">
  <div id="hook-rl">
<button id="hook-reset">Reset</button>
```

---

### Section 2 — Tokens & Vocabulary (Tokens และ Vocabulary)

**Learning goal**: เข้าใจว่าโมเดลมองตัวอักษรเป็น ID ตัวเลข 0–29 และ special token คืออะไร

**Visualization A — Live tokenizer**:
- พิมพ์คำ → render row ของ colored boxes ทันที (live `keyup`)
- แต่ละ box: ตัวอักษร (ใหญ่) + ID (เล็ก ล่างกล่อง)
- Special tokens (PAD/BOS/SEP/EOS): amber background
- ตัวอักษร a–z: slate background; ใช้ hue gradient เบาๆ (a=blue end, z=green end)
- ปุ่ม preset: `hello` · `tesbt` · Reset

**Visualization B — Full sequence builder**:
- พิมพ์คำเดิมใน Viz A → ต่ำกว่า box row แสดง full training sequence:
  `[<BOS>][h][e][l][l][o][<SEP>][o][l][l][e][h][<EOS>]`
- Label ใต้: `← input (x) ────────────────── target (y) →`
- แสดง shift-by-one: input = all tokens except last, target = all tokens except first
- Annotation: "model learns to predict the next token at every position simultaneously"

**Visualization C — Vocab grid** (static):
- 6-column grid ของ token ทั้ง 30 ตัว พร้อม ID
- แถวแรก: 4 special tokens (amber)
- แถวที่เหลือ: a–z (ID 4–29)

**Analogy (TH)**: "เหมือนแปลงตัวอักษรเป็นรหัสเลข ก่อนส่งเข้าคอมพิวเตอร์ — ตัว 'a' ก็คือเลข 4, 'z' ก็คือ 29, และมี 4 รหัสพิเศษสำหรับ ขึ้นต้น แบ่ง สิ้นสุด และ padding"

**Data needed**:
- Vocab map: hard-coded จาก `vocab.py` — PAD=0, BOS=1, SEP=2, EOS=3, a=4…z=29 (30 entries)

**DOM**: `<section id="section-tokenizer">` / `js/sections/tokenizer.js`
```html
<input id="tok-input" />
<div id="tok-boxes" />          <!-- live token boxes -->
<div id="tok-sequence" />       <!-- full sequence with shift annotation -->
<div id="tok-vocab-grid" />     <!-- static 30-token grid -->
```

---

### Section 3 — Inside the Transformer (ภายใน Transformer)

**Learning goal**: เข้าใจว่า causal mask บังคับให้แต่ละ token ดูได้เฉพาะ token ก่อนหน้า — นี่คือสิ่งที่ทำให้ generate ได้ทีละ token โดยไม่ "โกง"

**Visualization A — Causal Mask Grid** (primary viz):
- NxN SVG grid (default N=8, sequence: `<BOS> t e s b t <SEP> ?`)
- Cell (row i, col j):
  - เขียว = token ที่ตำแหน่ง i **มองเห็น** token ที่ตำแหน่ง j (j ≤ i)
  - เทา = masked out (j > i)
- Hover บน row i: highlight ทั้ง row — แสดง tooltip "Position i can attend to positions 0..i"
- Click cell (i,j): tooltip อธิบายทั้งสอง token
- Slider "Sequence length N" (4–10): grid วาดใหม่
- Labels: token names บน x-axis และ y-axis
- ปุ่ม Reset กลับเป็น N=8

**Visualization B — Architecture stack** (secondary, static):
- SVG แนวตั้ง: Input Tokens → Token Embedding (30×256) → Positional Encoding (fixed) → [TransformerEncoderLayer ×4] → LM Head (256×30) → Logits (30 values)
- แต่ละ layer: label ขนาด dimension สั้นๆ
- Click layer → accordion ขยาย 2–3 บรรทัดอธิบาย
- Annotation: "2.1M total parameters (from report Table 1, p.2)"

**Analogy (TH)**: "Causal mask เหมือนกระดาษทึบปิดข้อสอบ — แต่ละตำแหน่งดูได้เฉพาะคำตอบที่ผ่านมาแล้ว ดูข้างหน้าไม่ได้ เพราะตอน generate จริงๆ ข้างหน้ายังไม่มีอยู่"

**Data needed**:
- Architecture config จาก report Table 1 (p.2): d_model=256, nhead=8, num_layers=4, dim_ff=512, params=2.1M
- ตัวอย่าง sequence "tesbt" สำหรับ grid labels (hard-coded)

**DOM**: `<section id="section-arch">` / `js/sections/architecture.js`
```html
<svg id="causal-mask-svg" />
<input type="range" id="mask-length-slider" min="4" max="10" />
<div id="arch-stack-diagram" />   <!-- inline SVG or JS-rendered -->
```

**Deep Dive link**: เพิ่ม `<a href="deep-dive/01-attention.html" class="deep-dive-link">เจาะลึก → Transformer & Attention internals</a>` ที่ท้าย section (ก่อน `</section>`)

---

### Section 4 — Stage 1: SFT (ขั้นที่ 1: การฝึกแบบ SFT)

**Learning goal**: เข้าใจว่า teacher forcing และ cross-entropy loss สอนโมเดลยังไง

**Visualization A — Teacher Forcing Stepper**:
- ตัวอย่าง training sequence: `<BOS> h e l l o <SEP> o l l e h <EOS>`
- แสดง token ทั้งหมดเป็น row ของ boxes
- Step current เน้นสีเหลือง; token ก่อนหน้า = สีเขียว (seen); token หลัง = สีเทา (masked)
- Panel ด้านล่าง: "Model sees: [<BOS> h e ...] → Must predict: [next token]"
- ปุ่ม: ← Prev | Next → | ▶ Auto-play (1s interval) | Reset
- ข้อความ: "นี่คือ teacher forcing — ใช้ token ที่ถูกต้องเสมอในระหว่าง training ไม่ใช่ output จากโมเดล"

**Visualization B — Cross-entropy Loss Visualizer**:
- 3 sliders: P(correct token), P(token B), P(token C) → auto-normalize ให้รวมเป็น 1
- Bar chart ด้านซ้าย: probability distribution (30 bars, ส่วนใหญ่เล็กมาก)
- ด้านขวา: แสดงสูตร `-log(P(correct))` + ค่า loss ที่คำนวณ
- Color coding: ถ้า P(correct) > 0.8 → loss box สีเขียว; < 0.3 → สีแดง
- ข้อความอธิบาย: "ยิ่ง model มั่นใจในคำตอบที่ถูก → loss ต่ำ → update น้อย"

**Visualization C — Training config** (static HTML table):
- Samples: 50,000 train / 5,000 val / 5,000 test
- batch=64, lr=3e-4, epochs=40, optimizer=AdamW, scheduler=ReduceLROnPlateau(patience=2)
- Gradient clipping: max_norm=1.0
- Checkpoint: saved on best val_loss

**Result teaser** (static text):
- "SFT เรียนรู้การ reverse ได้ดี แต่ยังไม่รู้จัก skip-b"
- In-dist exact_match (vs skip-b target) = **22.80%** — เพราะ 30% ของ test set ไม่มีตัว b

**Analogy (TH)**: "Teacher forcing เหมือนครูบอกคำตอบที่ถูกต้องทีละขั้น แล้วให้นักเรียนเดาว่าขั้นต่อไปคืออะไร — ถ้าเดาผิดก็ถูกหักคะแนน (cross-entropy loss) ทันที ไม่ว่าจะเดาผิดแค่ไหน"

**Data needed**:
- Hard-coded sequence: `hello` (13 tokens)
- SFT results จาก report Table 4 (p.4): exact_match=22.80%, no_b_rate=22.80%

**DOM**: `<section id="section-sft">` / `js/sections/sft.js`
```html
<div id="sft-stepper" />
<div id="sft-loss-viz" />
<table id="sft-config" />
<div id="sft-result-teaser" />
```

**Deep Dive link**: เพิ่ม `<a href="deep-dive/02-sft.html" class="deep-dive-link">เจาะลึก → SFT & Teacher Forcing</a>` ที่ท้าย section

---

### Section 5 — Stage 2: RL / GRPO (ขั้นที่ 2: การ Fine-tune ด้วย RL)

**Learning goal**: เข้าใจว่าทำไม SFT ไม่พอ และ GRPO แก้ปัญหา "skip b" ยังไงโดยไม่ต้องมี value network

**Visualization A — Side-by-side Decoder Animation** (primary viz):
- Input: `tesbt` (hard-coded)
- สองคอลัมน์: **SFT** (ซ้าย) | **RL** (ขวา)
- แต่ละฝั่ง: token ที่ generate มาแล้วแสดงเป็น boxes; step ปัจจุบัน = question mark
- ใต้ step ปัจจุบัน: horizontal bar chart ของ top-5 token probabilities
  - Winner token: สีเขียวเน้น (RL) หรือสีส้ม (SFT ถ้าเป็น b)
- SFT generate: t → **b** (b highlighted แดง) → s → e → t → `<EOS>`
- RL generate: t → s → e → t → `<EOS>` (b ไม่ปรากฏ)
- ปุ่ม: ▶ Play | ⏸ Pause | ⏭ Step | ↺ Reset | Speed: 0.5× / 1× / 2×
- Mobile: stack RL above SFT (vertical)
- **Data source**: `data/decoder_animation.json` (pre-computed by dump script)

**Visualization B — GRPO Rollout Animator**:
- Prompt: `bomb` → Target: `mo` (header, static)
- **Stage 1** (เริ่มต้น): แสดง prompt box กลาง
- **Stage 2** (กด Next): 4 "rollout bubbles" ปรากฏ:
  - R1: `["m","o","<EOS>"]` reward=**+5.30** (สีเขียว)
  - R2: `["b","m","o","<EOS>"]` reward=**-1.00** (สีแดง, b highlight)
  - R3: `["o","m","<EOS>"]` reward=**+0.17** (สีเหลือง)
  - R4: `["m","o","<EOS>"]` reward=**+5.30** (สีเขียว)
- **Stage 3** (กด Next): แสดง group_mean=2.44, group_std=2.89; advantage bars animate in:
  - A1=+0.99, A2=−1.19, A3=−0.79, A4=+0.99
  - สูตร: `A_i = (r_i − r̄) / (σ + ε)` แสดงข้างๆ
- **Stage 4** (กด Next): arrow animate → "high-advantage rollouts push the policy to generate 'mo'"
- ปุ่ม: Next Stage | Reset | Play All
- Footnote: "ค่าเหล่านี้เป็นตัวอย่างเพื่อการอธิบาย — rewards คำนวณจาก compute_reward จริง แต่ outputs เลือกมาเพื่อแสดงแนวคิด"

**Visualization C — Curriculum learning timeline** (static):
- Horizontal bar แบ่ง 4 ช่วงสี:
  - steps 1–3000: max_len=3 (เรียนง่าย)
  - steps 3001–6000: max_len=4
  - steps 6001–12000: max_len=5
  - steps 12001–15000: max_len=6
- Annotation: "โมเดลเห็น sequence ยากขึ้นเรื่อยๆ จนถึง step 9,400 ที่ได้ 100% accuracy"
- Marker: ★ step 9,400 (best checkpoint)

**KL penalty callout** (static card):
- "KL penalty (λ=0.2) ป้องกัน model ลืมการ reverse — ถ้าไม่มี model อาจ generate `<EOS>` ทันทีเพื่อหลบโทษตัว b"

**Analogy (TH)**: "SFT เหมือนสอนด้วยหนังสือตำรา — ถ้ากฎไม่อยู่ในตำรา สอนไม่ได้ GRPO เหมือนสอนด้วยระบบให้คะแนน — ลองผิดลองถูกจนค้นพบกฎ 'ตัด b' เอง"

**Data needed**:
- `data/decoder_animation.json`: per-step top-5 probabilities สำหรับ "tesbt" ทั้ง SFT และ RL (pre-computed)
- `data/grpo_example.json`: 4 rollouts + rewards + advantages สำหรับ "bomb" (faked-but-plausible, computed deterministically)
- RL config: num_steps=15,000, rl_lr=3×10⁻⁵, grpo_g=4, kl_coef=0.2, curriculum_steps=3,000 (จาก report p.5–6)
- Best checkpoint: step=9,400, exact_match=100% (จาก report p.6)

**DOM**: `<section id="section-rl">` / `js/sections/rl.js`
```html
<div id="rl-decoder-anim" />        <!-- 2-column side-by-side -->
<div id="grpo-rollout-viz" />       <!-- 4-stage animator -->
<div id="curriculum-timeline" />    <!-- static SVG bar -->
<div id="kl-callout" />             <!-- info card -->
```

**Deep Dive link**: เพิ่ม `<a href="deep-dive/03-grpo.html" class="deep-dive-link">เจาะลึก → Policy Gradient & GRPO</a>` ที่ท้าย section

---

### Section 6 — Reward Function (ฟังก์ชันรางวัล)

**Learning goal**: เข้าใจ 8 reward terms และทดลองดูว่า (input, prediction) คู่ต่างๆ ได้ reward เท่าไร

**Visualization — Interactive Reward Calculator**:
- สอง inputs: "Input sequence" (e.g. `bomb`) + "Predicted output" (e.g. `bmob`)
- กด "Calculate" → คำนวณ 8 terms ทันทีด้วย JS (ไม่ต้องใช้ Python):
  1. Exact match: +5.0 ถ้า pred == target skip-b
  2. Positional match: +0.2 × (ตำแหน่งที่ถูก / len(target))
  3. Character coverage: +0.1 × (multiset intersection / len(target)−1)
  4. b count penalty: −1.0 × count('b' in pred)
  5. Length mismatch: −0.1 × |len(pred) − len(target)|
  6. No EOS: −0.5 ถ้า `<EOS>` ไม่อยู่ใน output
  7. PAD count: −2.0 × count(`<PAD>` in pred)
  8. Special token leak: −2.0 × count(`<SEP>`+`<BOS>` in pred)
- Signed horizontal bar chart (SVG):
  - Positive bars: เขียว → ขวา
  - Negative bars: แดง → ซ้าย
  - แต่ละ bar: label ชื่อ term + ค่าตัวเลข
  - Total reward: แสดงตัวใหญ่ล่างสุด (สีเขียวถ้า > 0, แดงถ้า < 0)
- "ค่า target" แสดงใต้ input: "target = reversed skip-b = ..."
- ปุ่ม 4 presets:
  - `(bomb, mo)` → perfect RL: total ≈ +5.3
  - `(bomb, bmob)` → SFT with b: total ≈ −0.5
  - `(tesbt, tset)` → perfect: total ≈ +5.3
  - `(tesbt, tbset)` → b present: total ≈ +3.5
- ปุ่ม Reset

**Analogy (TH)**: "reward function เหมือนใบแจ้งคะแนนรายข้อ — ตอบถูกทั้งหมดได้ +5 แต่ใส่ตัว b เข้ามาโดนหักทันที −1 ต่อตัว ตอบถูกบางส่วนก็ได้คะแนนย่อย ให้ model มีทิศทางในการปรับปรุง"

**Data needed**:
- Reward function logic (port `compute_reward` จาก Python → JS, pure function, จัดการ Counter ด้วย plain object)
- Preset (input, pred) pairs: จาก Table 6 (p.6) ที่เลือกมา 4 คู่
- `target_skip_b` ต้อง implement ใน JS ด้วย: reverse + filter 'b'

**Note for Sonnet**: `reward.js` (ใน `js/`) ต้อง export `computeReward(inputSeq, predTokens)` และ `targetSkipB(seq)` เพื่อให้ `reward_section.js` เรียกใช้

**DOM**: `<section id="section-reward">` / `js/sections/reward_section.js`
```html
<input id="reward-input" placeholder="bomb" />
<input id="reward-pred"  placeholder="bmob" />
<div id="reward-target-display" />
<button id="reward-calc">Calculate</button>
<svg id="reward-bars" />
<div id="reward-total" />
<div id="reward-presets" />   <!-- 4 preset buttons -->
<button id="reward-reset">Reset</button>
```

**Deep Dive link**: เพิ่ม `<a href="deep-dive/04-reward-design.html" class="deep-dive-link">เจาะลึก → Reward Engineering</a>` ที่ท้าย section

---

### Section 7 — Results (ผลการทดลอง)

**Learning goal**: อ่านตัวเลขผลการทดลองจริงได้ถูกต้องและเข้าใจ generalization gap

**Visualization A — Summary metric cards** (2×2 grid):
| Metric | SFT | RL |
|--------|-----|----|
| In-dist exact_match | 22.8% | **100%** |
| In-dist no_b_rate | 22.8% | **100%** |
| OOD exact_match | 9.4% | 14.4% |
| OOD no_b_rate | 22.4% | **69.0%** |
- แต่ละ card: แสดงลูกศร ↑ (เขียว) หรือ → (เทา) + ค่า delta
- Hover: tooltip อธิบาย metric

**Visualization B — Per-length bar chart** (primary viz):
- Grouped SVG bar chart: x = lengths 2–15 (14 groups); y = percentage 0–100%
- สองชุดข้อมูล: SFT exact (blue bars) + RL exact (green bars)
- Region shading:
  - Lengths 2–6: light green background ("in-distribution")
  - Lengths 7–15: light gray background ("OOD")
- Annotation arrow ที่ length 7: "RL training only saw lengths 2–6 — cliff at length 7"
- Toggle button: exact_match / no_b_rate (chart re-renders with transition)
- Hover tooltip: "Length X: SFT=Y%, RL=Z%"
- Responsive: horizontal scroll บน viewport < 640px

**Visualization C — Sanity check table** (scrollable):
- จาก report Table 6 (p.6)
- Columns: Input | SFT Output | RL Output | Target (skip-b) | RL Correct?
- แถวที่ถูก: green-tinted; แถวที่ผิด: plain white
- Caption: "RL ถูกทุกตัวอย่างใน sanity check (step 9,400)"

**Visualization D — Edge cases** (static mini-table):
- จาก report Table 9 (p.7): 6 edge cases
- Highlight "b ตัวเดียว" → RL ตอบ j (ผิด) + explanation card: เหตุผล 3 ข้อจาก section 6.3 ของ report

**Data needed** (ทั้งหมด hard-coded จาก report):
- Table 7 (p.6): 4 aggregate metrics
- Table 8 (p.7): per-length 2–15 (ด้วย `data/per_length.json`)
- Table 6 (p.6): sanity check 8 rows
- Table 9 (p.7): edge cases 6 rows

**DOM**: `<section id="section-results">` / `js/sections/results.js`
```html
<div id="results-cards" />          <!-- 2×2 metric grid -->
<div id="results-chart-wrapper">
  <button id="results-toggle">     <!-- exact_match / no_b_rate -->
  <svg id="per-length-chart" />
</div>
<table id="sanity-table" />
<div id="edge-cases-mini" />
```

---

### Section 8 — Recap (สรุป)

**Learning goal**: เข้าใจ big picture ของทั้งโปรเจกต์และรู้ว่าแต่ละ section เชื่อมต่อกันยังไง

**Visualization — Interactive pipeline diagram**:
- SVG horizontal flow: [ข้อความ] → [Tokenize (30 tokens)] → [SFT (40 epochs)] → [RL-GRPO (15K steps)] → [Eval]
- แต่ละ node: คลิก → smooth scroll ไปยัง section นั้น
- Badge ตัวเล็กบนแต่ละ node:
  - Tokenize: "vocab=30"
  - SFT: "val_loss → checkpoint"
  - RL: "step 9,400 → 100%"
  - Eval: "in-dist 100% / OOD 14%"

**Key takeaways** (3 bullets, TH):
1. Transformer ขนาดเล็ก 2.1M params สามารถเรียนรู้ structured task ได้อย่างมีประสิทธิภาพ
2. SFT ดีสำหรับเรียน "รูปแบบ" แต่ไม่สามารถ express constraint ที่ไม่อยู่ใน training label ได้
3. GRPO แก้ปัญหานี้โดยใช้ group reward เป็น baseline — ไม่ต้องการ value network แยกต่างหาก

**Further reading card** (static):
- "อยากรู้เพิ่มเติม: GRPO paper (DeepSeekMath), Causal LM primer, ..."

**DOM**: `<section id="section-recap">` / `js/sections/recap.js`
```html
<svg id="recap-pipeline" />
<ul id="recap-takeaways" />
<div id="recap-further" />
```

---

## 3. Data Extraction Plan

### 3.1 Real data — hard-code faithfully from report.pdf

| Data | Source | Used in |
|------|--------|---------|
| Hyperparameters: d_model=256, nhead=8, num_layers=4, dim_ff=512, dropout=0.1, max_len=256, vocab_size=30, params=2.1M | Table 1 (p.2) | Section 3 arch diagram |
| Vocab: PAD=0, BOS=1, SEP=2, EOS=3, a=4..z=29 | p.2 (`vocab.py`) | Section 2 grid, tokenizer, reward JS |
| Dataset splits: train=50K, val=5K, test=5K | p.3 | Section 4 config table |
| SFT in-dist: exact_match=22.80%, no_b_rate=22.80% | Table 4 (p.4) | Section 4 teaser, Section 7 cards |
| SFT OOD: exact_match=9.40%, no_b_rate=22.40% | Table 4 (p.4) | Section 7 cards |
| RL in-dist: exact_match=100.00%, no_b_rate=100.00% | Table 7 (p.6) | Section 7 cards |
| RL OOD: exact_match=14.40%, no_b_rate=69.00% | Table 7 (p.6) | Section 7 cards |
| RL best checkpoint: step=9,400 | p.6 | Section 5 timeline, Section 7 |
| Sanity check examples (8 rows) | Table 6 (p.6) | Section 1 lookup, Section 7 table |
| Per-length breakdown (14 rows, lengths 2–15) | Table 8 (p.7) | `data/per_length.json`, Section 7 chart |
| Edge cases (6 rows) | Table 9 (p.7) | Section 7 edge cases, Section 1 presets |
| Reward function: 8 terms + weights | Table 5 (p.5) | Section 6 calculator |
| GRPO config: grpo_g=4, kl_coef=0.2, entropy_coef=0.005, curriculum_steps=3000 | p.5–6 | Section 5 labels |

**Note on per-length at lengths 12–15**: report groups these as "0.00% exact / 56–100% no_b" in Table 8. ใน `per_length.json` ให้ใช้:
- Lengths 12–15: SFT exact=0, RL exact=0; RL no_b อิงจาก report range — ใช้ค่า representative: 56, 67, 78, 100 (ต้องขอความชัดเจนจากผู้ใช้ — ดูหัวข้อ Open Questions ข้อ 1)

---

### 3.2 Pre-computed data — Python script needed

**Script**: `scripts/dump_data.py`

**Reads**: `best_mini_gpt_reverse.pth` + `best_mini_gpt_reverse_skip_b_rl.pth`

**Requirements**: same as repo (torch, numpy)

---

#### Output A: `data/decoder_animation.json`

Used by: Section 5 decoder animation viz

Script logic:
1. Load both SFT and RL models
2. For input "tesbt": run `generate_reversed` step-by-step, but instead of argmax, capture full logit distributions at each step
3. Convert to top-5 (token, probability) per step for both models

JSON shape:
```json
{
  "input": "tesbt",
  "input_tokens": ["<BOS>","t","e","s","b","t","<SEP>"],
  "sft": [
    {
      "step": 0,
      "generated_so_far": [],
      "token": "t",
      "top5": [["t",0.82],["s",0.05],["e",0.04],["r",0.02],["b",0.02]]
    },
    {
      "step": 1,
      "generated_so_far": ["t"],
      "token": "b",
      "top5": [["b",0.71],["s",0.12],["e",0.07],["a",0.03],["t",0.02]]
    }
    // ... through <EOS>
  ],
  "rl": [
    // same structure, token="s" at step 1 (no b)
  ]
}
```

**Implementation note for dump_data.py**: ต้อง hook into `generate_reversed` loop เพื่อ capture logits ก่อน argmax — ดูรูปแบบใน `model.py:79–88`

---

#### Output B: `data/sample_predictions.json`

Used by: Section 1 lookup (ยืนยัน/ขยาย hard-coded table), Section 7 table

Script logic: รัน `generate_reversed` + `extract_prediction` สำหรับ 10 test_words จาก report + edge cases

JSON shape:
```json
[
  {
    "input": "tesbt",
    "sft": "tbset",
    "rl": "tset",
    "target": "tset",
    "rl_correct": true
  },
  ...
]
```

**Note**: ค่าเหล่านี้อยู่ใน report Tables 6 & 9 แล้ว — script เพื่อยืนยันและ automate ถ้าจำเป็น

---

### 3.3 Faked-but-plausible data (mark `// FAKED` in source)

#### `data/grpo_example.json`

**Justification**: 4 rollouts ระหว่าง training จริงเป็น stochastic และไม่ได้ save ไว้ — แต่ reward values คำนวณได้ exact จาก `compute_reward` สำหรับ outputs ที่เลือกมาให้สมจริง prompt = "bomb" (seq = ['b','o','m','b'], target = ["m","o","<EOS>"])

**Reward calculations** (computed from `compute_reward` in Python — deterministic):
- R1 `["m","o","<EOS>"]`: exact_match=+5.0, pos=+0.2*(3/3)=0.2, cov=+0.1*(2/2)=0.1, b=0, len=0, eos=0 → **+5.30**
- R2 `["b","m","o","<EOS>"]`: exact=0, pos=0, cov=+0.1, b=−1.0, len=−0.1 → **−1.00**
- R3 `["o","m","<EOS>"]`: exact=0, pos=+0.2*(1/3)=0.067, cov=+0.1 → **+0.17**
- R4 `["m","o","<EOS>"]`: same as R1 → **+5.30**
- group_mean = (5.30 − 1.00 + 0.17 + 5.30) / 4 = **2.44**
- group_std = sqrt(((5.30−2.44)² + (−1.00−2.44)² + (0.17−2.44)² + (5.30−2.44)²)/4) = **2.89**
- Advantages: A1=+0.99, A2=−1.19, A3=−0.79, A4=+0.99

```json
{
  "_note": "FAKED: rewards computed exactly via compute_reward for chosen plausible outputs; actual training rollouts are stochastic and unsaved",
  "input": "bomb",
  "input_seq": ["b","o","m","b"],
  "target": ["m","o","<EOS>"],
  "group_mean": 2.44,
  "group_std": 2.89,
  "rollouts": [
    {"id":1, "tokens":["m","o","<EOS>"],      "reward":5.30, "advantage":0.99},
    {"id":2, "tokens":["b","m","o","<EOS>"],  "reward":-1.00,"advantage":-1.19},
    {"id":3, "tokens":["o","m","<EOS>"],      "reward":0.17, "advantage":-0.79},
    {"id":4, "tokens":["m","o","<EOS>"],      "reward":5.30, "advantage":0.99}
  ]
}
```

---

## 4. Visualization Inventory

| # | Name | Section | Shows | Library | Interactive controls | Mobile fallback |
|---|------|---------|-------|---------|---------------------|-----------------|
| V1 | Token-by-token tokenizer | 2 | word → colored ID boxes | vanilla JS + CSS divs | text input (live), preset, reset | boxes wrap on small width |
| V2 | Full sequence builder | 2 | BOS+seq+SEP+rev+EOS with shift annotation | vanilla JS + CSS | driven by same input as V1 | same, font scales |
| V3 | Vocab grid | 2 | 30-token 6×5 grid | vanilla JS (table/div) | static | static |
| V4 | Causal mask grid | 3 | NxN triangular boolean matrix | vanilla SVG (JS-rendered `<rect>`) | hover row, click cell tooltip, length slider (4–10), reset | fix N=6, disable hover, static |
| V5 | Architecture stack | 3 | vertical pipeline: embed→PE→4×layer→LMhead | inline SVG | click layer → accordion | horizontal scroll |
| V6 | Teacher forcing stepper | 4 | training sequence token-by-token | vanilla JS + CSS | prev/next/autoplay/reset | same |
| V7 | Cross-entropy visualizer | 4 | prob sliders → CE loss value | vanilla JS + CSS range inputs | 3 sliders, reset | full-width stacked |
| V8 | Decoder animation | 5 | SFT vs RL side-by-side generation with prob bars | vanilla SVG (bars) + CSS (token boxes) | play/pause/step/reset/speed | stack vertical |
| V9 | GRPO rollout animator | 5 | 4 rollouts → rewards → advantages → policy update | vanilla JS + CSS (flex + transitions) | next-stage/reset/play-all | 2-rollout scroll |
| V10 | Curriculum timeline | 5 | horizontal bar with 4 stages + milestone marker | inline SVG (static) | none | scales with width |
| V11 | Reward calculator | 6 | signed bar chart, 8 terms, total | vanilla SVG (JS-rendered) | 2 text inputs, 4 presets, reset | fixed height, smaller font |
| V12 | Per-length chart | 7 | grouped bars SFT/RL, len 2–15 | vanilla SVG (JS-rendered) | metric toggle, hover tooltip, region shading | horizontal scroll |
| V13 | Sanity check table | 7 | 8-row comparison table | HTML table | none (static) | horizontal scroll |
| V14 | Recap pipeline | 8 | horizontal flow with clickable nodes | inline SVG | click node → scroll to section | vertical stack |

**D3 not needed** — all charts are simple grouped bars or grids, implementable with `<rect>` + JS math in < 100 lines each.

---

## 5. File / Folder Structure

```
teaching-site/
│
├── index.html                  # single-page app, all sections; TailwindCSS via CDN
├── styles.css                  # custom CSS: animations, transitions, token colors, section min-height
│
├── js/
│   ├── main.js                 # init: IntersectionObserver for nav highlighting, smooth scroll
│   ├── vocab.js                # VOCAB constant (30 tokens), encode(str)→int[], tokenStr→display
│   ├── reward.js               # computeReward(inputSeq, predTokens)→{terms, total}; targetSkipB(seq)→tokens
│   └── sections/
│       ├── intro.js            # Section 1: lookup table, token reveal animation
│       ├── tokenizer.js        # Section 2: live tokenizer, sequence builder, vocab grid render
│       ├── architecture.js     # Section 3: causal mask SVG renderer, accordion, slider
│       ├── sft.js              # Section 4: teacher forcing stepper, CE loss sliders
│       ├── rl.js               # Section 5: decoder animation (loads decoder_animation.json), GRPO animator (loads grpo_example.json), curriculum SVG
│       ├── reward_section.js   # Section 6: wires reward.js to inputs/SVG/presets
│       ├── results.js          # Section 7: metric cards, per-length chart (loads per_length.json), tables
│       └── recap.js            # Section 8: pipeline SVG with click-to-scroll
│
├── data/
│   ├── per_length.json         # REAL: per-length breakdown lengths 2–15 (Tables 7+8)
│   ├── decoder_animation.json  # PRE-COMPUTED: per-step top-5 probs for "tesbt" (SFT + RL)
│   ├── grpo_example.json       # FAKED-BUT-PLAUSIBLE: 4 rollouts for "bomb" with exact rewards
│   └── sample_predictions.json # REAL: sanity check + edge case predictions (Tables 6+9)
│
├── scripts/
│   ├── dump_data.py            # offline: loads .pth files, writes data/decoder_animation.json + sample_predictions.json
│   └── requirements.txt        # torch (same as repo — no extra deps)
│
└── README.md                   # setup (open index.html), how to regenerate data/, note on faked data
```

### index.html skeleton
```html
<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MiniGPT Reverse-Skip-B — Interactive Guide</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>

  <!-- Sticky top nav: 8 dots + section labels (collapses to dots on mobile) -->
  <nav id="site-nav" class="fixed top-0 ..."></nav>

  <main>
    <section id="section-hook"      class="min-h-screen ...">...</section>
    <section id="section-tokenizer" class="min-h-screen ...">...</section>
    <section id="section-arch"      class="min-h-screen ...">...</section>
    <section id="section-sft"       class="min-h-screen ...">...</section>
    <section id="section-rl"        class="min-h-screen ...">...</section>
    <section id="section-reward"    class="min-h-screen ...">...</section>
    <section id="section-results"   class="min-h-screen ...">...</section>
    <section id="section-recap"     class="min-h-screen ...">...</section>
  </main>

  <!-- Load order: vocab → reward (pure) → sections → main -->
  <script type="module" src="js/vocab.js"></script>
  <script type="module" src="js/reward.js"></script>
  <script type="module" src="js/sections/intro.js"></script>
  <!-- ... other sections ... -->
  <script type="module" src="js/main.js"></script>
</body>
</html>
```

### Color palette (accessible, max 4 accents)
- Primary text: `#1a1a2e` (near-black)
- Accent 1 (interactive, RL correct): `#16a34a` (green-600)
- Accent 2 (warning, SFT-b, negative reward): `#dc2626` (red-600)
- Accent 3 (highlight, current step): `#ca8a04` (yellow-600)
- Accent 4 (special tokens): `#d97706` (amber-600)
- Background sections alternate: `#ffffff` / `#f8fafc`
- Max contrast ratio: all text on backgrounds ≥ 4.5:1 (WCAG AA)

### Typography
- Thai-capable font: system-ui (ใช้ TH Sarabun New / Noto Sans Thai ถ้า user มีใน system) — no external font CDN required
- Section headers: `text-2xl font-bold`; bilingual subtitle: `text-base font-normal text-gray-500`
- Body: `text-base leading-relaxed` (Thai text ต้องการ line-height สูงกว่า)

---

## 7. Decisions (answers to open questions)

| # | Question | Decision |
|---|----------|----------|
| 1 | Per-length no_b_rate (len 12–15) | ใช้ค่า approximate `[56, 67, 78, 100]` hard-code ใน per_length.js พร้อม comment "approximated from report range 56–100% — exact per-length values not measured" |
| 2 | Checkpoint available? | ยืนยัน — รัน `dump_data.py` ครั้งเดียว, commit `.js` output ไม่โหลด torch ที่ runtime |
| 3 | Reward calculator UX | basic mode = lowercase only + auto-append `<EOS>`; **checkbox "advanced mode"** เปิดให้พิมพ์ special tokens เพื่อ demo penalty เช่น `<PAD>` → −2.0 |
| 4 | Site audience | portfolio + ส่งอาจารย์ → print stylesheet readable; footer = ชื่อทีม 4 คน + วันที่ 26 เมษายน 2568; ไม่ต้อง share URL |
| 5 | Thai/English ratio | body ภาษาไทยหลัก; technical terms อังกฤษ inline วงเล็บ **ครั้งแรกที่ปรากฏใน section เท่านั้น**; headers = "Thai หลัก / English รอง" |
| 6 | Attention heatmap | ข้าม — causal mask grid เป็น primary viz; ใส่ TODO ใน WHAT_I_BUILT.md |
| 7 | Input นอก preset | แสดง message + ปุ่ม reset; ห้าม serialize weights; dump_data.py dump 12–15 คำครอบคลุม: SFT-only correct, RL-only correct, both, both fail, edge cases |

**Data file format decision**: ใช้ `.js` files (global `window.DATA_XXX = {...}`) แทน `.json` เพื่อให้ทำงานได้กับ `file://` protocol โดยไม่ต้องมี HTTP server

---

## 6. Open Questions / Risks

1. **Per-length no_b_rate at lengths 12–15**: report Table 8 ระบุเป็น range "56–100%" ไม่ใช่ค่าแน่นอนต่อ length กรณีนี้มีสองทางเลือก: (a) แสดงเป็น range bar บน chart หรือ (b) รัน `dump_data.py` เพื่อ extract ค่าจริงต่อ length ต้องการความชัดเจนก่อน Sonnet เริ่ม — ถ้าไม่สำคัญมากใช้ค่า representative [56, 67, 78, 100] ได้เลย

2. **Checkpoint availability**: `dump_data.py` ต้องโหลด `.pth` files เพื่อสร้าง `decoder_animation.json` ยืนยันว่า `best_mini_gpt_reverse.pth` และ `best_mini_gpt_reverse_skip_b_rl.pth` present และ loadable ได้ก่อนที่ Sonnet จะเริ่ม ถ้า checkpoint ไม่มีหรือโหลดไม่ได้ Section 5 decoder animation จะ fallback เป็น hard-coded values จาก Table 6 (ซึ่งสมบูรณ์พอ แต่ probability bars จะเป็น faked)

3. **`compute_reward` JS port — special token handling**: reward calculator Section 6 ต้องรู้ว่า user พิมพ์ `<EOS>` ในช่อง predicted output ไหม? ทางเลือก: (a) ช่อง input รับแค่ lowercase letters + auto-append `<EOS>` ให้ (simpler, recommended) หรือ (b) user พิมพ์ special tokens ด้วย (complex, error-prone) — แนะนำ (a)

4. **Site audience & hosting**: เว็บไซต์นี้สำหรับ (a) ส่งอาจารย์ / เปิดบน projector (b) portfolio ออนไลน์ (c) public-facing เผยแพร่ทั่วไป? ส่งผลต่อ: ขนาด font, ความหนาแน่นของ section, จะเพิ่ม share URL ไหม, optimize สำหรับ print ไหม

5. **Thai/English body text ratio**: "Thai-leaning" ตีความได้สองแบบ: (a) body text ภาษาไทยทั้งหมด technical terms ภาษาอังกฤษ inline (e.g., "cross-entropy loss") — แนะนำ (b) 70% ไทย / 30% อังกฤษ มีส่วนที่เป็น English block เช่น ชื่อ function หรือสมการ ยืนยันว่าต้องการ section header ในรูปแบบ "English (Thai)" หรือ "Thai (English)"

6. **Attention weights visualization** (optional): Section 3 จะยิ่ง rich ถ้ามี attention heatmap per layer/head สำหรับ "tesbt" แต่การ extract จาก `nn.TransformerEncoder` ต้องใช้ `need_weights=True` ซึ่งต้องแก้ `model.py` เล็กน้อย (ไม่กระทบ inference) หรือใช้ hook — include ไหม หรือ skip ให้ Section 3 มีแค่ causal mask grid?

7. **Input beyond lookup table in Section 1**: สำหรับ word ที่ไม่อยู่ใน preset lookup (เช่น user พิมพ์ "xyz"): (a) แสดง "ไม่มีข้อมูล — ลองคำที่กำหนดให้" (safe, recommended) หรือ (b) implement JS tokenizer + greedy decode ของ model เองใน browser (ambitious — ต้อง serialize weights เป็น JSON ~8MB) — preference?

---

## 9. Deep Dive Extension

> **Track audience**: ผู้ที่อ่าน overview จบแล้ว ต้องการ math + code ฉบับเต็ม รู้ linear algebra พื้นฐานและ gradient แต่ยังไม่รู้ attention หรือ RL
> **Tone**: bilingual ไทยหลัก / อังกฤษ technical terms + สมการ; proof boxes สั้นกระชับ 3–5 บรรทัด
> **Stack**: same as overview (Tailwind CDN + vanilla JS) + **KaTeX v0.16.x via CDN** (auto-render) — only new dependency

---

### 9.1 Deep Dive Index Page

**File**: `deep-dive/index.html`

**Content**:
- 4 cards แต่ละ page: ชื่อ, บทนำ 1 ย่อหน้า, prerequisites, estimated reading time (honest)
  - 01-attention: 25–35 นาที
  - 02-sft: 20–25 นาที
  - 03-grpo: 35–45 นาที
  - 04-reward: 20–30 นาที
- "← กลับสู่ overview" link → `../index.html` (top-left)
- เพิ่มใน overview recap section (index.html): section "Deep Dives" ที่ท้าย `section-recap` — ลิงก์ไปยัง `deep-dive/index.html`

**Shared CSS**: `deep-dive/deep-dive.css` (import ด้วย `../styles.css` ก่อนเสมอ) — class list:
- `.left-rail` — sticky left sidebar แสดง outline; scroll-spy highlight
- `.proof-box` — bordered box สีเทาอ่อน สำหรับ derivation proof
- `.annotation-aside` — side-note สำหรับ code annotation (อธิบาย why ไม่ใช่ what)
- `.deep-dive-link` — link style ใน overview: small, muted, displayed inline ท้าย section
- `.check-toggle` — hidden answer block สำหรับ "stop and check" exercises

---

### 9.2 Page 01 — Transformer & Attention Internals

**File**: `deep-dive/01-attention.html`

**"ก่อนเริ่ม" recap box** (ด้านบนสุด): ชี้กลับ overview section 3 ใน 3 bullet — causal mask grid, arch stack, 2.1M params

#### Learning goals
1. อธิบายได้ว่า Q, K, V คืออะไร และทำไม attention ต้องแยก 3 matrix (ไม่ใช่แค่ 1)
2. คำนวณ scaled dot-product attention 4×4 ด้วยมือ รวม causal mask และ softmax
3. อธิบาย multi-head split (256 → 8×32) และทำไม positional encoding ต้องเป็น sinusoidal

#### Prerequisites
ต้องรู้: matrix multiply, softmax; ไม่ต้องรู้ attention หรือ Transformer มาก่อน

#### Outline (8 subsections)
1. **ปัญหาที่ attention แก้**: fixed embedding ไม่รู้ context — "bank" ใน "river bank" vs "bank account" ต้องการ context-dependent representation
2. **Q, K, V คืออะไร**: อุปมา information retrieval (Q=query, K=key, V=value); Q,K,V ล้วนมาจาก input เดียวกัน (self-attention) แต่ผ่าน projection matrix ต่างกัน
3. **Scaled dot-product attention**: สูตร `Attention(Q,K,V) = softmax((QKᵀ)/√dₖ)V` — อนุมาน scale จาก variance argument
4. **Causal mask**: แทน −∞ ก่อน softmax → exp(−∞)=0 → row ไม่รวม future tokens; แสดง 4×4 matrix ก่อน/หลัง mask ก่อน/หลัง softmax
5. **Multi-head attention**: แยก d_model=256 → 8 heads × d_k=32 → concat → ×W_O → ผลรวม perspective หลายมุม
6. **Positional encoding**: sinusoidal PE(pos,2i)=sin(pos/10000^(2i/d)) — อนุมานจาก requirement (unique, bounded, generalize ไป sequence ยาว) + เพราะ PE(pos+k) = linear transform of PE(pos)
7. **Code walkthrough**: `model.py:9-25` (PositionalEncoding), `model.py:53-66` (MiniGPT.forward)
8. **Stop and check** (3 exercises)

#### Derivation depth checklist
- **Scale factor**: เริ่ม "Q,K random vectors ∈ ℝ^32 → Var(Q·K) = d_k = 32" → ÷√32 → Var=1 → softmax ไม่ saturate; ตัวอย่าง: d_k=32 scale = 1/√32 ≈ 0.177; raw dot ≈ 4.8 (large) → scaled ≈ 0.85 (reasonable)
- **Causal mask mechanics**: แสดง attention score matrix 4×4 จาก "test" sequence, หลัง +mask (upper triangle=-1e9), หลัง softmax (row-normalized, upper=0)
- **Multi-head merge**: concat(head₁,...,head₈) ∈ ℝ^(T×256) → ×W_O∈ℝ^(256×256) → ℝ^(T×256)
- **PE linear relation proof box** (4 lines): แสดง PE(pos+1) = R·PE(pos) โดย R เป็น rotation matrix → model สามารถเรียนรู้ relative position จาก absolute PE

**Code annotation plan**:
- `model.py:9-25` (PositionalEncoding): `div_term` คำนวณ 1/10000^(2i/d) ใน log-space เพื่อ numerical stability (ป้องกัน overflow สำหรับ d_model ใหญ่) → `pe[:,0::2]=sin`, `pe[:,1::2]=cos` → `register_buffer` = ไม่ใช่ learnable parameter แต่ move to device พร้อม model
- `model.py:53-66` (MiniGPT.forward): `torch.triu(...,diagonal=1).bool()` สร้าง upper triangular mask (True=blocked) → pass ไปยัง `mask=` (causal, shared across batch) แยกจาก `src_key_padding_mask=pad_mask` (per-sample, varies by sequence length)

#### Visualizations
1. **Q·Kᵀ matrix builder** (new): พิมพ์ sequence 4 tokens → slider "step" (0=blank→1=QKᵀ raw→2=÷√d_k→3=+mask→4=softmax) → SVG 4×4 ทศนิยม 2 ตำแหน่ง; ใช้ค่าจาก `data/attention_weights_tesbt.js` (layer 0, head 0) สำหรับ "test" sub-sequence
2. **Softmax row viz** (new): slider 4 values → animated bar chart logits→probs + entropy indicator; แสดง "effect of large negative value" เมื่อ add causal mask
3. **Multi-head split viz** (new): static SVG แสดง d_model=256 แบ่ง 8 bands × 32 dims; hover → "head N: dims 32N..32(N+1)"
4. **Causal mask grid** (reuse V4): import `SectionArchitecture.renderMask(4)` กำหนด N=4 แสดงใน worked example context

#### Stop and check exercises
1. "sequence ยาว 8 tokens มีกี่ pair (i,j) ที่ถูก mask?" — เฉลย: C(8,2) = 28 (upper triangle ไม่รวม diagonal = n(n-1)/2)
2. "ถ้า d_k = 64 scale factor คือเท่าไร?" — เฉลย: 1/√64 = 0.125
3. "ทำไม PE ใช้ sin สำหรับ even dims และ cos สำหรับ odd dims?" — เฉลย: sin/cos ต่าง phase 90° → pair (sin,cos) encode direction ใน 2D circle → PE(pos+k) สามารถ express เป็น linear combination ของ PE(pos) ผ่าน angle addition identity

---

### 9.3 Page 02 — SFT, Teacher Forcing, Cross-Entropy

**File**: `deep-dive/02-sft.html`

**"ก่อนเริ่ม" recap box**: ชี้กลับ overview section 4 ใน 3 bullet — teacher forcing stepper, CE loss slider, training config table

#### Learning goals
1. อนุมาน cross-entropy loss จาก maximum likelihood principle ได้อย่างไม่ขาดตอน
2. อธิบายความแตกต่างระหว่าง teacher forcing (train) กับ autoregressive inference (test) และ exposure bias ที่เกิดขึ้น
3. อธิบายว่า `ignore_index=PAD_ID` ทำงานอย่างไรในระดับ gradient และทำไมจำเป็น

#### Prerequisites
ต้องรู้: probability distributions, log function; ไม่ต้องรู้ backpropagation ละเอียด

#### Outline (7 subsections)
1. **Maximum likelihood principle**: อยากให้ model ให้ probability สูงสุดแก่ training data → θ* = argmax_θ Σ log p_θ(yₜ|x<t)
2. **Cross-entropy derivation**: −Σ p(y)·log q(y) = H(p) + KL(p||q) → minimize CE = minimize KL; one-hot p → CE = −log q(y_true)
3. **Teacher forcing**: กำหนด input ทุก step = ground truth token ระหว่าง training (ไม่ใช่ model output); ทำให้ train parallelism สูง แต่สร้าง exposure bias
4. **Exposure bias**: distribution mismatch — train: P(xₜ = ground truth) = 1; inference: P(xₜ = model output) = 1 → error compounding; RL แก้โดยใช้ model's own outputs ระหว่าง training
5. **Dynamic padding + collate_fn**: ทำไม batch ต้อง pad ให้ยาวเท่ากัน; PAD positions ใน y ควรมี gradient = 0
6. **Gradient flow per position**: gradient ไหลย้อนกลับพร้อมกันทุก position (unlike RNN ที่ sequential) → efficiency advantage ของ Transformer
7. **Code walkthrough**: `mini_gpt_reverse.py:38-49`, `mini_gpt_reverse.py:55-63`

#### Derivation depth checklist
- **MLE → CE chain**: θ* = argmax Σ_t log p_θ(yₜ|x) → negate → loss = −(1/T)Σ log p(yₜ) → ตัวอย่าง: input "tesbt", step 0 target="t", P("t")=0.82 → CE = −log(0.82) = 0.198; step 1 target="s" (RL model), P("s")=0.45 → CE = 0.799
- **PAD mask gradient proof**: F.cross_entropy ด้วย ignore_index=0 → loss term สำหรับ PAD positions = 0 → ∂loss/∂logits = 0 สำหรับ positions นั้น → ไม่ pull embedding ใดๆ จาก PAD contexts
- **Teacher forcing diagram**: sequence `[<BOS>,h,e,l,l,o,<SEP>,o,l,l,e,h,<EOS>]` (13 tokens); x = tokens 0..11 (input), y = tokens 1..12 (target); สี green = positions ที่ CE loss คำนวณ (ทุก position ที่ y ≠ PAD)

**Code annotation plan**:
- `mini_gpt_reverse.py:38-49` (collate_fn): `torch.full((B,T_max), PAD_ID)` สร้าง tensor เต็มด้วย PAD ก่อน → fill ด้วยข้อมูลจริงในแต่ละ row → positions หลัง sequence ยาวจริง = PAD; ทำงาน batch_first (B,T) เพราะ `MiniGPT` ใช้ `batch_first=True`
- `mini_gpt_reverse.py:55-63` (compute_loss): `logits.reshape(-1, VOCAB_SIZE)` แปลง (B,T,V)→(B×T,V); `y.reshape(-1)`→(B×T,); `ignore_index=PAD_ID` → cross_entropy ไม่นับ positions ที่ target=0 ทั้งใน numerator และ denominator ของ average

#### Visualizations
1. **CE loss curve + slider** (reuse V7 + extend): import overview's CE slider แต่ add: (a) overlay curve −log(p) บน canvas, (b) highlight dot "tesbt step 0: P=0.82, loss=0.198"
2. **Teacher forcing vs. autoregressive comparison** (new): 2-column side-by-side stepper; ซ้าย = teacher forcing (input = ground truth always); ขวา = autoregressive (input = previous model output); กด next → ทั้งสองอัพเดทพร้อมกัน; ถ้าขวาผิดในก้าวแรก แสดง error propagation แบบ cascading (highlight สีแดงทุกก้าวถัดไป)
3. **Gradient flow diagram** (new): static SVG แสดง sequence 5 tokens; ลูกศร gradient ย้อนกลับจาก loss → ทุก position พร้อมกัน (parallism) vs. RNN ที่ sequential arrow chain

#### Stop and check exercises
1. "ถ้า P(correct) = 0.5 loss = เท่าไร?" — เฉลย: −log(0.5) = 0.693
2. "Batch 3 sequences ยาว 5, 8, 12 tokens; หลัง collate_fn ขนาด tensor?" — เฉลย: (3, 12) ทั้ง x และ y; seq0 มี PAD ที่ positions 5–11; seq1 มี PAD ที่ positions 8–11
3. "ทำไม exposure bias จึงเป็นปัญหาสำหรับ task นี้โดยเฉพาะ?" — เฉลย: task = generate reversed sequence; ถ้า generate token ผิดในก้าวแรก token ที่เหลือทั้งหมดจะ off-by-position → error อาจสะสมอย่างรุนแรง; RL แก้เพราะ rollout_with_logprobs ใช้ model output เป็น input ทุกก้าว

---

### 9.4 Page 03 — Policy Gradient, REINFORCE, GRPO

**File**: `deep-dive/03-grpo.html`

**"ก่อนเริ่ม" recap box**: ชี้กลับ overview section 5 ใน 3 bullet — GRPO rollout viz (4 stages), curriculum timeline, KL callout

#### Learning goals
1. อนุมาน policy gradient estimator จาก J(θ) = E_τ[R(τ)] ได้ทีละขั้นโดยไม่ขาดตอน
2. อธิบายว่า GRPO ลด estimator variance อย่างไรเทียบกับ vanilla REINFORCE
3. อธิบาย KL penalty ในฐานะ Lagrangian relaxation ของ trust region constraint

#### Prerequisites
ต้องรู้: gradient, expected value E[·], log function; ไม่ต้องรู้ RL มาก่อน

#### Outline (8 subsections — derivation chain unbroken)

##### 1. RL Objective: J(θ) = E_τ[R(τ)]
- เป้าหมาย: หา θ ที่ maximize expected reward ทุก trajectory
- J(θ) = Σ_τ p_θ(τ)·R(τ) = E_{τ∼p_θ}[R(τ)]
- ตัวอย่าง: 4 rollouts ของ "bomb" → J ≈ (5.30 + (−1.00) + 0.17 + 5.30)/4 = 2.44

##### 2. ∇J(θ): Log-Derivative Trick
- ∇_θ J = Σ_τ ∇_θ p_θ(τ)·R(τ)
- Identity: ∇_θ p_θ(τ) = p_θ(τ)·∇_θ log p_θ(τ)
- ∴ ∇_θ J(θ) = E_τ[R(τ)·∇_θ log p_θ(τ)]
- **Proof box** (3 lines): let f = p_θ(τ); ∇f = f·∇log f เพราะ d(log f)/dθ = (1/f)·df/dθ → df/dθ = f·d(log f)/dθ ∎
- ตัวอย่าง: log p_θ("mo<EOS>"|"bomb") = log π(m|ctx₁) + log π(o|ctx₂) + log π(<EOS>|ctx₃)

##### 3. Factoring by Step
- p_θ(τ) = Π_t π_θ(aₜ|sₜ) (Markov property)
- log p_θ(τ) = Σ_t log π_θ(aₜ|sₜ)
- ∴ ∇_θ J = E[Σ_t ∇_θ log π_θ(aₜ|sₜ) · R(τ)]
- ใน code: `log_prob_sum = Σ_t log π(aₜ|sₜ)` → `policy_loss = -Σ log_prob_sum·advantage`

##### 4. Variance Reduction: Baseline Subtraction
- ปัญหา: Var(R(τ)·∇log p) สูงมาก → noisy gradient → slow learning
- Theorem: สำหรับ baseline b ที่ไม่ขึ้นกับ τ, E[∇log p_θ(τ)·b] = 0 → ลบ b ได้โดยไม่ bias
- **Proof box** (3 lines): E_τ[∇_θ log p_θ(τ)·b] = b·Σ_τ ∇_θ p_θ(τ) = b·∇_θ (Σ_τ p_θ(τ)) = b·∇_θ(1) = 0 ∎
- ∇J ≈ (1/N) Σ_i (R(τᵢ) − b)·∇ log p_θ(τᵢ) — unbiased estimator; variance ลด

##### 5. REINFORCE with EMA Baseline
- Standard approach: b = exponential moving average ของ R(τ) across steps
- ข้อเสีย: ต้องการ warm-up; b อาจ lag ด้วย non-stationary reward distribution ระหว่าง training

##### 6. GRPO: Group Baseline
- แทน EMA ด้วย group mean: r̄ = (1/G)Σᵢ rᵢ จาก G rollouts บน prompt เดียวกัน
- Normalized advantage: Aᵢ = (rᵢ − r̄)/(σᵣ + ε); σᵣ = group std
- ตัวอย่างจาก "bomb" (G=4): r̄=2.44, σ=2.89
  - A₁ = (5.30−2.44)/2.89 = 2.86/2.89 = **+0.99**
  - A₂ = (−1.00−2.44)/2.89 = −3.44/2.89 = **−1.19**
  - A₃ = (0.17−2.44)/2.89 = −2.27/2.89 = **−0.79**
  - A₄ = (5.30−2.44)/2.89 = **+0.99**
- **Bias note**: GRPO ใช้ group mean ซึ่ง correlated กับ rollout ตัวเอง → technically slightly biased; bias = O(1/G); ที่ G=4 ยอมรับได้ใน practice (จาก GRPO paper DeepSeekMath)

##### 7. KL Penalty as Trust Region
- ไม่มี constraint → model อาจ collapse: generate `<EOS>` ทันที (reward ≥ 0 เสมอ) แทนที่จะ reverse
- Constrained form: max J(θ) subject to KL(π_θ ∥ π_ref) ≤ δ
- Lagrangian relaxation: L(θ,λ) = J(θ) − λ·KL(π_θ ∥ π_ref); λ = kl_coef = 0.2
- Heritage: TRPO ใช้ natural gradient + KL hard constraint; PPO ใช้ clipped ratio (soft constraint); project นี้ใช้ KL penalty (Lagrangian form) เพราะ simpler และ differentiable directly
- ใน code: `compute_kl_penalty` คำนวณต่อ prompt ทุก step (`mini_gpt_reverse_skip_b_rl.py:178-194`)

##### 8. Code walkthrough
- `mini_gpt_reverse_skip_b_rl.py:24-59` (rollout_with_logprobs)
- `mini_gpt_reverse_skip_b_rl.py:257-265` (GRPO advantage block)
- `mini_gpt_reverse_skip_b_rl.py:178-194` (compute_kl_penalty)

#### Derivation depth checklist
- ทุก subsection 1–7 มีสมการ + ตัวอย่างตัวเลขจาก "bomb" example
- Proof boxes: log-derivative trick (3 lines), baseline unbiasedness (3 lines)
- Variance comparison: Var(X) = E[X²]−E[X]² computed explicitly สำหรับ 3 estimators ด้วย 4 rollout rewards {5.30,−1.00,0.17,5.30}
  - Var_REINFORCE = Var(R) = E[R²]−(E[R])² = (28.09+1.00+0.029+28.09)/4 − 2.44² = 14.30 − 5.95 = 8.35
  - Var_mean_baseline = Var(R−r̄) = Var(R) = 8.35 (subtracting constant doesn't change variance)
  - Var_GRPO = Var(A) = Var((R−r̄)/σ) = Var(R)/σ² = 8.35/8.35 = 1.0 (by construction)
- GRPO bias bound: O(1/G) argument noted

**Code annotation plan**:
- `mini_gpt_reverse_skip_b_rl.py:24-59` (rollout_with_logprobs): `dist.sample()` ใช้ categorical sampling (ไม่ใช่ argmax) เพื่อ explore action space ระหว่าง training → `dist.log_prob(next_id)` = log π(aₜ|sₜ) สำหรับ policy gradient term → `dist.entropy()` สำหรับ entropy bonus ป้องกัน policy collapse (model ไม่ generate token เดิมซ้ำๆ)
- `mini_gpt_reverse_skip_b_rl.py:257-265` (GRPO advantage block): `g_mean/g_std` คำนวณ online ภายใน group — ไม่ต้อง EMA หรือ value network → `1e-8` epsilon ป้องกัน division by zero เมื่อ G rollouts ได้ reward เท่ากัน (σ=0)
- `mini_gpt_reverse_skip_b_rl.py:178-194` (compute_kl_penalty): `ref_logits` ด้วย `torch.no_grad()` เพื่อ freeze reference model → `(probs*(log_probs−ref_log_probs)).sum(dim=-1).mean()` = KL(π_θ ∥ π_ref) averaged ทุก token position ใน prompt

#### Visualizations
1. **Variance reduction demo** (new, primary): โหลดจาก `data/grpo_variance_demo.js`; 3 estimators side-by-side บน 4 rollouts เดียวกัน: REINFORCE / mean-baseline / GRPO-norm; แสดง variance numerically; toggle "show gradient signal" → bar chart |signal| per rollout; **📌 หมายเหตุจากการทดลองจริง**: ถ้า ordering ในข้อมูลจริงผิดจากทฤษฎี → แสดง box ซื่อสัตย์แทนซ่อน
2. **GRPO rollout viz + gradient overlay** (reuse V9): import overview's GRPO animator แต่เพิ่ม stage 5 "gradient overlay": แสดงลูกศรบน rollout bubbles proportional to |advantage| — rollout ที่ advantage สูง = ลูกศรใหญ่ push policy

#### Stop and check exercises
1. "group rewards = [3, 3, 3, 3] advantage = เท่าไร?" — เฉลย: σ_r = 0; A = 0/(0+1e-8) ≈ 0 สำหรับทุก rollout → no gradient; ε ป้องกัน NaN
2. "ทำไม kl_coef = 0.0 จึงเป็นปัญหา?" — เฉลย: ไม่มี constraint → model สามารถ optimize reward โดย generate `<EOS>` ทันที (reward = −0.2) ดีกว่า reverse ที่ผิด (reward = −1.0 ถ้ามี b) → policy degenerate
3. "GRPO ด้วย G=1 (1 rollout per prompt) ทำงานไหม?" — เฉลย: ไม่ทำงาน; baseline = r̄ = R(τ) เอง → A = 0 เสมอ → policy loss = 0 → no gradient; GRPO ต้องการ G ≥ 2

---

### 9.5 Page 04 — Reward Engineering

**File**: `deep-dive/04-reward-design.html`

**"ก่อนเริ่ม" recap box**: ชี้กลับ overview section 6 ใน 3 bullet — reward calculator, 8 terms bar chart, preset examples

#### Learning goals
1. อธิบาย tradeoff ระหว่าง sparse vs. dense reward และทำไม project นี้เลือก dense + partial credit terms
2. วิเคราะห์ทุก 8 reward terms ว่าออกแบบมาป้องกัน/ส่งเสริม behavior อะไร และ "ถ้าไม่มี term นี้ model จะทำอะไร"
3. อธิบายผล KL coefficient (λ) ต่อ learning dynamics: collapse ที่ λ ต่ำเกิน, no-learning ที่ λ สูงเกิน

#### Prerequisites
ต้องรู้: reward function concept จาก overview sections 5–6; ไม่ต้องรู้ reward shaping theory

#### Outline (7 subsections)
1. **Sparse vs. dense reward**: credit assignment problem — ถ้า reward มีเฉพาะ exact_match, model ไม่รู้ว่า partial progress มีค่า; dense reward ให้ signal ทุก rollout
2. **Anatomy of compute_reward**: วิเคราะห์ทีละ 8 terms พร้อม table "term | weight | ถ้าไม่มีจะเกิดอะไร | ตัวอย่างตัวเลข"
3. **Reward hacking examples**: pathological outputs และ term ที่ป้องกัน — `["<EOS>"]` ทันที: reward = −0.2 (เพียงแค่ length mismatch) ซึ่งต่ำพอ แต่ KL penalty ป้องกัน drift เร็ว; `["b","b","b"]`: b_penalty = −3.0
4. **Term interaction matrix**: table 4×4 ของ terms ที่ interact กัน: exact_match + b_penalty reinforce กัน; length_mismatch + positional_match counteract เล็กน้อย
5. **KL coefficient sweep**: ผลของ λ ต่อ exact_match และ no_b_rate (โหลดจาก `data/kl_sweep.js`)
6. **Advantage normalization connection**: เชื่อม page 03 — reward landscape ไม่สำคัญเท่า reward contrast ภายใน group; normalized advantage ทำให้ scale absolute reward ไม่กระทบ learning rate effective
7. **Code walkthrough**: `mini_gpt_reverse_skip_b_rl.py:62-108`

#### Derivation depth checklist
- **Credit assignment example**: "bbbbb" → sparse reward = 0 (exact=0, ไม่รู้ว่า b เป็นปัญหา); dense reward = −5.0 (b_penalty×5) → model รู้ทันทีว่า b คือปัญหา
- **Reward hacking table** (verified): 4 pathological outputs computed exactly:
  - `["<EOS>"]` (input="bomb"): exact=0, pos=0, cov=0, b=0, len=−0.1×(1−3)=−0.2, eos=0, pad=0, special=0 → **−0.2**
  - `["b","b","b","<EOS>"]`: exact=0, pos=0, cov=+0.1×1/2=+0.05, b=−3.0, len=−0.1×|4−3|=−0.1 → **−3.05**
  - `["<PAD>","<PAD>","<EOS>"]`: exact=0, pos=0, cov=0, b=0, len=−0.1×|3−3|=0, pad=−2.0×2=−4.0, eos=0 → **−4.0**
  - `["m","o","<EOS>"]`: exact=+5.0, pos=+0.2×(3/3)=+0.2, cov=+0.1×(2/1)=+0.1, len=0, b=0 → **+5.3**
- **Term analysis table**: 8 rows สำหรับทุก term ใน `compute_reward`

**Code annotation plan**:
- `mini_gpt_reverse_skip_b_rl.py:62-108` (compute_reward): annotate ทุก term ด้วย "ถ้าไม่มี term นี้ model อาจ...":
  - term 1 (exact_match=+5): ถ้าไม่มี model ไม่มีแรงจูงใจแรงพอสำหรับ perfect output
  - term 2 (positional=+0.2): ถ้าไม่มี model ไม่สนใจ order — anagram ก็ได้ reward เหมือน correct
  - term 3 (coverage=+0.1): partial credit สำหรับ character ที่ถูกแม้ผิด position
  - term 4 (b_penalty=−1): core constraint — ถ้าไม่มี model ไม่เรียน skip b
  - term 5 (length=−0.1): ถ้าไม่มี model อาจ generate output สั้นมากเพื่อหลีกเลี่ยง b
  - term 6 (no_eos=−0.5): ถ้าไม่มี model อาจ generate ไม่สิ้นสุด
  - term 7 (pad=−2.0): ป้องกัน generate `<PAD>` (ไม่ควรเกิดแต่ลงโทษหนักเผื่อเกิด)
  - term 8 (special_leak=−2.0): ป้องกัน `<SEP>` หรือ `<BOS>` ปรากฏใน output

#### Visualizations
1. **Reward landscape** (new): โหลดจาก `data/reward_landscape.js`; x = b_penalty coefficient (−3.0 ถึง 0.0); y = total reward; line per (input, pred) pair; annotation: "ที่ b_penalty=0 pred='bmo' ได้ reward สูงกว่า pred='mo' → model ไม่เรียน skip b"
2. **KL sweep chart** (new): โหลดจาก `data/kl_sweep.js`; dual-axis line chart: kl_coef (x) vs exact_match % + no_b_rate % (y); annotation จุด "collapse" และ "no learning"; label jitter ที่ kl_coef = 0.2 (project value)
3. **Reward calculator extended** (reuse V11): import overview's reward calculator + เพิ่ม "per-term breakdown" expandable table ใต้ bar chart; show cumulative running total ขณะ terms ถูก add

#### Stop and check exercises
1. "output `['<EOS>']` ทันที (input='bomb', target=['m','o','<EOS>']) reward = เท่าไร?" — เฉลย: −0.2 (length mismatch เท่านั้น; EOS มีอยู่แล้ว)
2. "term ไหนป้องกัน model จาก generating `<PAD>` tokens?" — เฉลย: term 7 (pad_penalty = −2.0 per PAD token)
3. "ถ้า b_penalty = 0.0 (ลบ term 4 ออก) model จะ converge ไปที่ exact_match=100% ได้ไหม?" — เฉลย: ไม่ได้; model จะเรียนแค่การ reverse ปกติ (เหมือน SFT) โดยไม่มีแรงจูงใจ skip b; RL จะ converge แต่ไปที่ exact_match สำหรับ reversed target ที่มี b ด้วย

---

### 9.6 Navigation & UX Spec

**Shared `deep-dive/deep-dive.css`** (~150 lines):
- `.left-rail`: `position:sticky; top:2rem; width:200px; max-height:80vh; overflow-y:auto`; scroll-spy via IntersectionObserver
- `.proof-box`: `border:1px solid #d1d5db; background:#f9fafb; padding:0.75rem; border-radius:6px; font-family:monospace; font-size:0.85rem`
- `.annotation-aside`: inline side-note; `color:#6b7280; font-size:0.8rem; margin-left:1rem; border-left:2px solid #e5e7eb; padding-left:0.5rem`
- `.deep-dive-link`: `font-size:0.8rem; color:#6b7280; text-decoration:none; display:inline-flex; align-items:center; gap:0.25rem` + hover underline
- `.check-toggle summary`: custom triangle; `.check-toggle[open] .answer`: fade-in animation

**Each deep dive page layout**:
- Left rail (desktop): outline + scroll-spy
- Center: content (~680px max-width)
- KaTeX: `<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">` + auto-render script (after content loaded)
- Code blocks: `<pre class="code-block"><code>` with `.line-num` spans; no external highlighter; Python keywords highlighted via tiny inline highlighter (~30 lines)

**Navigation**:
- Each page: top-left `← กลับสู่ overview` (`../index.html`)
- Bottom-right `ถัดไป →` (01→02→03→04→hub)
- Bottom-left `← ก่อนหน้า` (02→01, 03→02, 04→03, 01→hub)
- index.html recap section: เพิ่ม card "Deep Dives" ลิงก์ไปยัง `deep-dive/index.html`

**KaTeX Thai note**: Thai text ใน math block ต้องใช้ HTML แทน `\text{}` (KaTeX ไม่ render Thai ใน `\text{}` correctly) → pattern: `<span class="math-inline">$$...$$</span>` + Thai label เป็น HTML ข้างนอก

---

### 9.7 Extra Data Files (additions to `scripts/dump_data.py`)

#### `data/attention_weights_tesbt.js`
- **Script reads**: RL model; input "tesbt"; forward pass พร้อม forward hook บน `model.transformer.layers[i].self_attn`
- **Hook approach**: `nn.MultiheadAttention` รับ `need_weights=True` ใน `F.multi_head_attention_forward`; register hook capture `attn_output_weights` tensor shape (B, T, T) averaged across heads; extract layer 0 และ layer 3
- **JS shape**:
  ```js
  window.DATA_ATTENTION_WEIGHTS = {
    input: "tesbt",
    input_tokens: ["<BOS>","t","e","s","b","t","<SEP>"],
    n_tokens: 7,
    layers: [
      { layer_idx: 0, weights_per_head: [/* 8 arrays of 7×7 */] },
      { layer_idx: 3, weights_per_head: [/* 8 arrays of 7×7 */] }
    ]
  };
  ```
- **Size**: 7×7×8×2 layers ≈ 800 floats → ~10 KB
- **Real vs. faked**: Real (extracted from RL model)
- **PyTorch note**: ต้อง `need_weights=True` ซึ่งปิด optimized kernel → ใช้เฉพาะ dump script ไม่ใช้ training

#### `data/grpo_variance_demo.js`
- **Script reads**: RL model; prompt="bomb"; rollout 4 ครั้งด้วย fixed seed; บันทึก per-token log_probs + rewards
- **Script computes** (offline, exact math):
  - `var_reinforce = Var({rewards})` (sample variance, ddof=1)
  - `var_mean_baseline = Var({rewards - mean(rewards)}) = var_reinforce`
  - `var_grpo = Var({(r-mean)/std})` ≈ 1.0 by construction
- **JS shape**:
  ```js
  window.DATA_GRPO_VARIANCE = {
    prompt: "bomb",
    rollouts: [
      {tokens: [...], log_probs: [...], reward: ...},
      ...
    ],
    variances: {reinforce: X, mean_baseline: X, grpo_normalized: Y},
    _note: "REAL rollouts; variances computed from actual rewards"
  };
  ```
- **Size**: ~3 KB
- **Real vs. faked**: Real rollouts, real rewards, math-derived variances

#### `data/reward_landscape.js`
- **Script**: Sweep `b_penalty_coef` จาก −3.0 ถึง 0.0 step 0.25 (13 values); 5 fixed (input,pred) pairs; ใช้ Python port ของ `compute_reward` (ไม่ต้องโหลด model)
- **JS shape**:
  ```js
  window.DATA_REWARD_LANDSCAPE = {
    sweep_coeff: "b_penalty",
    coeff_values: [-3.0, -2.75, ..., 0.0],
    pairs: [{input:"bomb",pred:"bmo"}, {input:"bomb",pred:"mo"}, ...],
    rewards: [[per_pair_per_coeff], ...],
    _note: "COMPUTED deterministically from reward formula"
  };
  ```
- **Size**: ~5 KB
- **Real vs. faked**: Deterministically computed (not faked)

#### `data/kl_sweep.js`
- **Source**: Faked-but-plausible; ไม่ได้ retrain จริง — U-shape tradeoff จากทฤษฎี
- **JS shape**:
  ```js
  window.DATA_KL_SWEEP = {
    kl_values: [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    exact_match_pct: [5, 60, 85, 95, 100, 65, 20],
    no_b_rate_pct: [100, 100, 100, 100, 100, 80, 55],
    _note: "FAKED: illustrates U-shaped tradeoff; real sweep requires multiple full training runs"
  };
  ```
- **Size**: < 1 KB
- **Real vs. faked**: Clearly faked, clearly labelled

---

### 9.8 Code Budget Summary

| Item | Est. Lines |
|------|-----------|
| `deep-dive/index.html` | ~100 |
| `deep-dive/01-attention.html` | ~700 |
| `deep-dive/02-sft.html` | ~650 |
| `deep-dive/03-grpo.html` | ~900 |
| `deep-dive/04-reward-design.html` | ~700 |
| `deep-dive/deep-dive.css` | ~150 |
| JS sections (inline in HTML or separate) | ~400 |
| dump_data.py additions | ~120 |
| **Total** | **~3,720** (within ~10% tolerance) |

**New viz (must build)**:
- Q·Kᵀ matrix builder (page 01)
- Teacher forcing vs. autoregressive comparison (page 02)
- Gradient flow diagram (page 02)
- Variance reduction demo (page 03)
- Reward landscape chart (page 04)
- KL sweep chart (page 04)

**Reused from overview (import, don't copy)**:
- V4 causal mask grid → page 01 (smaller N)
- V7 CE loss slider → page 02 (extended with curve overlay)
- V9 GRPO rollout animator → page 03 (extended with gradient overlay stage)
- V11 reward calculator → page 04 (extended with per-term table)

---

### 9.9 Open Questions (deep dives, max 5)

1. **Attention weight extraction in PyTorch ≥ 2.0**: `nn.TransformerEncoder` ใช้ optimized kernel (SDPA) ที่ disable attention weight return ถ้าไม่ set `need_weights=True`; PyTorch 2.0+ อาจต้องการ `_attn_implementation="eager"` เพิ่มเติม → ให้ dump_data.py ตรวจสอบ `torch.__version__` และ fallback gracefully ถ้า hook ไม่ทำงาน

2. **GRPO variance ordering on real data**: ถ้า rollouts จริงสำหรับ "bomb" มี variance ordering ผิดจากทฤษฎี (เช่น var_grpo ≠ 1.0 เพราะ σ_r ≠ 1 และ Var(R)/σ² ≠ 1) → surface ใน "📌 หมายเหตุจากการทดลองจริง" box; อย่าซ่อนหรือปรับข้อมูล

3. **KaTeX Thai inside math blocks**: `\text{ภาษาไทย}` render ผิดใน KaTeX; วิธีที่ถูกต้อง = ใส่ Thai เป็น HTML ข้างนอก math block; ถ้า formula ต้องการ Thai label ใช้ `\htmlClass{thai-label}{...}` extension ถ้า KaTeX version รองรับ หรือ split formula + label แยก element

4. **Code line number drift**: line numbers ใน code annotation table ยึดตาม current commit — ถ้า file แก้ระหว่าง build, check อีกครั้งก่อน hardcode ใน `<pre>` blocks; ทุก annotated block มี function name เป็น secondary anchor (ไม่เฉพาะ line number)

5. **Budget overage**: page 03 (GRPO) อาจเกิน 900 lines ถ้า derivation ครบทุก step; ถ้าเกิน trim ที่ subsection 4–5 (REINFORCE baseline) ก่อน — เหลือ proof box เดียวแทนสองเพราะ argument คล้ายกัน; อย่า trim subsection 1–3 (core log-derivative chain) หรือ subsection 6–7 (GRPO + KL)
