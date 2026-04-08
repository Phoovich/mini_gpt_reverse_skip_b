# Mini Project - Lecture 1 Foundations of LLMs

---

Step 1 : Training GPT from scratch to reverse a sequence.
Given a sequence of English alphabets X = [x1 , x2 ,... xN]
train a causal language model to generate Y = [xN, xN-1, ..., x1]

<BOS> h e l l o --> MiniGPT --> <BOS> h e l l o <SEP> o l l e h <EOS>

Length of X = 2 - 6 characters.

Note: MiniGPT uses nn.TransformerEncoder with a causal mask (torch.triu),
not nn.TransformerDecoder. The behaviour is equivalent to a decoder.

---

Step 2 : Fine-tune the trained model using reinforcement
learning (GRPO with KL penalty) to make the model skip “b” (bomb)
letter in the reversed output.

<BOS> a b c d e --> MiniGPT (RL) --> ........ e d c a <EOS>
