# Text encoder access: gated Llama-3

Opened 2026-09-30 (the blocker was first seen 2026-09-28). **Whose call:** the user.

## Question

ARDY and Kimodo both encode prompts with LLM2Vec
(`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` adapters) on the gated
`meta-llama/Meta-Llama-3-8B-Instruct`. The HF account `jaynye` got a 403 on 2026-09-28,
and still did on 2026-09-30. Without it, neither model can be conditioned on style or
behavior prompts.

## Options

1. Wait for Meta to grant access on the HF account, then cache embeddings for a small
   prompt bank (the 8B encoder runs once, offline).
2. Use access through another account or collaborator, then cache embeddings.
3. Use an ungated mirror of the Llama-3 weights. **Not without explicit user approval:**
   it would bypass the licence gate.
4. Stay text-free: condition only on root paths, keyframes and constraints. Style
   diversity then comes only from sampling noise.

## What is and is not affected

- **Throughput:** prompt content does not affect it. Both encoders emit one pooled
  4,096-d token per prompt, so the denoiser cost does not depend on the prompt (ledger
  D08). What matters is whether guidance is needed.
  - With zero text, ARDY without guidance matched or beat guided sampling on
    constraint-following locomotion
    ([benchmark](../experiments/2026-09-30-motion-backend-benchmark.md)).
  - If style prompts need guidance back, ARDY H40 at 5 steps drops from 2,824 to 1,472
    new-motion-s per wall-s with 2-pass guidance, or 996 with 3-pass.
  - That halves or thirds the break-even `S*`. Test this as soon as prompts work.
- **Quality and diversity:** not measurable until text works. Stylistic variety (brisk,
  elderly, distracted) depends on prompts.
