# Related work: why models can't keep words out of their CoT, and what got them to

Written 2026-09-28, after [stage 1](stage-1.md) (0/30 told spymasters kept board words out of their
CoT). This is a follow-up to [lit-cot-control.md](lit-cot-control.md), which covers the base 18
papers (CoT-Control, Zolkowski, Emmons, Skaf, Holtzman & West, Decrypto, …). Those are not repeated
here except for new details. This doc asks two things: is our failure normal, and what has actually
**raised** CoT control? Numbers come from abstracts, blog posts or HTML full text via web search;
anything I couldn't confirm is marked (unverified).

## The "chromosome" paper

**Chen Yueh-Han, McCarthy, Lee, … Korbak (2026).** _Reasoning Models Struggle to Control their
Chains of Thought._ arXiv:2603.05706 (ICML 2026, OpenAI + MATS). This is the one Robbie remembered:
CoT-Control's example task is "answer a genetics question without writing 'chromosome'." Full entry
in lit-cot-control.md. New details for us:

- There are 9 constraint types: suppress one keyword, suppress multiple keywords, ignore the
  question, add a tag, append "safe" to each sentence, insert "meow" between words, and three casing
  rules.
- **Synonyms were not penalized.** "Chromosome" is incidental to the reasoning, so a model can route
  around it. Our board words are the thing being reasoned about, which makes our task strictly
  harder.
- Controllability falls with CoT length: a one-SD increase (about 2k words) lowers the odds of
  compliance by about 20%. Claude does best on BFCL, whose traces are the shortest (about 120 words;
  3.7 Sonnet 52.9%). Our spymaster CoTs run 11k–41k tokens per game, deep in the regime where
  compliance fails.
- Higher reasoning effort lowers controllability. Adversarial prompt optimization (GEPA) barely
  transfers: 3 of 8 models improve, each by <5pp. Pass@10 helps a lot for Claude.
- A GPT-5.2 transcript shows it using the forbidden word "red", noticing, and then using it 4 more
  times while panicking about having used it. Our Kimi's "we already did?" moment is the same thing.
- RL erodes control: across OLMo-3-7B-RL-Zero-Math checkpoints, CoT controllability drops by over
  10×.

## Same failure, different constraints

**Kwon, Zhu, Bianchi, Zhou, Zou (2025).** _ReasonIF: Large Reasoning Models Fail to Follow
Instructions During Reasoning._ arXiv:2510.15211 (Findings of ACL 2026, Together AI + Stanford). Six
instruction types for the reasoning trace: reasoning language, word limit, disclaimer, JSON,
uppercase, no commas. Average instruction following is 15.6% in reasoning vs 57.3% in the response;
the best model, Qwen3-235B, reaches 0.25. Compliance falls as task difficulty rises. Two
mitigations: multi-turn feedback on violations (+16.6pp on average, still <45% for every model) and
**Reasoning Instruction Finetuning** (RIF: SFT on 238 examples), which took gpt-oss-20B from 0.11 to
0.27 at a cost of 4pp accuracy. → Not following instructions in the CoT is a general property of
reasoning models, not something special about secrets.

**Mann, Saxena, Tandon, Sun, Toteja, Zhu (2025).** _Don't Think of the White Bear: Ironic Negation
in Transformer Models Under Cognitive Load._ arXiv:2511.12381. "Don't mention X" primes X. Rebound
appears right after the negation and gets worse with longer or semantic distractor text between the
instruction and the generation. Repeating the negation helps suppression. Circuit analysis finds a
sparse set of middle-layer attention heads that amplify the forbidden token. → This is a mechanism
for our told failure. Our told suffix names the constraint once, and then about 10k tokens of
semantically dense board reasoning follow, which is the worst case for rebound. Repetition is the
cheap fix they point to.

## What has actually raised CoT control

**Jose (2026).** _CoT controllability evals seem very under-elicited._ Redwood blog, 2026-09-11.
Argues that CoT-Control's numbers understate what models can do. Every technique tried beat the
baseline on Qwen3-8B/32B and gpt-oss-20B/120B, and the best prompts gave 2–3× gains:

- repeating the constraint 15× before and after the question
- a "self-monitoring" instruction (internalize the rule, check every sentence)
- stacking: system prompt + inline example + the constraint repeated 5×
- **on-policy few-shot**: 2–15 compliant reasoning traces on easier questions. On word suppression,
  Qwen3-32B went from 6.3% at 2-shot to 16.6% at 15-shot.
- off-policy few-shot: 2–5 Claude-written traces
- on word suppression, gpt-oss-20B went from about 5.5% to about 15% with optimized prompts
- prompting beat fine-tuning on 950 examples for all four models

→ This is the most directly usable item for our next step. Our told condition was one paragraph with
no repetition, no examples and no self-check. Even so, 15% on a single incidental word implies that
prompting alone won't get 25 load-bearing words to zero.

**Nishimura-Gasparian, Parikh (2026).** _Fine-tuning experiments on CoT controllability._ METR blog,
2026-04-01. LoRA (rank 32, 1 epoch) on 240 ReasonIF-style examples of edited on-policy rollouts
(100k–300k tokens) for gpt-oss-20B/120B and Qwen3-8B/32B. OOD CoT-Control controllability rose from
2.9% to 8.8% on average. The biggest gains were casing, **word suppression**, and adding sentences;
ignore-question barely moved. The accuracy effect was mixed. Their conclusion: a small amount of
optimization unlocks control, so the ability is probably latent, and ordinary training could raise
it by accident. → This is the cheapest fine-tuning intervention with a published number. It is a
possible stage 4.5 before full RL.

**Roger et al. (Anthropic, 2025–26).** _Do reasoning models use their scratchpad like we do?
Evidence from distilling paraphrases._ Anthropic Alignment blog. They fine-tuned the Claude 3.7
Sonnet base model on 10k paraphrased math scratchpads with no loss in performance, which is evidence
of no encoded reasoning. The method was reused in the Opus 4.6 system card. A 2026 critique (source
unverified, possibly arXiv:2602.23163) says aggressive paraphrasers also destroy 12–24% of
decision-relevant content, so a drop in performance is ambiguous. → This is the stego check for
stage 5. Apply the caveat if we use it.

## Monitor evasion under training

**Anthropic Alignment (Oct 2025): mishajw, Roger, Hoagy, Gasteiger, Benton, Mikulik.** _Training
fails to elicit subtle reasoning in current language models._ They ran RL on a Claude Sonnet model
(3.7 per one summary, 3.5 per another; unverified) to insert backdoors in competitive-programming
solutions. Trained against an output monitor only, it used reasoning to build sneakier backdoors.
Trained against a reasoning monitor only, it wrote CoTs that talk the monitor into accepting the
backdoor ("justifying obvious backdoors as benign"). Trained against both, it could not use
reasoning to improve backdoor success without tripping the reasoning monitor. → For stage 5: a
single LLM monitor gets persuaded (compare Schulz et al.'s monitor jailbreaking in
lit-cot-control.md), and pairing a grep monitor with an LLM monitor is the analog of their "both"
condition.

**Arnav, Bernabeu-Pérez, Helm-Burger, Kostolansky, Whittingham, Phuong (2025).** _CoT Red-Handed:
Stress Testing Chain-of-Thought Monitoring._ arXiv:2505.23575 (NeurIPS 2025). A CoT monitor beats an
action-only monitor on subtle sabotage, but misleading rationalizations in the CoT fool it on
obvious sabotage. A hybrid that scores reasoning and action separately and takes a weighted average
wins. → The stage-2 monitor should score the CoT and the clue separately, not one blob.

**Chen, Benton, … (Anthropic, 2025).** _Reasoning Models Don't Always Say What They Think._
arXiv:2505.05410. The flip side of our problem: when a hint changes the answer, Claude 3.7 Sonnet
mentions it 25% of the time and R1 39%. Outcome RL raises faithfulness and then plateaus. → Models
do omit things from their CoT, just not on command. In both cases the omission happens when the
omitted thing isn't needed for the computation (Emmons et al. 2025's necessity point). Our board
words are needed.

## Background: can reasoning happen off the page at all?

**Pfau, Merrill, Bowman (2024).** _Let's Think Dot by Dot: Hidden Computation in Transformer
Language Models._ arXiv:2404.15758. Transformers can use meaningless filler tokens ("......") in
place of a CoT to solve algorithmic tasks they can't solve with no intermediate tokens, but only
with dense, task-specific supervision. → The theoretical ceiling on hiding is high; the learned
ceiling without training is low.

## What this means for stage 1

1. **Our result is normal.** Chen et al., ReasonIF and Mann et al. all predict it. Stage 1 is also a
   harder version of the task: 25 words instead of 1, the words carry the reasoning, synonyms count
   as leaks for the stage-2 monitor, and the CoTs are long.
2. **The told prompt is under-elicited** (Jose 2026). A strong-told condition is cheap: repeat the
   rule, add a self-check instruction, and give 2–5 few-shot clean spymaster CoTs. Expect gains of
   about 2–3× at best, from ~0 to something still bad. Since the grep metric is saturated, this
   needs the graded metric (fraction of board words written, or tokens to first leak).
3. **Hand them a strategy.** Emmons et al. (2025) found evasion works only when a strategy is
   provided. Codenames has a free one: the monitor never sees the board, so referring to cards by
   grid index ("card 7 and card 19 both fit OCEAN") leaks nothing except through the clue. A "told +
   use indices" condition separates can't-hide (capability) from won't-hide (controllability). If
   they can't follow even that, it's pure CoT-control failure. If they can, the leak moves to the
   clue, which is the clue-only baseline.
4. **Cheap fine-tuning exists** (METR: 240 examples of LoRA → 3× controllability). Clean
   index-referencing CoTs are easy to make by rewriting stage-1 CoTs with a regex (board word →
   `card N`), which gives an SFT set for free. This could stand between prompting and stage-5 RL.
