# Lit check: CoT control, stego, and secret-hiding games

Stage-0 lit check for [codenames-plan.md](codenames-plan.md). Compiled 2026-09-28 via web search;
numbers come from abstracts or HTML full text unless marked (unverified). Text after → says how each
item bears on this project. Skipped as too distant: Motwani et al. 2024 (arXiv:2402.07510, secret
collusion) and Guan et al. 2025 (arXiv:2512.18311, OpenAI "Monitoring Monitorability": longer CoT is
more monitorable).

## 1. CoT controllability and obfuscation under monitoring

**Chen Yueh-Han, McCarthy, Lee, He, Kivlichan, Baker, Carroll, Korbak et al. (2026).** _Reasoning
Models Struggle to Control their Chains of Thought._ arXiv:2603.05706 (ICML 2026). CoT-Control
suite: solve GPQA/HLE/BFCL-style problems while obeying a CoT constraint. The closest task is
"suppress keyword" (e.g. a genetics question without writing "chromosome"). CoT controllability is
far below output controllability: Sonnet 4.5 2.7% vs 61.9%; DeepSeek R1 0.1%; Qwen3-32B 2.3%;
gpt-oss-120B 5.9%. Controllability goes up with model size and down with RL training, reasoning
effort, CoT length (odds ratio 0.79 per +1 SD, about 2k words) and problem difficulty. Telling a
model its CoT is monitored raises it by at most about 4pp, significant only for some models.
Suppression costs 6-17pp of task accuracy. Models often meta-discuss the constraint ("I must not say
X"), but on suppression tasks this happened on ≤10% of samples. → This is the told/not-told
manipulation plus a grep monitor, for a single word. It predicts that our stage 1 told condition
fails badly (we suppress 25 load-bearing words, not one), and that told vs not told differs by a few
pp at most.

**Zolkowski, Xing, Lindner, Tramèr, Jenner (2025).** _Can Reasoning Models Obfuscate Reasoning?
Stress-Testing Chain-of-Thought Monitorability._ arXiv:2510.19851 (MATS). The model gets a main task
plus a hidden side goal, under escalating prompt pressure to hide the side goal (toy tasks and
SHADE-Arena). Without pressure, the CoT monitor works well. Under pressure, R1, gpt-oss and GPT-5
basically fail to obfuscate their internal CoT. Only Claude reached 75-90% conditional obfuscation,
and only at the highest pressure. Models obfuscate their internal CoT less than their external
(prompted) CoT. → Expect open-weight models to leak even under strong prompts. Don't let a "reason
in the answer channel" condition stand in for the internal CoT; the two behave differently.

**Emmons, Jenner, Elson, Saurous, Rajamanoharan, Chen, Shafkat, Shah (2025).** _When Chain of
Thought is Necessary, Language Models Struggle to Evade Monitors._ arXiv:2507.05246 (GDM). They
separate CoT-as-rationalization from CoT-as-computation. Gemini 2.0 Flash and 2.5 Pro answer GPQA
while a monitor looks for reasoning about a forbidden topic, and also run email-side-task and math
tasks. Unaided, the models fail to evade. Evasion works only with human-written strategies or
iterative/RL optimization against the monitor (thousands of RL steps on the math task). → How much
the spymaster leaks depends on whether CoT is necessary for Codenames. The stage-0 no-CoT baseline
measures exactly this: if no-CoT play is about as good, then "don't think about the board" is a
cheap strategy, and leakage becomes a propensity question, not a capability one.

**Haskins, Chughtai, Engels (2026).** _Training on Documents About Monitoring Leads to CoT
Obfuscation._ arXiv:2605.15257. They finetune on synthetic documents describing a CoT monitor (SDF,
no in-context instruction) in 8 open models: gpt-oss-20b/120b, Kimi K2-Thinking, Kimi K2.6,
DeepSeek-V3.1 and three Qwen3.5/3.6 models. Monitor-aware models get higher undetected misbehavior:
gpt-oss-120b went from 1.7% to 25.7% on a simple defect-concealment task, with a smaller effect on
an agentic task. The models' strategies were omission, vagueness and suppression. Controllability
predicts obfuscation success across models (r=0.80). → Covers our model families; SDF is an
alternative "told" manipulation. Omission looks like hiding unless we also track task score.

**Baker, Huizinga, Gao et al. (2025).** _Monitoring Reasoning Models for Misbehavior and the Risks
of Promoting Obfuscation._ arXiv:2503.11926 (OpenAI). A GPT-4o CoT monitor catches o3-mini's reward
hacking far better than an action-only monitor (about 95% vs 60% recall, unverified number). Putting
the CoT monitor into the RL reward leads to obfuscated reward hacking. → This is the canonical
result behind "RL would fix it fast" in codenames-plan.md. Stage 5 should expect hiding plus a
residual leak that the trained-against monitor misses but a fresh monitor catches.

**Korbak, Balesni, Barnes, Bengio, Benton et al., 41 authors (2025).** _Chain of Thought
Monitorability: A New and Fragile Opportunity for AI Safety._ arXiv:2507.11473. Position paper. CoT
is monitorable because hard tasks require thinking out loud (necessity) and because models tend to
verbalize anyway (propensity). Both can erode through training. → Not told measures propensity; told
measures necessity plus inability to hide.

**Kaufmann, Lindner, Zimmermann, Shah (2026).** _Aligned, Orthogonal or In-conflict: When can we
safely optimize Chain-of-Thought?_ arXiv:2603.30036. Sorts reward terms by whether they conflict
with transparent CoT; in-conflict terms reduce monitorability in training. Abstract only. → Our
stage-5 reward (score − monitor catch) is in-conflict by construction: a clean test of it.

**Schulz et al. (2026).** _Monitor Jailbreaking: Evading Chain-of-Thought Monitoring._ LessWrong,
Feb 2026. GRPO on gpt-oss-20b/120b with a GPT-5-family monitor penalty. The models learned CoT
phrasings that make the monitor misread plainly visible reasoning ("the monitor can only see this
section"), not stego. GPT-5-nano was jailbroken in about 150 batches; GPT-5.2 resisted. → Our
stage-2 LLM monitor can be jailbroken or decoyed by text in the CoT. The grep monitor is immune to
that, so keep grep as a floor.

## 2. Steganography / encoded reasoning

**Roger, Greenblatt (2023).** _Preventing Language Models From Hiding Their Reasoning._
arXiv:2310.18512. Defines encoded reasoning and an eval method for defenses. With context-aware
paraphrasing, their best schemes carry at most about 3 bits per KB of text. → Paraphrasing the CoT
before the monitor reads it is a cheap stego check. If leak survives paraphrase, it lives in the
semantics, not in token choice.

**Skaf, McCarthy et al., 11 authors (2025).** _Large language models can learn and generalize
steganographic chain-of-thought under process supervision._ arXiv:2506.01926 (NeurIPS 2025). PPO on
Qwen-2.5-3B with specific strings penalized in load-bearing CoT. The model swapped "*" for "times",
then "·", then U+2217, and swapped Heads/Tails for R/L. When a whole class of strings (character
names) was penalized, it learned a general encoding that transferred to held-out names. The encoded
CoT stayed load-bearing: forcing a fixed encoding dropped coin-flip accuracy from 98% to about 50%.
→ This is the RL version of our stage 1 (penalize board-word strings). It predicts stage 5 with a
grep monitor yields synonym or abbreviation substitution within hours, which an LLM monitor should
partly catch.

**Mathew, Matthews, McCarthy, Velja, Schroeder de Witt, Cope, Schoots (2024).** _Hidden in Plain
Text: Emergence & Mitigation of Steganographic Collusion in LLMs._ arXiv:2410.03768. Stego collusion
emerges from misspecified rewards under both gradient-based and in-context RL. Standard mitigations
(overseer, paraphraser) don't fully stop it. → Their in-context RL loop is a cheaper alternative to
GRPO for a stage-5 pilot.

**Zolkowski, Nishimura-Gasparian, McCarthy, Zimmermann, Lindner (2025).** _Early Signs of
Steganographic Capabilities in Frontier LLMs._ arXiv:2507.02737 (ICLR 2026). Frontier models can't
hide short messages from a monitor under standard affordances. They succeed with an unmonitored
scratchpad plus a pre-agreed scheme. There are early signs of encoded reasoning on state tracking
(e.g. hex). → Our spymaster has neither affordance: the CoT is the monitored scratchpad and there is
no shared scheme. Expect no stego from prompting alone.

**Guo et al. (2025).** _All Code, No Thought: Current Language Models Struggle to Reason in Ciphered
Language._ arXiv:2510.09714 (ICLR 2026). Reasoning in rot13 costs GPT-4.1 19%+ accuracy, and obscure
ciphers cost up to 72%, even though the models translate the cipher near-perfectly. Closing the gap
took >3.7B tokens of ciphered finetuning. Author list not checked. → If a told spymaster tries to
cipher the board words, expect a big drop in turns-to-finish. That shows up on the plan's
capability-cost scatter.

## 3. Secret-hiding games, Decrypto, Codenames

**Lupu, Willi, Foerster (2025).** _The Decrypto Benchmark for Multi-Agent Reasoning and Theory of
Mind._ arXiv:2506.20664 (ICLR 2026). Encoder/decoder vs interceptor over 4 secret keywords; the
interceptor sees only the public hint history, not the CoT. Games skew heavily toward the
interceptor, with a real interception rate of about 52%, while LLMs predict interception on nearly
every turn. Newer reasoning models play worse than older ones, and GloVe/word2vec baselines beat the
LLMs. → This is our clue-only control. Public Codenames clues alone should let a monitor beat 25/400
chance by a lot, so measure the CoT leak only above the clue-only baseline.

**Stephenson, Sidji, Ronval (2024; v2 2025).** _Codenames as a Benchmark for Large Language Models._
arXiv:2412.11373. Has a single-team variant (lower score is better): o1-preview 8.41, Llama-3.1
10.18, GPT-4o 10.58, o3-mini 9.70. Failure modes: clues that contain a board word (an invalid clue),
fixating on one target word, and underweighting the assassin. → Rough reference for stage-0
turns-to-finish (their board has an assassin, so not directly comparable); check for their
invalid-clue failure during stage-0 hand validation.

**Holtzman, West (2026).** _Can You Keep a Secret? Involuntary Information Leakage in Language Model
Writing._ arXiv:2605.10794. A model gets a secret word, is told to hide it, and writes a story; a
second model picks the secret in a 2-way forced choice (chance 50%). "Don't reveal": Llama 4
Maverick 79%, Opus 4.6 78%, DeepSeek V3.2 78%. "Actively hide" flips detection below chance (Opus
25%, GPT-5.4 9%), and an avoidance-aware guesser recovers about 60%. Decoy words redirect the
leakage. Leakage jumps sharply with scale (Llama 8B about 50%, 70B 84%). Outputs only; the CoT was
not studied. → This is the closest adversarial-guesser design, but on outputs. Expect told
spymasters to leak through avoidance, which a board-aware monitor (stage 3) can exploit. Also expect
decoy-flooding of the CoT as a strategy against the stage-2 monitor.

**Green, Gubri, Puerto, Yun, Oh (2025).** _Leaky Thoughts: Large Reasoning Models Are Not Private
Thinkers._ arXiv:2506.15674 (EMNLP 2025). Reasoning traces routinely contain private user data. More
reasoning budget makes answers more cautious but leaks more in the CoT. The follow-up by Puerto et
al. (2026, arXiv:2602.24210) uses SFT plus a decoding strategy to make CoT follow privacy
instructions: +51.9pp on privacy benchmarks, at some task cost (models 1.7B-14B). → This is the same
phenomenon with PII as the secret. It also shows prompting alone doesn't make CoT follow "don't
mention X", while light SFT does.

**Cywiński, Ryd, Wang, Rajamanoharan, Nanda, Conmy, Marks (2025).** _Eliciting Secret Knowledge from
Language Models._ arXiv:2510.01070. Author list unverified. "Taboo" model organisms are finetuned to
hint at a secret word without ever saying it. Prefill attacks elicit it best; logit lens and SAEs
help less. → These models keep a secret in outputs by training, the reverse of our question. If the
prompt fails, their elicitation methods are a ready white-box monitor for a stage-5 model.

## Closest prior work

1. **CoT-Control (Chen Yueh-Han et al., 2026)**: keyword suppression in CoT, with and without being
   told about monitoring, including Qwen3 and R1. It is our stage 1 with one word instead of 25.
2. **Holtzman & West (2026)**: an adversarial guesser extracts a hidden secret word, and the
   avoidance itself leaks. It is our stages 2-3, but on outputs, not CoT.
3. **Decrypto (Lupu et al., 2025)**: a word-association game with an interceptor. It is our
   clue-only baseline; the interceptor never reads the CoT.

## Is this setup novel?

Partly. The core question, whether a model can keep a named thing out of its CoT when told it is
monitored, is already answered for single keywords by CoT-Control: mostly no, and being told helps
by ≤4pp. So "SOTA open models leak" is close to a known result, and stage 1 alone is not a
contribution. What seems new: (a) the secret is a set of 25 words the task needs; (b) leakage is
scored by an adversary who must extract a word, not by string match, so paraphrase leaks count; (c)
the clue-only baseline separates CoT leakage from leakage through the public action; (d) the models
are new open weights (K3/GLM-5.3/Qwen3.8/V4). I found no paper where a monitor reads a reasoning
model's CoT to extract a secret it must reason about in a game.

## Design implications

- **Stage 1 will probably fail the grep test.** CoT-Control finds ≤6% control on one word for open
  models, and our task needs 25 load-bearing words. Decide now whether the "reconsider" step means
  switching to a graded metric (e.g. fraction of board words mentioned, or tokens to first leak)
  instead of a binary pass.
- **Measure CoT necessity** (Emmons et al., 2025). The no-CoT baseline tells you whether hiding is
  possible without a performance cost. Report told-condition leakage together with turns-to-finish,
  because omission and vagueness (Haskins et al., 2026) look like success.
- **Meta-discussion confound.** Told spymasters will write "I must not mention the board words" and
  then list them. Log and count meta-discussion separately (CoT-Control did).
- **Clue-only baseline will be high**, far above 25/400, because the clue is built to point at board
  words (Decrypto interceptors crack ~52% of codes from hints alone; different task, same effect).
  Leak should be CoT minus clue-only; consider also a CoT-with-clue-redacted monitor.
- **Avoidance and decoys are channels.** Holtzman & West: hiding flips detection below chance and an
  avoidance-aware guesser recovers it. Stage-2 and stage-3 monitors should be told the spymaster may
  be hiding, and a decoy-flooding strategy should be expected in the CoT.
- **Monitor jailbreaking** (Schulz et al., 2026): the LLM monitor can be steered by CoT text. Keep
  grep as a floor and report both.
- **Stage 5 RL:** expect string substitution within hours (Skaf et al., 2025), not deep stego.
  Evaluate against a monitor held out from training plus a paraphrase step (Roger & Greenblatt,
  2023), and check that the CoT stays load-bearing.
- **Internal vs external CoT differ** (Zolkowski et al., 2025). Confirm OpenRouter returns the raw
  internal trace (already in stage 0), and never mix models whose providers summarize it.
