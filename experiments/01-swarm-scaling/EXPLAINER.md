# Explainer · 01 Swarm scaling

*About a 5-minute read.*

## 1. What we tested and why

You asked whether many small models working together could become smarter *exponentially* as you add more of them, rather than gaining a fixed amount per model. We tested that claim directly with a swarm of tiny models. The answer is no: they get much better, but the improvement slows down steadily rather than compounding.

## 2. How it works: an analogy

Picture a pub quiz where you ask 16 people the same question and go with the most common answer (**majority vote**). If everyone made *random, unrelated* mistakes, the crowd would almost never be wrong. Each extra person cuts the chance of a wrong majority by a constant factor. That is **exponential** improvement, and it's a 1785 result called **Condorcet's jury theorem**.

The catch is that real people share blind spots. If a question is a trick question, most of the room falls for it together. Adding more people doesn't fix a *shared* mistake. That is what stops the gains from being exponential.

Terms:
- **Model / member:** here, a tiny neural network (a **transformer**, the same design as ChatGPT, but 103,000 parameters instead of hundreds of billions). Each is 0.4 MB.
- **Swarm of N:** N different members answer and vote.
- **Error rate:** the fraction of questions the swarm gets wrong.
- **Power law:** error ∝ N^−b. Doubling N cuts error by a *fixed fraction*, which gets slower and slower in absolute terms. **Exponential:** error ∝ e^−cN. Adding a member cuts error by a fixed fraction, which collapses fast.

## 3. The process, step by step

1. **Task.** 3-digit addition such as `417+389=`. It's generated on the fly, so the answers are always checkable and no test data can leak into training.
2. **Train members.** 16 tiny models per run, each about 12 seconds on one CPU core. Each gets a different random start and a different random slice of training problems, so they make *different* mistakes.
3. **Vote.** For N = 1, 2, 4, 8, 16, pick N members at random (100 times) and measure majority-vote accuracy on 400 unseen problems.
4. **Controls.**
   - *Clone swarm* (negative control): 8 copies of the same member. Voting can't help, and the test must show that.
   - *One model asked N times:* separates "more votes" from "more different voters".
   - *Same compute:* instead of 4 small models, train one model 4× longer.
   - *Independent-errors what-if:* shuffle which questions each member gets wrong, to see what perfect independence would give.
5. **Judge.** Five pass/fail criteria were written down before the confirmatory run, with thresholds from the papers. The script prints PASS/FAIL.

**Choices you might disagree with:**
- **Tiny models we trained, not Qwen or SmolLM.** The cloud sandbox couldn't download weights, and training our own lets us control diversity exactly. Real LLMs are the next experiment.
- **Arithmetic, not open-ended questions.** One right answer is the *best* case for voting. If it isn't exponential here, it won't be on fuzzier tasks.
- **"Exponential" judged by which curve fits better** (exponential vs power law), rather than a fixed accuracy number.
- **Some exploration came first**, before the criteria were frozen. We then re-ran everything on new seeds and a new test set.

## 4. What we found, including what failed

| | Result |
|---|---|
| Swarm beats its members | ✅ 8 members voting: +16 points over the average member (82% → 99% at 16) |
| **Exponential?** | ❌ **No. Error falls as a power law, error ∝ N^−1.44** |
| Diversity is the cause | ✅ 8 *different* members beat 1 member asked 8 times by 12.5 points |
| Worth the compute? | ❌ One model trained 4× longer: 99.2% vs 97.6% for 4 voting |
| Adaptive stopping | ✅ Stop once 2 members agree: 98.3% using 2.8 models per question, vs 98.6% with all 8 |
| Negative control | ✅ Clone swarm gained exactly 0, so the test can tell |

**Verdict: FAIL ✨ INTRIGUING.** FAIL means the exponential claim didn't hold, and that is still a useful result. There *is* a short burst where doubling helps more each time (2 → 4 members added 13 points), and then it flattens.

## 5. Why it matters

- **Phones and laptops:** you can't fit a giant model, but you can fit several small ones. This tells you how far that gets you and where it stops.
- **Battery:** adaptive stopping is the practical win. Easy questions use 2 models, hard ones use more, for about a third of the compute of always using 8.
- **Design rule:** buy *diversity* (different training data, architectures, vendors), not *count*. And if you can verify an answer (run the code, check the math), use that instead of voting. At least one member was right 99.6% of the time with only 4 members.
- **Who cares:** on-device AI teams (Apple, Google, Qualcomm), anyone building offline agents, and anyone pricing "multi-agent" systems that bill per model call.

## 6. The broader conversation

- [More Agents Is All You Need](https://arxiv.org/abs/2402.05120) (Li et al. 2024): sampling and voting scales with agent count on LLMs. The claim we stress-tested.
- [Are More LLM Calls All You Need?](https://arxiv.org/abs/2403.02419) (Chen et al. 2024): voting can even get *worse* with more calls when easy and hard questions mix.
- [Scaling LLM-based Multi-Agent Collaboration](https://arxiv.org/abs/2406.07155) (Qian et al. 2024): finds logistic (S-shaped) growth with agent count, like ours.
- [Should we be going MAD?](https://arxiv.org/abs/2311.17096) (Smit et al. 2023): multi-agent debate often doesn't beat simple voting.
- [Large Language Monkeys](https://arxiv.org/abs/2407.21787) (Brown et al. 2024): coverage (any attempt right) keeps climbing, so verifiers are the lever.

## 7. Questions someone might ask you

**"So swarms don't work?"** They work well: 82% → 99%. They just don't compound. Each doubling buys less than the one before once you're past about 4 to 8 members.

**"Why not exponential?"** Shared mistakes. If the members were truly independent, 8 would be perfect. A few problems fool most of them the same way, and more voters can't outvote a shared mistake.

**"Wouldn't bigger models change this?"** It would change the numbers, not the shape. LLMs trained on similar internet data probably share *more* blind spots, not fewer. Experiment 02 tests real small LLMs.

**"Why train tiny models instead of using real ones?"** It gave full control over diversity and compute, with no downloads, on a laptop in minutes. It's a clean lab version of the question.

**"When would you actually use a swarm?"** When you can't retrain (combining off-the-shelf models on a device), when you have diverse models, and especially when you can check answers. Use adaptive stopping so easy questions stay cheap.

**"How do you know the test isn't rigged to pass?"** The clone swarm (no diversity) had to fail, and it did, gaining exactly zero. The criteria were frozen before the confirmatory run, on fresh seeds.
