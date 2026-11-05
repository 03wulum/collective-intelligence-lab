# Explainer · 04 A tiny learned checker

*About a 3-minute read.*

## 1. What we tested and why

In experiment 01, a group of 4 tiny models almost always had the right answer *somewhere* among them (95-99% of the time), but a majority vote often picked a wrong one. So we asked: can a tiny "judge" model learn to spot the right answer, the way a teacher marks homework, and beat the vote?

## 2. The idea, as an analogy

Four students answer a sum. Instead of going with the most popular answer, a marker checks each one and picks the one that looks right. This works brilliantly *if marking is easier than solving*. Think of checking a Sudoku versus solving one. For research on big models (OpenAI's 2021 "verifiers" paper), this trick beat a model 30× larger.

## 3. What we did

1. Trained 8 tiny models (0.4 MB each) per run to do 3-digit addition.
2. Collected 64,000 of their answers on new sums and labelled each right or wrong.
3. Trained a judge of the same size to predict "right or wrong" from the written sum `417+389=806`.
4. On unseen sums, let the judge pick from 4 models' answers and compared it with the vote.
5. **Control:** a judge trained on shuffled labels, so it learns nothing. The real judge has to beat it.

## 4. What happened

The judge picked the right answer only 38% of the time, against 73% for the vote and 52% for a single model. It did no better than the judge that learned nothing. **Verdict: FAIL.**

## 5. Why

For addition, checking `417+389=806` means doing the sum yourself. Marking is as hard as solving. A judge the same size as the students is no better a marker than they are students. A single model trained with all the swarm's and judge's training budget scored 99.9%, beating everything.

## 6. What it teaches

- A checker only pays off when checking is easier than solving. Examples are code with unit tests, puzzles with rules you can verify, and answers you can plug back in.
- For tiny models with a fixed budget, *one well-trained model* has now beaten the swarm twice (experiments 01 and 04).

## 7. Applications and further reading

- Cobbe et al. 2021, *Training Verifiers to Solve Math Word Problems*.
- Lightman et al. 2023, *Let's Verify Step by Step*.
- Brown et al. 2024, *Large Language Monkeys*. It shows that the right answer is usually among many samples, and that picking it is the hard part.

## Post-readiness

**Not worth a post on its own.** It is a clean negative result. It is better folded into a thread on "what makes tiny-model teams work", together with experiment 01.

Draft (249 chars): *Tried giving a swarm of 0.4 MB models a 0.4 MB "judge" to pick the right answer. The judge did worse than a plain vote (38% vs 73%). For arithmetic, checking a sum is as hard as doing it. Verifiers only pay off when checking is cheaper than solving.*
