# Technical document

Team DIMACodm. More than a Hack 2027, Phase 1: Ball Possession.

Repository: https://github.com/aliamrrr/dimacodm-mtah2027-possession

## 1. Approach

**The question.** For every frame of the tactical video, say which team has the ball or that the
ball is dead. The ranking metric, spellF1, matches whole possessions by temporal overlap, so what
matters is where possessions start and stop, not how many frames are right.

**The constraint that shaped everything.** The rules forbid training or tuning on the provided
match. The system is therefore trained and selected on public data about other games, and the
difficulty moves to a different place: making those other games look, to the model, like the
automatic tracking of the provided match.

**The model.** Two questions per frame: which team controls the ball, and is the ball dead.

1. *Per frame, a set transformer.* Each of the 46 anonymous track slots that is present, and the
   ball, is a token: position, velocity, team, position and speed relative to the ball. Thirty
   global features complete the frame: quantiles of the players' speeds before and after the
   frame, ball beyond a line, players off the pitch, kick-off formation, players tracked per team.
   Two masked self-attention layers (64 dimensions, 4 heads) see only the slots that are present.
   The tokens are pooled per team (mean and max) and joined with the ball token.
2. *Across time, a dilated convolution.* Nine residual blocks with dilations 1 to 256 give every
   frame 511 frames (20 s) of context on each side.
3. *Heads.* The state (H, A or D), the only one used to predict; and three auxiliary ones that
   shape the representation: the ball position (which the model has to guess where the tracking
   lost the ball), the player in possession (a pointer on the tokens), and the frames where the
   team in possession changes.
4. *Ensemble.* Six models (five folds by game, plus one trained on every game). Each runs twice,
   as is and with the two teams exchanged, so the output is exactly symmetric between the teams.
   Their log-odds are averaged.
5. *Decoder.* A Viterbi decoder over four states (H, A, dead after H, dead after A) turns the
   per-frame log-odds into possessions. Changing team, stopping and restarting each have a cost;
   a restart is cheaper while the players accelerate, and a change of team is cheaper where the
   model detects one. Each half is decoded on its own.

**The loss follows the metric.** spellF1 counts each possession once, a per-frame loss counts it
by its length. Each frame is therefore weighted by (duration of its possession) ^ -0.5, with a
floor at 0.5 s, so short possessions are not drowned by long ones.

**Inference** is numpy in float64 (`deep_net.py`); the decoder inputs are rounded to 1e-4. The
same file comes out on any machine. It takes about 15 minutes on a 4-core laptop, without a GPU.

**Why this rather than something else.** A logistic model on hand-made distance features with the
same decoder (`ali`, kept in the repository as the reference) reaches 0.46 on our first benchmark
against 0.75 for the deep model: possession depends on the configuration of all players and on
what happens seconds before and after, which a set model with temporal context captures and
per-frame distances do not.

## 2. Assets used

| asset | used | how, or why not |
|---|---|---|
| tactical tracking (ATD) | yes | positions of the players and of the ball, team of each slot |
| ATD speed column, third ball field | no | everything is recomputed from positions, so that the same code runs on the other games |
| Smart Tagging | no | it has no equivalent on the other games, so its use could not be validated |
| homography | no | not needed to decide possession |
| tactical video | no | |
| broadcast video and tracking | no | not synchronised with the tactical video |
| public ground truth (5 minutes) | checks only | see section 5 |

## 3. Training data

None of it comes from the provided match. We labelled nothing by hand; all labels come from the
publishers of the data.

| source | games | tracking | labels, and who made them |
|---|---|---|---|
| Metrica Sports sample data (github.com/metrica-sports/sample-data, commit `e706dd5`) | 3 | 25 fps | event data annotated by Metrica; per-frame H / A / D derived mechanically from the events (`bench.derive_labels`) |
| IDSSE / DFL (Bassek et al., Scientific Data 2025, CC-BY 4.0, doi 10.6084/m9.figshare.28196177) | 7 | 25 fps | `BallStatus` and `BallPossession` on every frame, from the DFL's quality-controlled tracking. This is how the challenge describes its own ground truth |
| SkillCorner open data (github.com/SkillCorner/opendata, MIT, commit `4340d27`) | 20 | 10 fps, resampled to 25 | team in possession per phase of play, by SkillCorner. Training only; its stoppage frames are never a target |

Every file is downloaded by the code and checked against a pinned hash. The ten Metrica and DFL
games are also the benchmark: each one is only ever scored by a model that did not see it (five
folds by game). Each source keeps its own label convention through a learned embedding; the DFL
one is used to predict, because CHALLENGE.md describes the ground truth as built the same way.

**Making clean tracking look automatic.** These games come with clean tracking. Every training
window is degraded with magnitudes drawn at random: anonymous slots that follow a player for 8 s
to 200 s, gaps, identity changes, duplicates, position noise from none to 40 cm, episodes where a
large part of the players is not tracked, a ball missing 25 % to 55 % of the time in runs of 0.6 s
to 30 s, and a ball that keeps moving during stoppages (`deep_data.py`).

The ranges were set from a label-free description of the provided match's tracking, in coarse bins
only (how smooth positions are, how long the ball stays visible or missing, how long a slot
follows a player, how many objects are tracked, whether a visible ball is ever still). RULES.md
allows inspecting the data to design features. No label, frame or value of the match is used, and
no model is fitted on it. It showed that our first simulation was in the wrong regime:

| | first simulation | automatic tracking of the match |
|---|---|---|
| position jitter | 3 to 40 cm per frame | under 0.5 cm |
| ball seen or missing, in runs of | 0.6 to 3 s | 10 to 30 s |
| a slot follows a player for | 8 to 45 s | 1 to 3 min |
| frames with fewer than 16 objects tracked | almost never | 5 to 15 % |
| ball visible but still | over 30 % of the frames with a ball, in the source data | 5 to 15 % |

The last line is the important one. In the source data a dead ball lies still or is hidden; in an
automatic tracking it is visible and moving. A model trained on the former takes such stoppages
for play. We reproduced this on the other games by making the ball of 70 % of the stoppages move
(it glides to the restart spot, wanders, is carried, or is passed between players): the share of
dead-ball frames found fell from 98 % to 72 %. Training with such stoppages restored it.

## 4. Possession rules implemented

- **Three states, no abstention.** Every graded frame is H, A or D. `X` is written only outside
  the two halves, whose limits come from the official `games.py`.
- **Ball not tracked.** Possession carries on. The model infers the team from the players and
  from the last and next sightings within its 20 s of context.
- **Loose or contested ball.** Possession stays with the team that had it: changing team has a
  cost in the decoder, paid only by sustained evidence for the other team.
- **Brief possessions are kept.** We first read "a simple touch without control is not a change
  of possession" as a duration rule (a spell under 2 s between two spells of the other team does
  not count). CHALLENGE.md says that the public ground truth wins over any reading of the
  definition, and it contains short possessions that this rule would delete. The rule is switched
  off (`min_spell_s: 0`); the code is kept for the record.
- **Stoppages.** The model's cues are the players (collective slowing, kick-off formation,
  players off the pitch) and the ball beyond a line. It is trained not to rely on a ball that
  stops moving.
- **Restarts.** The decoder remembers which team had the ball before the stoppage, and a restart
  by the same team costs slightly more than one by the other team.
- **Dead-ball ownership** is not asked for: a dead ball is `D`.
- **Halves** are decoded separately, so nothing carries over half-time. The frames graded before
  the second-half kick-off are labelled `D` by the model itself; no frame number is written in
  the code.

## 5. Results and checks

**On other games** (ten games, each predicted by a model that never saw it; the benchmark whose
tracking is closest to an automatic one; mean of the per-source means, labels as published):

| | spellF1 | dead-ball frames found |
|---|---|---|
| model trained without moving dead balls | 0.641 | 72 % |
| final model, same decoder | 0.725 | 98 % |
| final model, decoder searched out-of-fold | **0.752** (Metrica 0.780, DFL 0.723) | |

Per game, the final system scores between 0.66 and 0.80. The decoder parameters were searched by
coordinate ascent on these out-of-fold predictions (`deep_tune.py`). For reference, the published
floor of the organisers' baseline is 0.243 on the provided match.

**On the provided match**, checks only, none of them used to choose a parameter:

- `mtah data check-loader`: our loader rebuilds the four official baselines byte for byte.
- The frames graded before the second-half kick-off, which FAQ.md says are all dead ball, are all
  `D` in our file.
- On the public 5 minutes, as the official README suggests ("compare the poss and dur columns"):
  19 possessions against 17 in the ground truth, mean duration 12.8 s against 14.2 s; the four
  stoppages start and end within one second of the truth, except two boundaries within three.
  spellF1 on this window is 0.77; with 17 possessions it carries little information.
- `mtah final` produces the same bytes twice, with and without the tracking cache.

These checks did tell us where the pipeline was wrong: an earlier version produced too many
possessions on the public window, a later one found the stoppages but made them too short.
Each fault was then reproduced on the other games and corrected there, with their labels.

## 6. Limitations

- Possessions shorter than 2 s are the hard part: fewer than half are recovered on the other
  games, and each one missed also damages the possession it interrupts.
- Our benchmark is a simulation of automatic tracking. Two gaps between simulation and reality
  were found and closed; others may remain, and the score on other games is an estimate, not a
  measurement, of the score on the provided match.
- With the ball untracked for tens of seconds, the team in possession is inferred from the players
  alone, and a change of possession inside such a stretch can be missed.
- The label convention of the training games (possession changes at every touch) is assumed to be
  that of the provided match. The public window is consistent with it, on 17 possessions.
- Smart Tagging and the video are unused; both could help with restarts and with the lost ball.
- Training on a GPU is not bit-reproducible. The published weights are pinned by hash and the
  prediction from them is bit-reproducible.

## 7. Reproduction

```bash
git clone https://github.com/aliamrrr/dimacodm-mtah2027-possession.git
cd dimacodm-mtah2027-possession
uv sync --locked
# unpack the data bundle into data/, or set MTAH_DATA_DIR
uv run mtah data check-loader
uv run mtah final ali_deep          # writes final/DIMACodm.csv; compare with final/SUBMISSION.json
```

Python 3.11 and every dependency are pinned in `uv.lock`. No GPU; `mtah final` runs the solution
twice, about 15 minutes each on a 4-core laptop. Training is described in docs/TRAINING.md.
