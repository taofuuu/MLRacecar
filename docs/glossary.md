# Glossary

> **In plain words:** Every technical term used in these docs, explained simply, with where it
> shows up in this project. Grouped by topic; skim the group you need.

## Planning and teamwork

| Term | Plain meaning | Where it shows up here |
|------|---------------|------------------------|
| **Acceptance criteria** | A checklist that says exactly when a ticket counts as finished. | Every ticket on the board. |
| **ADR** (Architecture Decision Record) | A short file recording one big decision: what we picked, what else we considered, and why. | `docs/adr/` |
| **Backlog** | The list of all work we plan to do. | GitHub Issues and the Project board. |
| **Definition of Done / Ready** | Shared checklists: when a ticket can be started (Ready) and when it's truly finished (Done). | `CONTRIBUTING.md` |
| **Epic** | A big chunk of work made of several tickets. | Milestones M5–M8 are planned at epic level. |
| **Milestone** | A stage of the project with a goal and a target date. | M0–M8 in the roadmap. |
| **MVP** (Minimum Viable Product) | The smallest version that shows the main idea working. | M4: an AI that laps your track. |
| **Priority P0 / P1 / P2** | Must-have / should-have / nice-to-have. When short on time, P2s are cut first. | Ticket labels. |
| **Rolling-wave planning** | Plan the near future in detail and the far future roughly, then add detail as you get closer. | M0–M4 detailed, M5–M8 rough. |
| **Spike** | A short experiment to answer a question before committing to a design. | M6-1: how to run the demo in a browser. |
| **Velocity** | How much work actually gets done per week. Used to adjust the dates. | Rechecked after M0. |

## Git and GitHub

| Term | Plain meaning | Where it shows up here |
|------|---------------|------------------------|
| **Branch** | A separate copy of the code where you make changes without touching the main version. | One branch per ticket, e.g. `feat/10-spline-centerline`. |
| **Conventional Commits** | A standard format for commit messages: `type(area): what changed`. | `feat(track): add spline resampling` |
| **Pull request (PR)** | A request to merge a branch into `main`. Checks run and the change gets reviewed first. | Every ticket is finished through a PR. |
| **Squash merge** | Merging a PR as one single tidy commit. | How every PR gets merged. |
| **Trunk-based development** | Everyone works off one main branch (`main`) using short-lived branches that merge back quickly. | Our branching strategy. |
| **SemVer** (Semantic Versioning) | Version numbers like `1.4.2` = major.minor.patch. Major changes break things, minor ones add features, patches fix bugs. | Releases v0.1.0 … v1.0.0. |
| **Changelog** | A human-readable list of what changed in each version. | `CHANGELOG.md` |

## Software design

| Term | Plain meaning | Where it shows up here |
|------|---------------|------------------------|
| **Layered architecture** | Code organized like floors of a building: upper floors may use lower ones, never the reverse. | `core` at the bottom, `cli` at the top. |
| **Core** | The bottom floor: physics, track math, race rules. Plain math, no graphics or AI libraries. | `src/mlracecar/core/` |
| **Composition root** | The one place where all the parts get connected together. | The `racecar` command (`cli`). |
| **Dependency** | Something your code needs in order to run: another module or an installed library. | numpy, pygame, torch, … |
| **Optional extra** | A group of libraries you only install if you need that feature. | `train` (PyTorch) and `render` (pygame-ce). |
| **pygame / pygame-ce** | A Python library for opening a window, reading the mouse and keyboard, and drawing shapes. pygame-ce is its actively maintained edition. | The track editor and the race window ([ADR-0012](adr/0012-pygame-ce-for-windows-and-drawing.md)). |
| **Protocol / interface** | A promise about what methods something has, without saying how it works. Lets you swap implementations. | `Agent`: human, SB3, or our own PPO all fit the same slot. |
| **MVC** (Model–View–Controller) | Splitting an app into data (model), drawing (view), and input handling (controller). | The track editor. |
| **Camera** | Which part of the world the window shows, and how zoomed in. It converts metres to screen pixels and back. | Panning and zooming in the editor and the race window. |
| **Grid snap** | New and moved points jump to the nearest grid crossing, so a track lines up neatly. | Press G in the editor. |
| **Immutable** | Can't be changed after it's made. Changing it means making a new copy with the change. | Editor drafts: undo is going back to an earlier copy ([ADR-0011](adr/0011-immutable-editor-drafts.md)). |
| **Command pattern** | Each edit is stored as an object that knows how to do *and* undo itself. A classic way to build undo/redo. | Considered for the editor, replaced by immutable drafts ([ADR-0011](adr/0011-immutable-editor-drafts.md)). |
| **Observer pattern** | One thing publishes updates and many listeners receive them. | Snapshots go to the screen, replay recorder, and video writer at once. |
| **HUD** (heads-up display) | The information drawn over the race: speed, lap, lap times, warnings. | The top-left panel in the race window. |
| **Debug overlay** | Extra drawings for developers that can be switched on and off, like the middle line of the road or each car's velocity. | Keys 1-4 in the race window. |
| **Offscreen rendering / RGB array** | Drawing a frame into memory instead of a window, as a grid of red-green-blue pixel values. | Tests, CI, video export, and what an AI could see. |
| **FPS** (frames per second) | How many pictures a second the window draws. 60 looks smooth. | The race window's target. |
| **Snapshot** | A frozen picture of the whole race at one moment: every car's position, speed, lap. | Drawn, recorded, and sent to the web demo. |
| **Ghost cars** | Cars that drive through each other instead of colliding, so many can learn on the same track at once without getting in each other's way. | Every car until collisions arrive (M7). |
| **Read-only array** | An array that refuses to be changed: writing to it raises an error. | Snapshot arrays, so nothing that keeps a snapshot can change the race. |
| **JSON** | A plain-text format for structured data that almost every program can read. | Track files (`tracks/*.json`). |
| **JSON Schema** | A formal, machine-readable description of what a JSON file must contain. Other tools can use it to check files. | [`track-file-v1.json`](schemas/track-file-v1.json). |
| **Schema / schema version** | The agreed structure of a file, plus a number saying which version of that structure it uses. | Track files: `"schema_version": 1`. |
| **Migration** | Code that upgrades an old file format to the new one, so old files keep working. | Track and replay formats. |
| **Idempotent** | Running it twice gives the same result as running it once. Safe to re-run. | The backlog seeding script. |
| **Typed config** | Settings files whose values get checked (right type, valid range) when loaded. | `configs/*.yaml` checked by pydantic. |
| **YAML** | A plain-text format for settings: `key: value` lines, indented to group them into sections. | Settings files, e.g. `configs/default.yaml`. |
| **Override / layers** | Settings are built in layers: defaults, then files, then `--set`. A later layer overrides (wins over) an earlier one, but only for the settings it mentions. | `racecar config my.yaml --set vehicle.mass=1500` |
| **SI units** | The standard scientific units: metres, kilograms, seconds, newtons, watts, radians. | Inside the simulation; settings files use degrees and kilowatts instead. |
| **Model card** | A file kept with a trained AI that says what it expects to see, how it was trained, and with which code. | `model_card.json`, checked whenever a model is loaded ([Trained agents](agents.md)). |

## Simulation and math

| Term | Plain meaning | Where it shows up here |
|------|---------------|------------------------|
| **Spline** | A smooth curve drawn through a set of points. | The track centerline. |
| **Cubic spline (C2)** | A smooth curve through your points, built from one cubic formula per stretch. "C2" means its position, direction, *and* bend all change smoothly. | How your clicked dots become a smooth track ([ADR-0010](adr/0010-c2-cubic-spline-centerline.md)). |
| **C1 / C2 continuity** | C1: the direction never jumps (no kinks). C2: the bend (curvature) never jumps either. | Why corners drawn as steady arcs bend steadily. |
| **Catmull-Rom spline** | A curve through points whose direction is smooth but whose bend jumps at each point (C1). | Our first choice, replaced by the C2 spline. |
| **Arc length (s)** | Distance travelled along the curve. | How far around the lap a car is. |
| **Segment / polyline** | A segment is a straight line between two points; a polyline is a chain of segments. | Track edges and the centerline are polylines. |
| **2D cross product** | One number that says whether vector *b* turns left (positive) or right (negative) from vector *a*. | Which side of the road a car is on; whether two segments cross. |
| **Projection (onto a polyline)** | Finding the closest point on a line to a given point. | Progress along the lap and distance from the road's middle. |
| **Floating-point rounding / tolerance** | Computers store decimals approximately, so exact comparisons can fail by a hair. A tolerance accepts "close enough". | Rays aimed exactly at a corner where two edge segments meet must still hit. |
| **Curvature** | How sharply a curve bends (1 / radius). | Tight corners have high curvature. |
| **Pose** | Where something is *and* which way it faces: a position plus a heading angle. | Starting-grid slots; car positions. |
| **Checkpoint** | A line across the road. A lap only counts if the car crosses every checkpoint in order. | Evenly spaced around the lap; checkpoint 0 is the start/finish line. |
| **Off track** | The car's centre is off the road, so half the car is on the grass. What happens then is a setting: nothing, slow down, back on the road, or the run ends. | `race.off_track` |
| **Wrong way** | Driving backwards along the track, faster than walking pace. | The `WrongWay` event. |
| **Valid lap** | A lap in which the car crossed every checkpoint in order. Only valid laps count and set lap times. | `LapCompleted.valid` |
| **Sector** | One of the three parts a lap is split into, each timed separately, as in real racing. | `LapCompleted.sectors` |
| **Amortized O(1)** | The work per step stays the same small amount on average, however big the input (here: however long the track). | Finding each car on the road. |
| **Pure pursuit** | A simple way to steer along a path: aim at a point a few metres ahead on it and steer onto the arc that reaches it. | The test-only scripted driver. |
| **Starting grid / pole position** | The cars' starting spots: two lanes, staggered, behind the start line. Pole position is the front spot. | `Track.start_grid`. |
| **Smoothstep** | A blend between two values that starts and ends gently (flat), instead of in a straight ramp. | How road width changes between dots: no kinks in the edges. |
| **Boundary folding** | When a corner is so tight the inner edge of the road crosses over itself. | A track validation rule. |
| **Kinematic bicycle model** | Simple car physics: the car goes where its wheels point, with no skidding. | First car model (M2-2). |
| **Grip / friction coefficient** | How hard the tyres can push sideways before sliding, as a multiple of the car's weight. 1.0 means the car can corner at up to 1 g. | The `grip` setting; it limits how fast a car can take a corner. |
| **g** | The acceleration of gravity, 9.81 m/s². Forces in cars are often measured in g: 1 g sideways pushes you into the door as hard as gravity pulls you down. | Cornering limits. |
| **Sideslip angle** | The small angle between where a car points and where its middle is actually moving while it turns. | The kinematic model's centre moves at this angle. |
| **Dynamic bicycle model / Pacejka** | More realistic car physics where tires can slip, so drifting and understeer happen. Pacejka is a standard tire-grip formula. | Second car model (M2-3). |
| **Understeer / oversteer** | The car turns less than you steer (front slides) / more than you steer (rear slides). | The kinematic model understeers at the grip limit; the dynamic model adds both. |
| **Fixed timestep** | The simulation always advances by the same tiny time slice (e.g. 1/120 s), so results don't depend on computer speed. | The World loop. |
| **Semi-implicit Euler** | A simple, stable way to step physics forward in time. | Vehicle integration. |
| **Action repeat** | The AI decides 20 times per second; each decision is held for several physics steps. | 120 Hz physics, 20 Hz decisions. |
| **Raycast** | Shooting an invisible line from the car and measuring how far until it hits the track edge, like a laser rangefinder. | The car's "eyes": [Sensors](sensors.md). |
| **Lidar** | A sensor that measures distances with laser beams fanned out around it. Our rays work the same way, without the lasers. | [Sensors](sensors.md). |
| **Field of view** | How wide an angle a sensor covers. 180° is from straight left to straight right. | `sensors.field_of_view`. |
| **Digest (hash)** | A short fingerprint computed from some data: the same data always gives the same digest, and any change gives a different one. | `ObservationSpec.digest` ties a trained AI to the exact observations it learned from. |
| **Normalized** | Rescaled to a standard range, usually 0 to 1, so a neural network gets inputs of similar size. | Ray distances divided by the range. |
| **Broad phase** | A quick first pass that skips obviously-irrelevant things before doing precise math. | Rays only check nearby track edges. |
| **Vectorized** | Doing math on whole arrays at once instead of looping one item at a time. Much faster in Python. | All cars update in one go. |
| **Struct-of-arrays** | Storing data as "one list per property" (all x positions, all speeds) instead of "one object per car". Enables vectorizing. | `World` state. |
| **Batched** | Handling many items in one call. | Many cars, many environments. |
| **Deterministic / bitwise reproducible** | Same inputs → exactly the same outputs, down to the last digit. | Same seed = same race. |
| **Seed / RNG** | The starting number for a random-number generator. Same seed = same "random" sequence. | Every run records its seeds. |
| **Throughput** | How much work per second, e.g. simulation steps per second. | Benchmarks in the README. |
| **Car-step** | One car driving for one step (one driver decision). 1,024 cars taking one step together make 1,024 car-steps. | The speed table in the README. Each car-step is one experience the AI can learn from. |

## Testing and tooling

| Term | Plain meaning | Where it shows up here |
|------|---------------|------------------------|
| **CI** (Continuous Integration) | A server that automatically runs all checks on every change. | GitHub Actions, on Windows and Linux. |
| **Lint / linter** | A tool that flags style problems and likely bugs without running the code. | ruff |
| **Type checking** | A tool that checks you're passing the right kinds of values around (e.g. a number, not text). | mypy |
| **Unit test** | A test of one small piece in isolation. | Most of `tests/unit/` |
| **Fuzz test** | Throwing large amounts of random, often nonsensical input at code to prove it never crashes. | Track checks must handle any points a user could place. |
| **Bounding box** | The smallest upright rectangle around a shape. If two boxes don't touch, the shapes inside can't either. | The broad phase that makes crossing checks ~46x faster. |
| **Property-based test** | Instead of hand-picked examples, the tool generates hundreds of random inputs and checks a rule always holds. | Geometry tests with Hypothesis. |
| **Contract test** | Checks our code follows an external standard's rules. | Our env passes Gymnasium's official checker. |
| **Equivalence test** | Checks two ways of computing something give the same answer. | One car alone = the same car among 1,024. |
| **Regression test / golden file** | Save a known-good result once; future runs must match it. Catches accidental changes. | Saved reference laps (M2-8). |
| **Smoke test** | A quick end-to-end run that just checks nothing crashes. | A 2,000-step training run in CI. |
| **Benchmark** | A measurement of speed. | Simulation steps per second. |
| **Coverage** | What percentage of the code is run by the tests. | Target ≥ 90% for `core`. |
| **Thread / worker thread** | A second line of work running at the same time as the main one. | The editor checks tracks on a worker thread so the window never waits ([ADR-0013](adr/0013-track-checks-in-the-background.md)). |
| **Headless** | Running without a screen or window. | Training, CI, and the editor's tests (pygame's windows open off-screen). |
| **Pre-commit hook** | Checks that run automatically every time you commit. | ruff, mypy, import-linter. |
| **import-linter** | A tool that fails the build if code breaks the "floors" rule. | Enforces the layered architecture, and keeps pygame out of code that must run headless. |
| **Lockfile** | A file listing the exact version of every installed library, so everyone gets identical installs. | `uv.lock` |
| **uv** | A fast tool that installs Python and libraries and manages the project environment. | `uv sync`, `uv run …` |
| **MkDocs / GitHub Pages** | MkDocs turns our Markdown docs into a website; GitHub Pages hosts it for free. | [taofuuu.github.io/MLRacecar](https://taofuuu.github.io/MLRacecar/) |
| **Docstring** | The description written at the top of a module, class, or function in the code. | The [API reference](reference.md) is generated from them. |

## Reinforcement learning (RL)

| Term | Plain meaning | Where it shows up here |
|------|---------------|------------------------|
| **Reinforcement learning** | Learning by trial and error: try actions, get rewarded or penalized, slowly do more of what works. | How the car learns to drive. |
| **Agent** | The decision-maker: anything that looks at the situation and picks an action. | Keyboard (you), SB3, our PPO. |
| **Interpolation** | Working out in-between values from two known ones, like where a car is a third of the way between two moments. | Smooth motion: the race moves 20 times a second, the screen draws 60. |
| **Environment** | The world the agent acts in; it reports what happened after each action. | `RacingEnv` |
| **Observation** | What the agent "sees" each step. | Ray distances, speed, angle to the track. |
| **Action** | What the agent does each step. | Steering and gas/brake, each from −1 to 1. |
| **Reward** | A score after each step telling the agent how well it did. | + for moving forward along the track, − for leaving it. |
| **Episode** | One attempt from start until it ends (crash, finish, or time limit). | One run around the track. |
| **Termination vs. truncation** | The episode ended because something happened (crashed) vs. because we stopped it (time limit). The difference matters to learning. | Env end rules. |
| **Run folder** | Everything about one training run in one place: its settings, code version, seeds, saved AIs, and test results. | `runs/<date>_<name>/` ([Training](training.md)). |
| **Practice run** | One of the cars' runs while the AI learns. It still acts a little randomly, to explore, unlike in a test. | The `practice/` charts in TensorBoard. |
| **Saved agent (training checkpoint)** | A copy of the AI saved while it trains, to keep the best one or carry on after stopping. Not the same as a track checkpoint. | `checkpoints/best`, `checkpoints/last`. |
| **Deterministic action** | Always the policy's best guess instead of a random draw around it, so the same situation gives the same action. | How a trained agent drives by default. |
| **Policy** | The agent's "brain": the rule mapping observations to actions. A neural network here. | What training produces. |
| **MDP** (Markov decision process) | The formal description of an RL problem: states, actions, what happens next, rewards. *Markov*: the next state depends only on the current one and the action. | [RL fundamentals](rl-guide.md) §2. |
| **Partially observable** | The agent sees only part of the state, so it can't tell some different situations apart. | Our agent gets rays and a few numbers, not the whole world. |
| **Return** | The total reward from now until the episode ends, later rewards counted for less. | What the agent tries to make as big as possible. |
| **Discount (γ, gamma)** | How much less a reward counts for each step it lies in the future. 0.99 means "care about the next ~100 steps". | About 5 seconds ahead at 20 decisions a second. |
| **Dense / sparse reward** | Feedback every step (dense) vs. only now and then, like once a lap (sparse). Dense is far easier to learn from. | Progress along the track is dense; a lap bonus is sparse. |
| **Reward hacking** | The agent finds a way to score that isn't what we meant. | Running wide over the grass is faster than braking ([RL fundamentals](rl-guide.md) §5). |
| **Value function** | An estimate of the return from a situation: "how good is it to be here?". | PPO's critic. |
| **Advantage** | How much better an action turned out than the value function expected. | What PPO uses to decide which actions to make more likely. |
| **Actor-critic** | A method with two parts: the actor (the policy) acts, the critic (a value function) judges. | PPO and SAC are both. |
| **GAE** (generalized advantage estimation) | A way to estimate advantages that trades a little accuracy for much less noise, set by λ (lambda). | PPO's `gae_lambda`. |
| **Entropy** | How random the policy's choices are. A small bonus for it keeps the agent exploring. | `ent_coef`, `train/entropy_loss`. |
| **Epoch** | One pass over the collected data while learning. PPO does several per batch; its clipping makes that safe. | `n_epochs` (10 by default). |
| **Training** | Running many episodes and adjusting the policy to get more reward. | `racecar train` |
| **Evaluation** | Testing a trained policy without learning, to measure how good it is. | `racecar eval` ([Evaluation](evaluation.md)). |
| **Clean run / completion rate** | A run in which the car drives until the time runs out without ever leaving the road. The completion rate is the share of runs that are clean. | The headline number of `racecar eval` reports. |
| **Gymnasium** | The standard Python interface for single-agent RL environments. | `RacingEnv` follows it ([The RL environment](environment.md)). |
| **Vector environment** | Many copies of an environment stepped together: one call moves every copy, with one row per copy in each result. | `BatchedRacingEnv`: many cars in one world. |
| **Autoreset** | A vector environment starting a copy's next run by itself when its run ends, while the others carry on. | `NextStep` or `SameStep`. |
| **Render mode** | How an environment shows itself: `human` opens a window, `rgb_array` returns pictures (for videos). | `gymnasium.make(..., render_mode="human")`. |
| **Dependency injection** | Handing a part the pieces it needs from outside, instead of it creating them itself, so it doesn't depend on them. | The environment is handed its viewer, so it never imports pygame. |
| **PettingZoo** | The same idea as Gymnasium, for multiple agents at once. | Multi-car racing (M7). |
| **PPO** (Proximal Policy Optimization) | A popular, reliable RL algorithm. It improves the policy in small, safe steps. | Our first algorithm. |
| **SAC** (Soft Actor-Critic) | Another RL algorithm that reuses past experience; often more data-efficient. | Comparison in M8. |
| **PyTorch** | The library that builds and trains neural networks; Stable-Baselines3 runs on it. | The `train` extra. |
| **GPU / CUDA** | A graphics card can do the many small sums of neural-network training much faster than the processor (CPU). CUDA is NVIDIA's way for programs to use it. | PyTorch built for CUDA 13.0, on the RTX 4060; `racecar doctor` says if it's found. |
| **TensorBoard** | A web page that charts training as it runs: rewards, losses, racing numbers, and videos. | `uv run tensorboard --logdir runs`; each run's `tensorboard/` folder ([Training](training.md)). |
| **Experiment tracking / tracker** | Recording a training run's numbers and videos as it goes, to chart them and compare runs. The tracker is where they're sent. | `Tracker`: TensorBoard now; Weights & Biases could be added. |
| **SB3** (Stable-Baselines3) | A well-tested library of RL algorithms. | First learner (M4). |
| **Hyperparameters** | Settings of the learning process itself (learning rate, batch size…), not learned by the AI. | Tuned in M4 and M8. |
| **Generalization** | Doing well on situations never seen in training. | Driving brand-new tracks (M5). |
| **Curriculum learning** | Training on easy tasks first, then harder ones, like school. | Experiment in M5. |
| **Domain randomization** | Randomly varying the world (grip, weight, sensor noise) during training so the AI copes with surprises. | M5-4 |
| **Self-play** | The AI improves by racing copies of itself. | Multi-car training (M7). |
| **ONNX** | A standard file format for trained neural networks that many programs (including browsers) can run. | Browser demo (M6). |
| **Pyodide** | Python compiled to run inside a web browser. | Browser demo option (M6). |

## Statistics (for the M8 experiments)

| Term | Plain meaning | Where it shows up here |
|------|---------------|------------------------|
| **Ablation study** | Removing one ingredient at a time to see which ones actually matter. | Which reward terms help (M8-4). |
| **Confidence interval** | A range that probably contains the true value. Wide = uncertain, narrow = confident. | Comparing our PPO with SB3. |
| **Multiple seeds** | Repeating an experiment with different random starts, since RL results vary a lot run-to-run. | 5+ seeds per comparison. |
| **IQM** (Interquartile Mean) | An average that ignores the best and worst 25% of runs, so lucky or unlucky runs don't skew it. | M8 benchmark reports. |
| **Bootstrap** | Estimating uncertainty by re-sampling your own results many times. | How the confidence intervals are computed. |
