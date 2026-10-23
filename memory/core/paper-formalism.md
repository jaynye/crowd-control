# Stochastic Motion Graph Crowds for LiDAR-Conditioned Humanoid Traversal

**Paper and figure handoff, 2026-10-14.** Part I is a compact manuscript core for a
three-page paper; Parts II–V are supporting material for the separate Astra session,
not additional manuscript pages. Equations describe the implemented system unless
explicitly marked as a general interpretation or future extension. All experimental
numbers below are existing measurements; this document introduces no new runs.

The central contribution to emphasize is a scalable crowd-generation system:
**sample human motion offline, make stochastic interaction decisions online, and
realize those decisions by sampling compatible motion-graph continuations.** Robot
learning demonstrates an application and exposes remaining limitations. Neither the
motion prior nor the pedestrian decision rules are learned in our robot-training loop.

## Part I. Compact manuscript core

### Abstract

Training humanoid traversal policies requires diverse, reactive crowds across
thousands of simulated environments. We generate human-motion libraries offline
from a pretrained motion prior and assemble them online using a stochastic motion
graph. A separate encounter controller selects persistent avoidance maneuvers, while
graph traversal balances motion continuity, requested trajectories, and configurable
behavior preferences. The resulting kinematic humans provide articulated geometry
for LiDAR observations. We formulate robot traversal as a partially observable
control problem and train a privileged PPO teacher followed by a LiDAR student.
Experiments establish inexpensive graph traversal and working policy distillation,
while identifying residual crowd stalls, motion artifacts, and robot collisions.

### 1. Offline motion, online stochastic graph

We target $E=4{,}096$ environments with $m\in[4,16]$ humans each. Let
$x_f=(p_f,\psi_f,Q_f,c_f)$ denote a human motion frame: planar root position,
body heading, body-local joint positions, and foot-contact labels. A fixed pretrained
motion prior generates future frames conditioned on executed history
$\mathcal H$ and a requested root path $\Gamma$:

$$
\widetilde x_{1:H}\sim p_\phi(\,\cdot\mid\mathcal H,\Gamma,\Psi),
\qquad e=\widetilde x_{1:L},\qquad \Delta_g=L/f_g=0.4\;\mathrm{s}.
\tag{1}
$$

Here $\Psi$ is optional body-heading conditioning. Our ARDY library uses
$f_g=20$ Hz, $H=40$ future frames, four history frames, and $L=8$ retained frames per
call. The retained frames condition the next call. The graph therefore stores short
continuations of executed motion. ARDY supplies the current library; Kimodo was
benchmarked as an alternative prior. No prior inference occurs during robot training.
The pretrained generator is credited to [Zhao et al. (2026)](https://research.nvidia.com/labs/sil/projects/ardy/).

Let $\mathcal B$ be the set of library boundary frames. At boundary $b$, define
$d_b=[\operatorname{vec}(Q_b);w_v\operatorname{vec}(V_b)]$, where $V_b$ contains
world joint velocities expressed in the boundary's heading frame and $w_v$ has units
of seconds. Planar translation and yaw are removed; absolute ground height is retained.
For each $b$, precompute its $k$ nearest boundaries with valid outgoing motion. A
candidate $e$ plays the next $L$ frames after its own start boundary $s(e)$ and ends at
$s^+(e)$:

$$
\mathcal C(b)=\{e:\ s(e)\in\operatorname{kNN}_k(d_b)\},\qquad
\mathcal G=(\mathcal B,\{b\xrightarrow{e}s^+(e):e\in\mathcal C(b)\}).
\tag{2}
$$

Thus nearest-neighbor matching determines which **continuations** are available.
The matched boundary is the start of the selected motion; the next graph state is
its endpoint. The natural continuation starts at $b$ itself and has zero descriptor
distance when eligible. This construction follows the general motion-graph idea of
recombining compatible motion segments, with prior-generated data and the selection
rule below; see [Kovar et al. (2002)](https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/MoGraphs/).

For human $i$ at graph step $n$, let $z_{i,n}$ contain its boundary, executed root,
heading, pose, and velocity. Place each candidate's relative root trajectory in that
human's world frame, obtaining $\widehat\Gamma_{i,e}$. Score it against the
controller's requested path $\Gamma_{i,n}$. For a nonempty admissible set,

$$
\begin{aligned}
C_{i,n}(e)&=
\underbrace{\sqrt{\frac1L\sum_{\ell=1}^{L}
\|\widehat p_{i,e,\ell}-\Gamma_{i,n,\ell}\|_2^2}}_{\text{root-path error}}
+w_p\frac{\|d_{b_{i,n}}-d_{s(e)}\|_2}{\sqrt J}
+w_e\|\widehat v^{\rm end}_{i,e}-v^{\rm req}_{i,n}\|_2
+\sum_a w_a c_a(i,n,e),\\
\Pr(e_{i,n}=e\mid z_{i,n},\Gamma_{i,n})&=
\frac{\mathbf1[e\in\mathcal A_{i,n}]\exp[-C_{i,n}(e)/T_g]}
{\sum_{e'\in\mathcal A_{i,n}}\exp[-C_{i,n}(e')/T_g]}.
\end{aligned}
\tag{3}
$$

$J$ is the number of joints, $\mathcal A_{i,n}\subseteq\mathcal C(b_{i,n})$ is the
admissible candidate set, and $T_g$ is temperature in cost units. Optional $c_a$
terms prefer forward travel, body turns, persistent posture, velocity continuity,
or neighbor clearance. Optional masks reject excessive vertical motion or planted
foot/body twisting. Gumbel sampling implements (3); $T_g=0$ selects a minimum-cost
candidate. An empty admissible row holds its current pose and boundary.

Execution applies bounded root-path correction and a decaying pose offset, then sets
$b_{i,n+1}=s^+(e_{i,n})$. The reference setting uses $k=128$, $T_g=0.01$ m,
$w_v=0.1$ s, $w_p=1$, $w_e=0.25$ s, and corrections limited to $5^\circ$ and
$10\%$ speed. Graph selection occurs every $0.4$ simulated seconds; interpolation
supplies poses between boundaries. This selection interval is distinct from the
time required for a commanded stop or turn to execute.

### 2. Reactive stochastic encounters

Body-motion nodes are separate from behavior modes. Human $i$ carries a goal,
persistent traits $\theta_i$, and a committed mode
$\beta_i\in\{\mathrm{cruise},\mathrm{sidestep},\mathrm{veer},\mathrm{pivot},\mathrm{yield}\}$,
together with its phase, target, and timers. Intended velocities predict closest
approach to nearby humans and perceived external agents. A threat is nearby,
approaching, and predicted to enter a passing margin. For urgency $u_i\in[0,1]$ and
effective assertiveness $\bar a_i\in[0,1]$, an eligible cruising human intervenes with
probability

$$
\lambda_i=\lambda_0(1-\bar a_i)u_i^\kappa,\qquad
\Pr(\mathrm{switch\ in}\ \delta t)=1-\exp(-\lambda_i\delta t).
\tag{4}
$$

The controller samples a maneuver and retains it through its completion conditions.
Veering requests an arc; pivoting requests body reorientation with zero target
translation. A stop rule suppresses predicted closing motion at a configurable
clearance, initially $0.01$ m. The graph's soft clearance cost discourages continuations
that violate that request. Finite graph coverage and execution lag prevent a hard
collision-free guarantee. Two humans may intervene simultaneously.

Robot perception has a separate reset trait: oblivious, stubborn, or cooperative.
Stubborn humans obey stopping but do not initiate avoidance for the relevant agent;
oblivious humans ignore the robot. The decisive human–human configuration additionally
retires human-directed stubbornness after finite stalled patience. All stochastic
draws for the encounter and graph controllers use per-human streams. These are
configurable synthetic behavior rules, not fitted human interaction statistics.

### 3. Robot POMDP and teacher–student training

We model traversal as
$\mathcal P=(\mathcal S,\mathcal A,\mathcal T,\mathcal O,\Omega,r,\gamma,\rho_0)$.
The latent state $s_t$ includes robot configuration and velocity, all humans'
continuous and discrete controller states, active motion and observation buffers,
phases and command timers, previous action, goals, scene geometry, and randomized
dynamics. The robot chooses joint-position actions
$a_t$; simulation and the reactive crowd induce

$$
s_{t+1}\sim\mathcal T(\cdot\mid s_t,a_t),\qquad
o^S_t\sim\Omega(\cdot\mid s_t),\qquad
J(\pi)=\mathbb E_{\rho_0,\mathcal T,\pi}\!
\left[\sum_{t=0}^{D-1}\gamma^t r_t\right].
\tag{5}
$$

$D$ is episode length. Partial observability comes from occlusion, finite sensing,
noise, and hidden human goals and traits. The POMDP terminology follows
[Kaelbling et al. (1998)](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf).
Current feedforward policies use proprioception, the goal-derived velocity command,
the previous action, and four LiDAR scans. The teacher adds the six nearest humans'
relative positions, relative velocities, clearances, and robot plaza pose:

$$
o^S_t=(o^{\rm prop}_t,c^{\rm robot}_t,a_{t-1},\ell_t,\ell_{t-5},\ell_{t-10},\ell_{t-15}),
\qquad o^T_t=(o^S_t,\chi_t),\qquad a_t\sim\pi_\vartheta(\cdot\mid o^T_t).
\tag{6}
$$

Indices in (6) count $50$ Hz robot-control steps. The teacher has privileged
observations, but does not observe every latent variable in $s_t$. Train it with
[PPO](https://arxiv.org/abs/1707.06347), using a separate critic with privileged
observations $o^V_t$:

$$
\begin{aligned}
L_{\rm clip}(\vartheta)&=\mathbb E_t\!left[
\min\!\left(\varrho_t\widehat A_t,
\operatorname{clip}(\varrho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right],\\
\varrho_t&=\frac{\pi_\vartheta(a_t\mid o^T_t)}
{\pi_{\vartheta_{\rm old}}(a_t\mid o^T_t)},\qquad
\min_{\vartheta,\omega}\ -L_{\rm clip}
+c_VL_V(\omega)-c_H\mathbb E_t\mathcal H[\pi_\vartheta(\cdot\mid o^T_t)].
\end{aligned}
\tag{7}
$$

The reward combines the inherited locomotion objective, velocity tracking, goal
heading, and penalties for human proximity/contact and wall proximity. Robot–human
penetration beyond $0.05$ m terminates an episode; terminal collision penalties are
an evaluated optional reward change.

Freeze the trained teacher and collect fresh trajectories with the student acting.
At each visited state, query the teacher's mean action and regress the student's mean:

$$
a_t=\mu_\eta(o^S_t)+\varepsilon_t,\quad
\varepsilon_t\sim\mathcal N(0,0.1^2I),\qquad
L_{\rm IL}(\eta)=\mathbb E_{(o^S,o^T)\sim d^{\pi_\eta}}
\left[\frac1{d_a}\|\mu_\eta(o^S)-\operatorname{sg}[\mu_{\vartheta^*}(o^T)]\|_2^2\right].
\tag{8}
$$

Here $d^{\pi_\eta}$ denotes the student's rollout distribution, $d_a$ is action
dimension, and $\operatorname{sg}$ stops gradients. This is online imitation on
student-visited states, related to
[DAgger](https://proceedings.mlr.press/v15/ross11a.html); our runner refreshes its
rollout buffer rather than retaining an aggregate dataset. Pure distillation uses no
reward gradient. Direct PPO on $o^S$ is the baseline. Optional PPO continuation from
the distilled student is implemented; a simultaneous PPO-plus-imitation objective
is a future extension.

### 4. Results and limits

Experiments use NVIDIA A40 GPUs. The original 18.2-hour ARDY library took approximately
two minutes to generate. The reference graph advanced 65,536 humans through $0.4$ s
of motion in approximately $47.5$ ms, measured in isolation. In the historical
held-out command benchmark, mean root-path errors across scenarios were $0.7$–$1.4$
cm, with median responses $0$–$0.4$ s slower than online ARDY. These numbers describe
the original graph and benchmark, not the later encounter controller.

On three development seeds, the decisive encounter candidate reduced close-pair
stalls from 43 to 19 and their maximum duration from $5.2$ to $2.4$ s. Joint-sphere
contact frames fell from $1.36\%$ to zero, but individual wall/obstacle stops reached
$48$ s and motion-quality metrics worsened. This remains a development configuration.

Mean-action robot replays use 512 environments for 1,500 control steps, with a
robot-aware crowd of eight humans. Episode ends include collisions, falls, and timeouts.

| Policy | Training input | Collisions / episode ends | Falls / episode ends |
|---|---|---:|---:|
| Direct PPO, a2 | LiDAR | 101 / 581 (17.4%) | 1 / 581 (0.2%) |
| Teacher, t7 | LiDAR + privileged | 123 / 586 (21.0%) | 0 / 586 (0%) |
| Distilled student, s7 | LiDAR | 96 / 585 (16.4%) | 23 / 585 (3.9%) |
| Direct PPO + collision penalty, p250 | LiDAR | 82 / 557 (14.7%) | 0 / 557 (0%) |

Teacher and student collision differences are within the observed checkpoint spread
of roughly five percentage points; this spread is not a confidence interval.
Privileged inputs have not established an advantage over direct PPO. Distillation
works, with reduced goal arrivals and increased falls. The moderate collision penalty
gives a small observed reduction; a larger penalty produces falls in $84.5\%$ of
replay episode ends. These results support scalable synthetic crowd generation and
a functioning learning pipeline, while leaving reliable avoidance and crowd progress
unresolved.

## Part II. Mathematical details for authoring and figure labels

### A. Notation and state completeness

| Symbol | Meaning |
|---|---|
| $f,n,t$ | Motion-frame, graph-boundary-step, robot-control-step indices |
| $\phi$ | Frozen human-motion prior parameters |
| $\vartheta,\omega,\eta$ | Teacher actor, critic, student actor parameters |
| $b,s(e),s^+(e)$ | Current graph boundary, candidate start boundary, endpoint |
| $Q_b,V_b,d_b$ | Body-local joint positions, heading-aligned joint velocities, descriptor |
| $\Gamma,\Psi$ | Requested planar root path and optional body headings |
| $T_g,\lambda_i,\gamma$ | Graph temperature, behavior switching rate, robot discount |
| $\beta_i,\tau_i,\theta_i,h_i$ | Behavior mode, mode age, persistent traits, optional posture latent |
| $\chi_t,o^V_t$ | Teacher's extra features, critic observations |

One sufficiently rich human state at a graph boundary can be written

$$
y_i=(z_i,g_i,\beta_i,\tau_i,\theta_i,h_i,\nu_i),\qquad
z_i=(b_i,p_i,\psi_i,Q_i,v_i).
$$

$g_i$ is the goal/route condition. $\nu_i$ bundles maneuver phase, retained heading,
deadlines, blocked/stall timers, and any other controller memory. Between boundaries,
also include the active motion buffer and its playback phase. Human positions alone
are not a Markov state. Similarly, the boundary index alone does not determine graph
transition probabilities: the requested path, executed world state, and optional
style/neighbor features change the costs. This is a controlled hybrid process;
the graph topology is fixed while its transition probabilities depend on context.
Behavior holding times depend on phase, duration, and completion guards, so the
behavior process is semi-Markov when timers are omitted from its state.

Independent per-human random draws do not imply independent trajectories: every
human reacts to others. Counter-based streams preserve draws under batch permutation
and composition changes. They do not preserve trajectories if the interacting scene
changes. The legacy random-goal `PlazaBehavior` still uses a shared generator; the
independence statement applies to the encounter controller and graph sampling.

The implemented base descriptor uses one boundary pose and a one-frame velocity
estimate. Contact labels are stored for playback and diagnostics; they are not
components of this kNN descriptor. Do not describe the current matching rule as
contact-constrained matching over a multi-frame temporal window.

### B. Exact graph sampling and execution

The Gumbel-max implementation of (3) is

$$
e^*=\underset{e\in\mathcal A}{\arg\max}\left[-C(e)/T_g+g_e\right],\qquad
g_e=-\log(-\log U_e),\quad U_e\sim\mathrm{Uniform}(0,1).
$$

Each human's seed, counter, and candidate slot determine its pseudorandom uniforms.
No learned transition network or fitted Markov matrix is involved. The frozen prior
supplies motion support; online heuristics and the energy determine traversal.
There is no claim that graph rollouts exactly sample $p_\phi$ or preserve its
stationary distribution.

Let $\Delta p^e_\ell$ be the saved root displacement from the candidate's start,
$\Delta p^e_0=0$, $\alpha$ the bounded yaw correction, and $\sigma$ the bounded speed
scale. With planar yaw rotation $R$, the implemented incremental bend is

$$
\begin{aligned}
\widetilde p_\ell&=p_i+R(\psi_i)\sum_{r=1}^{\ell}
\sigma R(\alpha r/L)(\Delta p^e_r-\Delta p^e_{r-1}),\\
\widetilde\psi_\ell&=\psi_i+\Delta\psi^e_\ell+\alpha\ell/L,\\
\widetilde Q_\ell&=Q^e_\ell+
\left[1-3u_\ell^2+2u_\ell^3\right]\,(Q_i-Q_{s(e)}),\quad
u_\ell=\frac{\ell}{L+1}.
\end{aligned}
$$

Yaw and speed correction are gated off for small requested or candidate displacement;
turn-preserving settings can disable yaw bending during a committed body turn.
Candidate costs are evaluated before this correction. At finite $L$ the pose offset
is small but not exactly zero on the final frame. This blending does not establish
velocity continuity, planted-foot constraints, or physical validity. Absolute local
joint height is preserved in the coordinate transform; flat-ground masks provide
additional curation rather than projecting every pose onto the floor.

### C. Optional cost terms and posture

With $v^e_\ell$ the native candidate velocity and $\bar\psi^e_\ell$ its adjacent-frame
midpoint body heading, the forward-travel cost is

$$
c_{\rm forward}(e)=\frac1L\sum_{\ell=1}^{L}
\left[\|v^e_\ell\|_2-v^e_\ell\cdot f(\bar\psi^e_\ell)\right]_+,
\quad f(\psi)=(\sin\psi,\cos\psi).
$$

Forward curves have low cost; sideways and backward travel cost approximately one
and two times speed. Other implemented terms are terminal heading error weighted
by requested speed, a speed-independent retained-turn heading error, first-frame
velocity mismatch, and squared torso-pitch deviation from a style target. During
evasion the controller can relax forward/heading preferences without relaxing posture.

The clearance cost penalizes *additional* predicted intrusion relative to standing:

$$
c_{\rm clear}(i,e)=\sum_j
\left[[d^*_{ij}-d^{\min}_{ij}(e)]_+
-[d^*_{ij}-d^{\min}_{ij}(\mathrm{stay})]_+\right]_+,
$$

where neighbor trajectories extrapolate executed velocities over candidate frames.
Only supplied neighbors are considered, including wall proxies and perceived robot
agents. The term is soft and is not a formal safety certificate. The end-velocity
cost uses a three-frame finite difference at the end of the edge/request.

Optional posture $h_i\in\{U,H\}$ follows a two-state continuous-time Markov chain
observed at graph boundaries. For rates $a:U\to H$ and $b:H\to U$,

$$
G_h=\begin{bmatrix}-a&a\\b&-b\end{bmatrix},\quad
P_h(\Delta_g)=\exp(G_h\Delta_g),\quad
P_{UH}=\frac{a}{a+b}(1-e^{-(a+b)\Delta_g}),\quad
P_{HU}=\frac{b}{a+b}(1-e^{-(a+b)\Delta_g}).
$$

At reset, $h_i\sim\mathrm{Bernoulli}(p_{H,0})$; the first edge preserves that draw.
Zero rates preserve style. The initial distribution and transition rates are separate
controls, with a separate `graph_posture` random stream. A soft style preference does
not guarantee the realized pose class. The pitch split is a geometric diagnostic,
not a calibrated semantic label; posture need not appear in the three-page paper.

### D. Encounter hazard and transition guards

For human–human interactions, let $r_{ij}=p_j-p_i$ and
$w_{ij}=v^{\rm intent}_j-v^{\rm intent}_i$. With forecast horizon $\tau_c$,

$$
t^*_{ij}=\operatorname{clip}\!\left(-\frac{r_{ij}^{\mathsf T}w_{ij}}
{\|w_{ij}\|_2^2},0,\tau_c\right),\qquad
d^{\rm CPA}_{ij}=\|r_{ij}+t^*_{ij}w_{ij}\|_2.
$$

For zero relative velocity use $t^*=0$. A threat satisfies
$\|r_{ij}\|<R_b$, $r_{ij}^{\mathsf T}w_{ij}<0$, and
$d^{\rm CPA}_{ij}<2r_h+d_{\rm pass}$. If $d_i$ is the smallest current distance to
a threat, $u_i=\operatorname{clip}((R_b-d_i)/(R_b-2r_h),0,1)$; without a threat,
$u_i=0$. For an ordinary non-stubborn human, blocked duration $b_i^{\rm time}$ gives
$\bar a_i=a_i\exp(-b_i^{\rm time}/\tau_p)$. Active human-directed stubbornness sets
$\bar a_i=1$ for human encounters until the trait is retired by finite stall patience.
External-agent radii and separately sampled robot traits adjust these tests.

After switching, maneuver weights interpolate with urgency:

$$
q_k(u)=\frac{(1-u)w_k^{\rm far}+uw_k^{\rm near}}
{\sum_{k'}[(1-u)w_{k'}^{\rm far}+uw_{k'}^{\rm near}]}.
$$

Feasibility and blocked-state rules modify this distribution, e.g. suppressing yield
when already blocked. Optional forced stall recovery can bypass the stochastic
hazard. In the decisive candidate, a geometric search chooses small feasible heading
deviations; this minimizes a local predicted heading change, not executed detour length.

Use the following transition semantics in the diagram:

| Source | Destination | Guard / meaning |
|---|---|---|
| Cruise | Sidestep / veer / pivot / yield | Eligible threat, Bernoulli hazard, then feasible maneuver draw |
| Veer or pivot, turn phase | Same mode, pass phase | Body aligned with retained target, or turn timeout |
| Sidestep | Cruise | Maneuver duration elapsed, or wall/obstacle guard |
| Veer or pivot, pass phase | Cruise | Hold completed; with opt-in pass commitment, direct goal path clear or deadline; wall guard can end early |
| Yield | Cruise | Timeout, or previous threat clear after minimum dwell |
| Stalled execution | Recovery pivot | Opt-in mutual-stall recovery or retirement of human-directed stubbornness |

Waiting at a goal belongs to conditioning. Ordinary turn-before-walk also belongs
to conditioning/turn state within cruising; it is not a sixth encounter mode.
Stopping and separation override the commanded motion across modes, rather than
forming additional values of $\beta$. A stopped person may continue to pivot.

### E. POMDP interpretation and optimization details

A general POMDP policy may use the full history
$\mathfrak h_t=(o_0,a_0,\ldots,o_t)$ or belief
$b_t(s)=\Pr(s_t=s\mid\mathfrak h_t)$. Our policy uses the finite observation vector
in (6), not an explicit Bayesian filter; four LiDAR scans do not prove that this
vector is a sufficient Markov statistic. The teacher and critic also use features
of state rather than the full simulator state. A reactive robot-aware crowd makes
the crowd transition depend on robot actions through the evolving scene.

For PPO, $\widehat A_t$ is a generalized advantage estimate from the privileged
critic. Writing $\delta_t=r_t+\gamma(1-d_t)V_\omega(o^V_{t+1})-V_\omega(o^V_t)$,
$\widehat A_t=\sum_{j\ge0}(\gamma\lambda_{\rm GAE})^j\delta_{t+j}$ within an
uninterrupted rollout, with episode-boundary masks and timeout bootstrapping handled
by the runner. $L_V$ is PPO's clipped value-regression loss. Policies and critic are
MLPs with widths $(512,256,128)$, ELU activations, and normalized observations.
Teacher PPO uses a learned scalar Gaussian standard deviation; student rollout
noise is fixed at $0.1$. Deployment/replay uses mean actions. Actions parameterize
scaled joint-position targets for the G1's low-level controllers.

| Setting | Implemented reference |
|---|---|
| Robot control / motion output / graph selection | 50 Hz / 20 Hz / 2.5 Hz |
| Training population | 4,096 environments × 8 humans |
| Robot episode | 20 s, or earlier collision/fall termination |
| LiDAR | 144 azimuth × 12 elevation rays; minimum horizontal range per each of 72 azimuth bins |
| LiDAR history | Four scans spaced by 5 control steps (0.1 s); 0.3 s temporal span |
| Range preprocessing | Floor hits below 0.1 m excluded; misses and far returns map to 6 m; normalized ranges |
| Actor / teacher inputs | 387 / 421 scalars in the reference eight-human configuration |
| Teacher privileged features | Six humans × five features, plus normalized plaza position and heading sine/cosine |
| PPO | 24 steps/env, 5 epochs, 4 minibatches; $\epsilon=0.2$, $\gamma=0.99$, $\lambda_{\rm GAE}=0.95$ |
| PPO optimizer | Initial learning rate $10^{-3}$, adaptive KL schedule (target 0.01); entropy coefficient 0.01 |
| Distillation | MSE; Adam at $5\times10^{-4}$; 24 steps/env; 2 learning epochs |

The student imitates actions, not graph edges, human poses, rewards, or privileged
state. Its observation bottleneck remains: multiple hidden situations can yield the
same LiDAR input but different teacher actions. Small action loss alone therefore
does not certify matching navigation or stability.

For concise reward formalism, let $\delta_{it}$ be the minimum clearance between
13 robot body spheres and human $i$'s joint spheres, and $\delta_t=\min_i\delta_{it}$.
The crowd terms in the baseline are

$$
r_t=\Delta_r\left[
r_{\rm locomotion,t}
+\frac{1+\cos e^{\rm goal}_t}{2}\mathbf1[\text{moving command}]
-\sum_i\operatorname{clip}\!\left(\frac{0.4-\delta_{it}}{0.4},0,1\right)
-10\mathbf1[\delta_t<0]-c_{\rm wall,t}
+w_{\rm end}\mathbf1[\text{collision end}]
\right],\qquad \Delta_r=0.02\;\mathrm{s}.
$$

$r_{\rm locomotion,t}$ includes the inherited velocity-tracking and locomotion
regularizers; this shorthand does not replace their implementation. Collision ends
when $\delta_t<-0.05$ m. $w_{\rm end}=0$ in a2/t7/s7; the p250/p1000 experiments use
$-250/-1000$, so their actual terminal return penalties are $-5/-20$, respectively.
These clearances are geometric proxies, not rigid-mesh physical contacts. Humans
are kinematic; there is no learned human balance or pushing response. LiDAR uses
full-resolution rigid Core-skin pieces with paired graph rotations.

`--finetune` initializes the actor from the student and the critic from its teacher,
then runs PPO with a fresh optimizer. Despite older notes calling this “RL plus
imitation,” no ongoing $L_{\rm IL}$ term is implemented there. A future combined loss
would be $L_{\rm PPO}+\lambda_{\rm IL}L_{\rm IL}$; do not present that as a measured arm.

## Part III. Figure specifications for the Astra session

### Figure 1: motion graph and online crowd control (primary figure)

Use two clearly separated panels. The upper panel is the body-motion graph; the
lower panel is the encounter controller. Arrows from the lower panel provide
$\Gamma_i$, a body target, and priorities to graph scoring. This prevents confusing
behavior labels with graph nodes.

**Upper panel content:**

1. At left, a frozen prior $p_\phi$ generates several short motion strips offline.
   Mark every $L=8$ frames as a possible continuation and circle the preceding
   boundary frame. Include position/velocity descriptors beside boundary circles.
2. Show an active node $b_i$ and three illustrative eligible continuations. Label
   them $e_0$, $e_1$, $e_2$, beginning at matched boundaries $s(e_k)$ and ending at
   $s^+(e_k)$. Use a visual distinction between an implied descriptor match and the
   executed continuation. The mathematical graph arc is
   $b_i\xrightarrow{e_k}s^+(e_k)$.
3. Label $e_0$ as the natural continuation ($s(e_0)=b_i$). Cross-stream branches are
   other candidates. Use faint extra branches or “$k=128$” to make clear that three
   is schematic, not the actual candidate count.
4. Put the requested root path beside the candidate paths. On each candidate show
   $C_i(e_k)$ and $P_i(e_k)\propto e^{-C_i(e_k)/T_g}$. Highlight one sampled branch.
   Use symbolic probabilities, not invented measured values or visitation counts.
5. After selection, show “bounded root correction + pose-offset decay,” then the
   endpoint and next decision. Annotate $\Delta_g=0.4$ s. Add a small “hold current
   pose if no admissible candidates” loop, clearly a runtime fallback.

**Lower panel content:** use the transition table in II.D, with the five modes.
Show veer/pivot as turn and pass subphases if space permits. Draw stop/separation as
an override bar across modes, and the optional recovery pivot as a dashed arrow.
Place $1-e^{-\lambda_i\delta t}$ on cruise-to-intervention arrows. Show goals,
neighbor observations, and persistent traits as controller inputs. Two separate
human controllers can both switch; introduce no pairwise winner or tie-break.

**Suggested caption:** “Offline motion is recombined online through compatible
boundary states. Encounter decisions specify requested paths and body targets;
cost-conditioned graph sampling selects the next articulated continuation. Behavior
modes and motion nodes occupy different levels of the controller.”

### Figure 2: teacher–student learning (small secondary figure)

Show one simulator producing a sensor branch $o^S$ and an additional privileged
branch $\chi$. During teacher training, $(o^S,\chi)$ enters the teacher actor;
$o^V$ enters the critic; reward/advantage feeds PPO. During distillation, freeze the
teacher, execute student actions, and query teacher mean actions at the same visited
states. Draw the MSE link between means, with gradients only into the student. During
deployment only $o^S\to\pi_\eta\to a$ remains.

Label humans “stochastic kinematic motion graph,” separately from the learned robot
policy. The human-motion prior remains outside the simulation/training loop. Do not
draw a gradient through the crowd, a learned human behavior policy, full-state access
for the teacher, or a concurrent PPO+IL student update.

If only one figure fits, prioritize Figure 1 and state (5)–(8) in text.

## Part IV. Results provenance and claim boundaries

These references support numerical claims and allow author verification. Keep their
different populations, scenes, and graph versions visible when assembling results.

| Evidence | Source of record | Scope |
|---|---|---|
| Prior feasibility | [Backend benchmark](../experiments/2026-09-30-motion-backend-benchmark.md) | ARDY/Kimodo isolated generation; text-free conditions |
| Graph speed/tracking | [Library and graph experiment](../experiments/2026-09-30-ardy-library-graph.md) | Original 18.2 h library, reference $k=128$ graph; held-out seed 1, 128 humans for quality |
| Turning coverage | [Turning supplement](../experiments/2026-10-06-turning-library.md) | Additional 9.1 h of prior-generated turning motion; distinct later graph |
| Encounter stalls | [Decisive encounters](../experiments/2026-10-12-encounter-decisive.md) | Incumbent vs candidate, 16 humans, 60 s, development seeds 0/1/2 |
| Robot learning | [Teacher/student experiment](../experiments/2026-10-12-robot-aware-teachers.md) | a2 `model_3999`, t7 `model_2000`, s7 `model_1499`; include the 2026-10-14 fall correction |
| Collision reward | [Penalty experiment](../experiments/2026-10-13-collision-penalty.md) | p250/p1000 `model_3499`; checkpoint checks for p250 at 2500/3000 |
| Rigid human geometry | [Geometry benchmark](../experiments/2026-10-05-body-geometry-benchmark.md) | Sensing/posing microbenchmarks and separate plaza integration timings |
| Collision and fall rates, b0, the fall fix | [Fall fix and b0](../experiments/2026-10-18-fall-fix-and-b0.md) | `crowd eval traversal`, last 3 checkpoints, 4,096 envs × 60 s, paired goals, aware and blind crowds |
| Planning arms over frozen SONIC | [Planning arms](../experiments/2026-10-19-sonic-planning-arms.md) | Gate without a crowd; P0 and the six trained arms (PPO, teacher, student at 1 and 2 Hz) by `crowd eval traversal` (last 3 checkpoints, 4,096 envs × 60 s, aware and blind crowds; one training seed each) |

The pair-stall comparison uses a root-gap cutoff of $1.1$ m, speeds below $0.15$ m/s,
duration at least $1.2$ s, and a pending goal; the 43→19 metric includes stalled
maneuvers. Zero joint-sphere contacts in these traces does not establish zero full-skin
contacts, general collision freedom, or robot safety. Held-out seeds 101/307/911
showed zero joint-sphere contacts but a $47.2$ s individual stop; there is no paired
held-out incumbent evaluation. They have now been inspected and are not unseen data.

Since 2026-10-19, report collisions and falls per robot-minute from `crowd eval traversal`
(both crowds, with encounters and goals per moving minute); they supersede the collision
share below for new results. Historically, report collision share as $N_{\rm collision}/N_{\rm episode\ ends}$, with falls,
goal arrivals, speed, and evaluation horizon alongside it. It is not collision risk
per encounter or per meter. The p1000 row must never be shown as a safety improvement:
1,081 of 1,279 episode ends were falls. Student s7 also has a measurable stability cost.
No real-world transfer or calibrated human-likeness result has been established.

### Throughput interpretation

The original graph measurement corresponds to

$$
R_{\rm graph}=\frac{65{,}536\times0.4}{0.0475}
\approx5.52\times10^5\;\frac{\text{human-motion s}}{\text{wall s}}.
$$

At simulator speed $S$ simulated seconds per wall second, human-motion demand is
$EmS$. For $m=16$ and $S=0.4$, the isolated stepping time is about $4.75\%$ of one
wall second. This is an isolated timing-derived fraction, not a measurement of
concurrent GPU utilization or full training throughput. Storage and per-human state
are separate: all environments share graph tables, while runtime candidate scoring
is linear in population and candidate count for fixed edge length and skeleton.

For an online prior with batch size $B$, new horizon $H_s$ seconds, call time $T_B$,
and consumed fraction $u$, usable capacity is $uBH_s/T_B$. The historical ARDY
benchmark gives roughly $2{,}800$ new-motion seconds per wall second, and replan-every-
$0.4$ s consumes $u=0.2$. At $m=16,S=0.4$ this implies approximately 47 A40s of
isolated generation, consistent with the original report's rounded estimate of 46.
The generators' load time is excluded. Avoid turning this capacity comparison into
an end-to-end training speedup claim.

The initial traversal smoke check at 4,096 × 8 humans recorded 76.4 ms per environment
step (53.6k individual environment transitions/s); the first teacher training ran at
about 3.3 s per PPO iteration on an A40. These observations include a different
graph/controller, robot, sensing, and learning workload from the original graph
microbenchmark. They establish the implemented eight-human training configuration,
not an integrated 16-human comparison against online ARDY. The smoke check predates
the robot-aware crowd; [current status](status.md) retains this integration result.

### Reproducibility anchors

- Original graph speed: `runs/20261001T010546Z_graph_speed/run.json`; A40 on consu,
  driver 580.178.04, torch 2.9.0+cu128, CUDA 12.8, Python 3.13.15. This predates the
  current repository organization. Use its record rather than today's checkout to
  describe that experiment.
- Teacher: `runs/20261012T214348Z_train_traversal_teacher_t7/run.json`; A40 on qt-1,
  driver 595.84, seed 42, repository base `5c1ef9c` plus saved dirty source,
  fingerprint `a8597c69c262a3d499cd8f029786f3eb6f581febcdf38565e7c8d1bec7981682`.
- Student: `runs/20261012T234846Z_train_traversal_student_s7/run.json`; A40 on major,
  driver 580.178.04, seed 42, same repository base with source fingerprint
  `503d024d329224698bd0cb52ea391f0684346e4709b624f4ff5ff201daab6aa8`.
- ARDY source pin `693f74d13b3d04a0a22ce127ee79c929dd89756b`; H40 checkpoint revision
  `abe6c43beb28c867c950acb824b9c4ef3d63fb76`. Original library generation uses
  five diffusion steps, TF32, and no classifier-free guidance or semantic prompts.
- Current environment pins torch 2.9.0 and Warp 1.14.0. Historical `run.json`, source
  snapshots, training configuration, and checkpoint records define each experiment;
  a base git commit alone does not reconstruct the dirty training source.

## Part V. Three-page allocation, code map, and citations

For a two-column short paper, use approximately: page 1, abstract/problem plus graph
construction and Figure 1; page 2, encounter rule and robot-learning formalism, with
Figure 2 only if space permits; page 3, results table, limitations, and references.
Start with Part I, then shorten to the venue template. Keep equations (2)–(4) and
(8) if space is tight; describe the standard PPO objective in prose. Parts II–IV are
author/figure support and should not be pasted into the three-page manuscript.

| Formalism | Implementation |
|---|---|
| Offline prior sampling / executed-history library | [generate.py](../../src/crowdmotion/library/generate.py), [sources.py](../../src/crowdmotion/behavior/sources.py) |
| Boundary descriptors / candidate construction | [tables.py](../../src/crowdmotion/graph/tables.py), [build.py](../../src/crowdmotion/graph/build.py) |
| Graph energy, masks, Gumbel sampling, correction | [costs.py](../../src/crowdmotion/graph/costs.py), [runtime.py](../../src/crowdmotion/graph/runtime.py), [annotations.py](../../src/crowdmotion/graph/annotations.py) |
| Optional persistent posture | [posture.py](../../src/crowdmotion/graph/posture.py) |
| Encounter decisions and arcs | [encounter.py](../../src/crowdmotion/behavior/encounter.py), [planner.py](../../src/crowdmotion/behavior/planner.py) |
| One shared crowd loop | [controller.py](../../src/crowdmotion/behavior/controller.py) |
| Robot observations, rewards, distillation configuration | [traversal.py](../../src/crowdmotion/sim/mjlab/traversal.py) |
| Teacher/student checkpoint flow and PPO continuation | [train.py](../../src/crowdmotion/sim/mjlab/train.py) |
| PPO hyperparameters | [rl_cfg.py](../../mjlab/src/mjlab/tasks/velocity/config/g1/rl_cfg.py) |

Additional inspected dependency: installed RSL-RL `algorithms/distillation.py`
(`act`, `update`, `compute_returns`) confirms student-driven collection, detached
teacher means, MSE, and no return-based optimization during distillation.

Suggested primary references, verified 2026-10-14:

1. Zhao et al. **ARDY: Autoregressive Diffusion with Hybrid Representation for
   Interactive Human Motion Generation.** ACM TOG / SIGGRAPH, 2026.
   [Project and bibliography](https://research.nvidia.com/labs/sil/projects/ardy/).
2. Kovar, Gleicher, and Pighin. **Motion Graphs.** SIGGRAPH, 2002.
   [Author project page and paper](https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/MoGraphs/).
3. Kaelbling, Littman, and Cassandra. **Planning and Acting in Partially Observable
   Stochastic Domains.** Artificial Intelligence 101, 1998.
   [Author-hosted paper](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf).
4. Schulman et al. **Proximal Policy Optimization Algorithms.** 2017.
   [Paper](https://arxiv.org/abs/1707.06347).
5. Ross, Gordon, and Bagnell. **A Reduction of Imitation Learning and Structured
   Prediction to No-Regret Online Learning.** AISTATS, 2011.
   [Proceedings](https://proceedings.mlr.press/v15/ross11a.html).

These references establish the pretrained prior and standard methodological context.
They do not substantiate our numerical results; the project experiment records above
do. Avoid claiming the motion-graph concept, POMDP formulation, PPO, or imitation
learning as new algorithms introduced by this project.
