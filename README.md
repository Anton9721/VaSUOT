# VaSUOT: Variational Semi-Unbalanced Entropic Optimal Transport

VaSUOT is a research implementation of **semi-unbalanced entropic optimal
transport**. It learns a stochastic conditional transport plan from independent
samples of a source distribution $p_0$ and a reference target distribution
$p_1$.

Unlike balanced OT, VaSUOT fixes only the source marginal. The transported
target marginal is softly matched to $p_1$ through a KL penalty. This is useful
when target matching should tolerate missing modes, population shifts, or
imperfect reference data.

## Problem

For quadratic cost $c(x,y)=\tfrac12\lVert x-y\rVert^2$, entropic strength
$\varepsilon>0$, and target-relaxation weight $\tau>0$, VaSUOT solves

$$\min_{\pi:\,\pi_0=p_0}\mathbb{E}_{(x,y)\sim\pi}[c(x,y)]-\varepsilon H(\pi)+\tau\,\mathrm{KL}(\pi_1\Vert p_1).$$

The parameter $\tau$ controls the trade-off between transport cost and
agreement with the reference target marginal. Larger $\tau$ enforces closer
matching to $p_1$.

## Variational objective

The method parameterizes a dual potential $g_\theta$ and an auxiliary
normalizer $\xi_\phi$. With $z\sim\mathcal N(0,I)$, training maximizes

$$\mathcal L(\theta,\phi)=-\tau\,\mathbb E_{y\sim p_1}[e^{g_\theta(y)}]-\varepsilon\,\mathbb E_{x\sim p_0}[\xi_\phi(x)]-\varepsilon\,\mathbb E_{x\sim p_0,z}\left[\exp\left(-\frac{\tau}{\varepsilon}g_\theta(x+\sqrt{\varepsilon}z)-\xi_\phi(x)\right)\right].$$

Training uses minibatches from $p_0$, $p_1$, and Gaussian noise; it does not
require MCMC samples from the learned transport plan. The auxiliary network
$\xi_\phi$ is used only during training.

## Conditional inference

After learning $g_\theta$, VaSUOT samples

$$q_\theta(y\mid x)\propto\exp\left(-\frac{\tfrac12\lVert x-y\rVert^2+\tau g_\theta(y)}{\varepsilon}\right).$$

with unadjusted Langevin dynamics (ULA):

$$y_{s+1}=y_s+\frac{\eta}{\varepsilon}\left(x-y_s-\tau\nabla g_\theta(y_s)\right)+\sqrt{2\eta}\,z_s,\qquad z_s\sim\mathcal N(0,I).$$

ULA is approximate: its step size and number of steps must be evaluated for
each dataset and potential geometry.

## Benchmarks and evaluation

VaSUOT is evaluated on three two-dimensional synthetic settings and one latent
image-translation setting:

- **Class-imbalanced Gaussian mixtures:** tests how the target KL relaxation
  changes transported component masses.
- **Target outlier contamination:** tests whether the solver avoids assigning
  excessive mass to a distant target outlier.
- **Gaussian to Swiss Roll:** tests complex transport geometry and conditional
  stochasticity.
- **FFHQ ALAE latent translation:** evaluates unpaired attributes such as
  Male to Female and Adult to Child after decoding transported latents.

Common metrics are paired transport cost, empirical target $W_2$, barycentric
projection error against a numerical reference, solver/inference runtime, and
peak memory. Synthetic experiments additionally report component mass, outlier
mass, Energy Distance, and conditional variance. Latent image translation uses
FID and LPIPS alongside runtime.

The primary baselines are semi-unbalanced Sinkhorn, implemented with POT as a
discrete numerical reference, and SU-EgNOT, an MCMC-trained neural solver for
the same semi-unbalanced objective. Image translation also compares against
the original balanced VarEOT setup. Main results are aggregated over independent
random seeds; ablations vary $\tau$, $\varepsilon$, the Monte Carlo count $K$,
network width, and ULA hyperparameters.

## Code layout

- src/config.py - typed experiment configuration.
- src/models.py - scalar networks for $g_\theta$ and $\xi_\phi$.
- src/samplers.py - independent empirical and synthetic samplers.
- src/objectives.py - SU-VarEOT variational objective and diagnostics.
- src/trainer.py - optimization, safeguards, and EMA potential.
- src/inference.py - ULA conditional sampling and diagnostics.
- notebooks/ - original Swiss Roll, MSCI, and ALAE experiments.

The source modules form a shared solver layer; experiment-specific preprocessing
and evaluation remain outside this core.
