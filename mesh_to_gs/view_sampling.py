"""Training-only view priorities and reproducible camera/minibatch sequences."""
import numpy as np
from scipy.ndimage import binary_erosion


def detail_score(image, foreground=None):
    """Mean RGB high-pass energy inside the foreground, excluding silhouette edges."""
    image = np.asarray(image, np.float64)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected an HWC RGB image')
    mask = np.ones(image.shape[:2], bool) if foreground is None else np.asarray(foreground, bool)
    mask = binary_erosion(mask, iterations=1)[1:-1, 1:-1]
    high_pass = (4*image[1:-1, 1:-1]-image[:-2, 1:-1]-image[2:, 1:-1]
                 -image[1:-1, :-2]-image[1:-1, 2:])
    return float(np.abs(high_pass[mask]).mean()) if mask.any() else 0.


def score_distribution(scores):
    scores = np.asarray(scores, np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Expected finite nonnegative view scores')
    clipped = np.minimum(scores, np.quantile(scores, .95))
    return clipped/clipped.sum() if clipped.sum() > 0 else np.full(len(scores), 1/len(scores))


class ViewPriorities:
    """A uniform probability floor plus loss/detail importance; loss follows an EMA."""
    def __init__(self, losses, details, uniform_fraction=.5, ema=.9):
        if not 0 <= uniform_fraction <= 1 or not 0 <= ema < 1:
            raise ValueError('Invalid view sampling settings')
        self.losses = np.asarray(losses, np.float64).copy()
        self.initial_losses = self.losses.copy()
        self.details = np.asarray(details, np.float64).copy()
        if self.losses.shape != self.details.shape:
            raise ValueError('One loss and detail score per view required')
        score_distribution(self.losses)
        score_distribution(self.details)
        self.uniform_fraction, self.ema = uniform_fraction, ema
        self.counts = np.zeros(len(self.losses), np.int64)

    def probabilities(self):
        n = len(self.losses)
        loss, detail = score_distribution(self.losses), score_distribution(self.details)
        importance = (loss+detail)/2
        return self.uniform_fraction/n+(1-self.uniform_fraction)*importance

    def update(self, index, loss):
        if not np.isfinite(loss) or loss < -1e-5:
            raise ValueError('Invalid render loss')
        self.losses[index] = self.ema*self.losses[index]+(1-self.ema)*max(0., float(loss))
        self.counts[index] += 1


def training_order(seed, views, steps, reference_steps=0):
    """Match the legacy order; fix minibatch RNG advancement across step ablations.

    reference_steps=0 preserves the original behaviour. Using 500 in all
    experiments gives identical camera and Mesh-minibatch prefixes for 500,
    1500 and 3000 updates with the same views and sampling policy.
    """
    order_rng = np.random.default_rng(seed)
    order = np.concatenate([order_rng.permutation(views) for _ in range(steps//views+1)])[:steps]
    batch_rng = np.random.default_rng(seed)
    for _ in range((reference_steps or steps)//views+1):
        batch_rng.permutation(views)
    return order, batch_rng
