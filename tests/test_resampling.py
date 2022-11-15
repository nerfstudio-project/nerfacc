import pytest
import torch
from functorch import vmap

from nerfacc import pack_info, ray_marching, ray_resampling
from nerfacc.cuda import ray_pdf_query

device = "cuda:0"
batch_size = 128
eps = torch.finfo(torch.float32).eps


def _interp(x, xp, fp):
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    xp = xp.contiguous()
    x = x.contiguous()
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    indices = torch.searchsorted(xp, x, right=True) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)
    return m[indices] * x + b[indices]


def _integrate_weights(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.

    Args:
      w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.

    Returns:
      cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = torch.clamp(torch.cumsum(w[..., :-1], dim=-1), max=1)
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    zeros = torch.zeros(shape, device=w.device)
    ones = torch.ones(shape, device=w.device)
    cw0 = torch.cat([zeros, cw, ones], dim=-1)
    return cw0


def _invert_cdf(u, t, w_logits):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = torch.softmax(w_logits, dim=-1)
    # w = torch.exp(w_logits)
    # w = w / torch.sum(w, dim=-1, keepdim=True)
    cw = _integrate_weights(w)
    # Interpolate into the inverse CDF.
    t_new = vmap(_interp)(u, cw, t)
    return t_new


def _resampling(t, w_logits, num_samples):
    """Piecewise-Constant PDF sampling from a step function.

    Args:
        t: [..., num_bins + 1], bin endpoint coordinates (must be sorted).
        w_logits: [..., num_bins], logits corresponding to bin weights.
        num_samples: int, the number of samples.

    returns:
        t_samples: [..., num_samples], the sampled t values
    """
    pad = 1 / (2 * num_samples)
    u = torch.linspace(pad, 1.0 - pad - eps, num_samples, device=device)
    u = torch.broadcast_to(u, t.shape[:-1] + (num_samples,))
    return _invert_cdf(u, t, w_logits)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_resampling():
    batch_size = 1024
    num_bins = 128
    num_samples = 128

    t = torch.randn((batch_size, num_bins + 1), device=device)
    t = torch.sort(t, dim=-1).values
    w_logits = torch.randn((batch_size, num_bins), device=device) * 0.1
    w = torch.softmax(w_logits, dim=-1)
    masks = w_logits > 0
    w_logits[~masks] = -torch.inf

    t_samples = _resampling(t, w_logits, num_samples + 1)

    t_starts = t[:, :-1][masks].unsqueeze(-1)
    t_ends = t[:, 1:][masks].unsqueeze(-1)
    w_logits = w_logits[masks].unsqueeze(-1)
    w = w[masks].unsqueeze(-1)
    num_steps = masks.long().sum(dim=-1)
    cum_steps = torch.cumsum(num_steps, dim=0)
    packed_info = torch.stack([cum_steps - num_steps, num_steps], dim=-1).int()

    _, t_starts, t_ends = ray_resampling(
        packed_info, t_starts, t_ends, w, num_samples
    )

    # print(
    #     (t_starts.view(batch_size, num_samples) - t_samples[:, :-1])
    #     .abs()
    #     .max(),
    #     (t_ends.view(batch_size, num_samples) - t_samples[:, 1:]).abs().max(),
    # )
    assert torch.allclose(
        t_starts.view(batch_size, num_samples), t_samples[:, :-1], atol=1e-3
    )
    assert torch.allclose(
        t_ends.view(batch_size, num_samples), t_samples[:, 1:], atol=1e-3
    )


def test_pdf_query():
    rays_o = torch.rand((1, 3), device=device)
    rays_d = torch.randn((1, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    ray_indices, t_starts, t_ends = ray_marching(
        rays_o,
        rays_d,
        near_plane=0.1,
        far_plane=1.0,
        render_step_size=0.2,
    )
    packed_info = pack_info(ray_indices, rays_o.shape[0])

    weights = torch.rand((t_starts.shape[0], 1), device=device)
    weights_new = ray_pdf_query(
        packed_info,
        t_starts,
        t_ends,
        weights,
        packed_info,
        t_starts + 0.3,
        t_ends + 0.3,
    )


if __name__ == "__main__":
    test_resampling()
    test_pdf_query()
