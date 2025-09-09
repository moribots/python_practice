import einops
import numpy as np
from . import einops_practice as ep
from common.test_utils import _pass, _fail


def test_einops():
    print("Testing Einops Problems:")
    counter = 1
    # Problem 1
    tensor = np.random.rand(2, 10, 10, 3)
    expected = einops.rearrange(tensor, 'b h w c -> b c h w')
    try:
        result = ep.problem1(tensor)
        assert np.allclose(result, expected)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 2
    tensor = np.random.rand(2, 5, 10)
    expected = einops.rearrange(tensor, 'b s h -> b (s h)')
    try:
        result = ep.problem2(tensor)
        assert np.allclose(result, expected)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 3
    tensor = np.random.rand(2, 20)
    expected = einops.rearrange(tensor, 'b (h d) -> b h d', h=2)
    try:
        result = ep.problem3(tensor)
        assert np.allclose(result, expected)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 4
    tensor = np.random.rand(2, 5, 8, 10)
    expected = einops.rearrange(tensor, 'b s h d -> b h s d')
    try:
        result = ep.problem4(tensor)
        assert np.allclose(result, expected)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 6
    tensor = np.random.rand(2, 10, 10, 3)
    expected = einops.reduce(tensor, 'b h w c -> b c', 'mean')
    try:
        result = ep.problem6(tensor)
        assert np.allclose(result, expected)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 7
    tensor = np.random.rand(2, 10, 10, 3)
    expected = einops.reduce(tensor, 'b h w c -> b c', 'max')
    try:
        result = ep.problem7(tensor)
        assert np.allclose(result, expected)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 8
    tensor = np.random.rand(2, 5, 10, 10, 3)
    expected = einops.rearrange(tensor, 'b t h w c -> (b t) c h w')
    try:
        result = ep.problem8(tensor)
        assert np.allclose(result, expected)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 9
    tensor = np.random.rand(2, 10, 30)
    expected_q = einops.rearrange(tensor, 'b s (h d) -> b h s d', h=3)
    expected_k = expected_q
    expected_v = expected_q
    try:
        Q, K, V = ep.problem9(tensor)
        assert np.allclose(Q, expected_q) and np.allclose(
            K, expected_k) and np.allclose(V, expected_v)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 10
    tensor = np.random.rand(4, 8, 16)
    try:
        result = ep.problem10(tensor)
        # This is more of a demonstration, so just check it runs
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 11
    A = np.random.randn(7, 5)
    B = np.random.randn(5, 3)
    try:
        out = ep.problem11_mm_einsum(A, B)
        assert np.allclose(out, A @ B)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 12
    Ab = np.random.randn(4, 7, 5)
    Bb = np.random.randn(4, 5, 3)
    try:
        out = ep.problem12_bmm_einsum(Ab, Bb)
        ref = np.stack([Ab[i] @ Bb[i] for i in range(4)], axis=0)
        assert np.allclose(out, ref)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 13 + 14
    Bn, H, Tq, Tk, D = 2, 3, 4, 5, 6
    Q = np.random.randn(Bn, H, Tq, D)
    K = np.random.randn(Bn, H, Tk, D)
    V = np.random.randn(Bn, H, Tk, D)
    try:
        scores = ep.problem13_attention_scores(Q, K, scale=1.0/np.sqrt(D))
        ref = np.einsum('bhqd,bhkd->bhqk', Q, K) * (1.0/np.sqrt(D))
        assert np.allclose(scores, ref)
        ctx = ep.problem14_attention_weighted_sum(scores, V)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        ref_ctx = np.einsum('bhqk,bhkd->bhqd', attn, V)
        assert np.allclose(ctx, ref_ctx)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 15
    try:
        X = np.random.randn(2, 4, 7)
        Y = np.random.randn(2, 3, 7)
        D2 = ep.problem15_pairwise_sq_dists(X, Y)
        brute = ((X[:, :, None, :] - Y[:, None, :, :])**2).sum(-1)
        assert np.allclose(D2, brute)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 16
    try:
        x = np.random.randn(2, 3, 15)
        w = np.random.randn(5, 3, 4)
        y = ep.problem16_conv1d_valid(x, w)
        assert y.shape == (2, 5, 12)
        _pass(counter, "Einops")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Einops", e)
        counter += 1

    # Problem 17
    try:
        n = 17
        B, H, T, D = 2, 2, 4, 8
        Q = np.random.randn(B, H, T, D)
        K = np.random.randn(B, H, T, D)
        S = ep.problem17_causal_attention_scores(Q, K, scale=1.0/np.sqrt(D))
        assert S.shape == (B, H, T, T)
        _pass(n, "Einops")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(17, "Einops", e)

    # Problem 18
    try:
        n = 18
        B, H, T, D = 2, 2, 4, 8
        S = np.random.randn(B, H, T, T)
        V = np.random.randn(B, H, T, D)
        M = np.tril(np.ones((T, T), dtype=bool))[None, None, :, :]
        C = ep.problem18_attention_context_with_mask(S, V, M)
        assert C.shape == (B, H, T, D)
        _pass(n, "Einops")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(18, "Einops", e)

    # Problem 19
    try:
        n = 19
        x = np.random.randn(1, 1, 3, 6)
        cos = np.ones((3, 3))
        sin = np.zeros((3, 3))
        y = ep.problem19_rotary_apply(x, cos, sin)
        assert y.shape == x.shape
        _pass(n, "Einops")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(19, "Einops", e)

    # Problem 20
    try:
        n = 20
        x = np.random.randn(4, 5, 6)
        y = ep.problem20_layernorm_einsum(x)
        assert y.shape == x.shape
        _pass(n, "Einops")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(20, "Einops", e)

    # Problem 21
    try:
        n = 21
        B, D = 3, 5
        x = np.random.randn(B, D)
        y = np.random.randn(B, D)
        W = np.random.randn(D, D)
        out = ep.problem21_bilinear_einsum(x, W, y)
        assert out.shape == (B,)
        _pass(n, "Einops")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(21, "Einops", e)

    # Problem 22
    try:
        n = 22
        x = np.random.randn(2, 3, 12, 10)
        w = np.random.randn(5, 3, 3, 4)
        y = ep.problem22_conv2d_valid_einsum(x, w)
        assert y.shape == (2, 5, 10, 7)
        _pass(n, "Einops")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(22, "Einops", e)
