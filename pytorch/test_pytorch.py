from . import pytorch_practice as ptp
import torch
from common.test_utils import _pass, _fail


def test_pytorch():
    print("\nTesting PyTorch Problems:")
    counter = 1
    # Problem 1
    try:
        model = ptp.Problem1(10, 5, 1)
        x = torch.randn(1, 10)
        out = model(x)
        assert out.shape == (1, 1)
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 2
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.5, 2.5])
    try:
        loss = ptp.problem2(pred, target)
        expected = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(loss, expected)
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 3
    input_tensor = torch.randn(1, 1, 4, 4)
    weight = torch.randn(1, 1, 2, 2)
    bias = torch.zeros(1)
    try:
        result = ptp.problem3(input_tensor, weight, bias)
        expected = torch.nn.functional.conv2d(input_tensor, weight, bias)
        assert torch.allclose(result, expected)
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 4
    try:
        rnn = ptp.Problem4(10, 5)
        x = torch.randn(1, 10)
        h = torch.randn(1, 5)
        out, new_h = rnn(x, h)
        assert out.shape == (1, 5)
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 6
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    try:
        model = ptp.Problem1(10, 5, 1)
        ptp.problem6(model, X_train, y_train, epochs=1, batch_size=10)
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 7
    try:
        loss_fn = ptp.Problem7()
        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        loss = loss_fn(pred, target)
        assert isinstance(loss.item(), float)
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 8
    try:
        model = ptp.Problem1(10, 5, 1)
        ptp.problem8_save_model(model, 'test_model.pth')
        loaded_model = ptp.problem8_load_model(
            ptp.Problem1, 'test_model.pth', 10, 5, 1)
        assert isinstance(loaded_model, ptp.Problem1)
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1
    finally:
        # Clean up
        import os
        if os.path.exists('test_model.pth'):
            os.remove('test_model.pth')

    # Problem 9
    try:
        ptp.problem9_gpu_operations()
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 10
    loss = torch.tensor(1.0, requires_grad=True)
    try:
        model = ptp.Problem1(10, 5, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ptp.problem10_gradient_clipping(model, optimizer, loss)
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 11
    try:
        bn_layer = ptp.Problem11(10)
        x = torch.randn(5, 10)
        out = bn_layer(x)
        assert out.shape == x.shape
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 12
    try:
        dropout_layer = ptp.Problem12()
        x = torch.randn(5, 10)
        out = dropout_layer(x)
        assert out.shape == x.shape
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 13
    optimizer = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.01)
    try:
        scheduler = ptp.problem13_lr_scheduler(optimizer, 'step')
        assert scheduler is not None
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 14
    try:
        attention = ptp.Problem14_MultiHeadAttention(embed_size=64, heads=8)
        query = torch.randn(10, 32, 64)
        key = torch.randn(10, 32, 64)
        value = torch.randn(10, 32, 64)
        out = attention(query, key, value)
        assert out.shape == query.shape
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 15
    try:
        model = ptp.problem15_transfer_learning()
        assert model is not None
        _pass(counter, "PyTorch")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "PyTorch", e)
        counter += 1

    # Problem 7 (GAE)
    try:
        n = 7
        T = 8
        out = ptp.ml_problem7_gae(torch.zeros(T), torch.zeros(
            T+1), torch.zeros(T), torch.zeros(T), 0.99, 0.95)
        assert isinstance(out, torch.Tensor)
        _pass(n, "PyTorch")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(7, "PyTorch", e)

    # Problem 8 (PPO clip)
    try:
        n = 8
        out = ptp.ml_problem8_ppo_clip_objective(
            torch.randn(16), torch.randn(16), torch.randn(16), 0.2)
        assert out.shape == ()
        _pass(n, "PyTorch")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(8, "PyTorch", e)

    # Problem 9 (TD3 target)
    try:
        n = 9

        class Q(torch.nn.Module):
            def forward(self, s, a): return torch.randn(s.shape[0], 1)

        class A(torch.nn.Module):
            def forward(self, s): return torch.randn(s.shape[0], 2)
        q1t, q2t, at = Q(), Q(), A()
        tgt = ptp.ml_problem9_td3_target(q1t, q2t, at, torch.randn(
            10, 4), torch.randn(10, 1), torch.zeros(10, 1), 0.99, 0.2)
        assert tgt.shape == (10, 1)
        _pass(n, "PyTorch")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(9, "PyTorch", e)

    # Problem 10 (prioritized replay sample)
    try:
        n = 10
        idx = ptp.ml_problem10_prioritized_replay_sample(
            torch.rand(100), batch_size=16, alpha=0.6)
        assert isinstance(idx, torch.Tensor)
        _pass(n, "PyTorch")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(10, "PyTorch", e)

    # Problem 11 (EMA/Polyak)
    try:
        n = 11
        m1 = torch.nn.Linear(3, 3)
        m2 = torch.nn.Linear(3, 3)
        ptp.ml_problem11_ema_update(m1, m2, tau=0.005)
        _pass(n, "PyTorch")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(11, "PyTorch", e)

    # Problem 12 (advantage normalize)
    try:
        n = 12
        out = ptp.ml_problem12_advantage_normalize(torch.randn(32))
        assert out.shape == (32,)
        _pass(n, "PyTorch")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(12, "PyTorch", e)
