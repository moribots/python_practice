from . import reinforcement_learning_practice as rlp
import numpy as np
from common.test_utils import _pass, _fail


def test_rl():
    print("\nTesting RL Problems:")
    counter = 1
    # Problem 1
    Q = np.zeros((2, 2))
    try:
        rlp.problem1(Q, 0, 0, 1, 1, 0.1, 0.9)
        assert Q[0, 0] > 0
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 2
    policy = {0: 0, 1: 1}
    transitions = {(0, 0): [(0.9, 0), (0.1, 1)], (0, 1): [(0.5, 0), (0.5, 1)]}
    rewards = {(0, 0, 0): 0, (0, 0, 1): 0}
    try:
        V = rlp.problem2(policy, transitions, rewards, 0.9)
        assert len(V) == 2
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 3
    Q = np.array([[1, 0], [0, 2]])
    try:
        action = rlp.problem3(Q, 0, 0.1)
        assert action in [0, 1]
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 5
    Q = np.zeros((2, 2))
    try:
        rlp.problem5(Q, 0, 0, 1, 1, 0, 0.1, 0.9)
        assert Q[0, 0] > 0
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 6
    try:
        dqn = rlp.problem6()
        assert hasattr(dqn, 'forward')
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 7
    try:
        replay_buffer = rlp.problem7()
        assert hasattr(replay_buffer, 'store') or hasattr(
            replay_buffer, 'push')
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 8
    try:
        policy = rlp.problem8()
        assert callable(policy)
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 9
    try:
        actor_critic = rlp.problem9()
        assert hasattr(actor_critic, 'actor') or hasattr(
            actor_critic, 'critic')
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 10
    try:
        ucb_action = rlp.problem10()
        assert isinstance(ucb_action, int)
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 11
    try:
        td_lambda_values = rlp.problem11()
        assert isinstance(td_lambda_values, dict)
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 12
    try:
        linear_fa = rlp.problem12()
        assert hasattr(linear_fa, 'predict') or callable(linear_fa)
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 13
    Q = np.array([[1, 0], [0, 2]])
    try:
        action = rlp.problem13(Q, 0, 1.0)
        assert action in [0, 1]
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 14
    try:
        value_function = rlp.problem14()
        assert isinstance(value_function, dict)
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1

    # Problem 15
    Q1 = np.zeros((2, 2))
    Q2 = np.zeros((2, 2))
    try:
        rlp.problem15(Q1, Q2, 0, 0, 1, 1, 0.1, 0.9)
        assert Q1[0, 0] > 0 or Q2[0, 0] > 0
        _pass(counter, "RL")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "RL", e)
        counter += 1
