from . import robotics_fundamentals_practice as rfp
import numpy as np
from common.test_utils import _pass, _fail, debug_print


def test_robotics():
    print("\nTesting Robotics Problems:")
    counter = 1
    # Problem 1
    debug_print(f"Testing Problem {counter}")
    try:
        result = rfp.problem1(0, 0, 1, 1)
        assert np.allclose(result, (2, 0))
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        debug_print(f"Problem {counter} not implemented")
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 2
    try:
        pid = rfp.Problem2(1, 0, 0)
        signal = pid.update(1)
        assert isinstance(signal, (int, float))
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 3
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    try:
        path = rfp.problem3(grid, (0, 0), (2, 2))
        assert isinstance(path, list)
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 5
    try:
        result = rfp.problem5(0.5, 0.5, 1, 1)
        assert len(result) == 2  # joint angles
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 6
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.array([[1]])
    try:
        K = rfp.problem6(A, B, Q, R)
        assert K.shape == (1, 2)
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 7
    try:
        ekf = rfp.problem7()
        assert hasattr(ekf, 'predict') or hasattr(ekf, 'update')
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 8
    try:
        slam = rfp.problem8()
        assert hasattr(slam, 'update') or callable(slam)
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 9
    try:
        rrt = rfp.problem9()
        assert hasattr(rrt, 'plan') or callable(rrt)
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 10
    image = np.random.rand(10, 10)
    try:
        corners = rfp.problem10(image)
        assert isinstance(corners, np.ndarray)
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 30: Pose Error
    try:
        current_pose = [0.1, 0.0, 0.5, 0, 0, 0, 1]
        desired_pose = [0.0, 0.0, 0.5, 0, 0, 0, 1]
        pos_err, orient_err = rfp.problem30(current_pose, desired_pose)
        assert len(pos_err) == 3 and len(orient_err) == 3
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 31: Twist Error
    try:
        current_twist = [0.1, 0.0, 0.0, 0.0, 0.0, 0.1]
        desired_twist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        twist_error = rfp.problem31(current_twist, desired_twist)
        assert len(twist_error) == 6
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 32: Reward Function
    try:
        current_pose = [0.1, 0.0, 0.5, 0, 0, 0, 1]
        desired_pose = [0.0, 0.0, 0.5, 0, 0, 0, 1]
        current_twist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        action = [0.0, 0.0]
        reward = rfp.problem32(current_pose, desired_pose,
                               current_twist, action, 0.01)
        assert isinstance(reward, (int, float))
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 33: State Features
    try:
        joint_pos = [0.5, 0.0, 1.0]
        joint_vel = [0.0, 0.0, 0.0]
        ee_pose = [0.8, 0.0, 0.5, 0, 0, 0, 1]
        obstacles = [[1.0, 0.0, 0.5]]
        features = rfp.problem33(joint_pos, joint_vel, ee_pose, obstacles)
        assert isinstance(features, (list, np.ndarray))
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 35: Episode Termination
    try:
        current_pose = [0.02, 0.0, 0.5, 0, 0, 0, 1]
        desired_pose = [0.0, 0.0, 0.5, 0, 0, 0, 1]
        joint_pos = [0.5, 0.0]
        joint_limits = [[0.0, -np.pi/2], [np.pi, np.pi/2]]
        done, success, info = rfp.problem35(
            current_pose, desired_pose, joint_pos, joint_limits, 5.0, 10.0)
        assert isinstance(done, bool) and isinstance(success, bool)
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 40: Safety Monitoring
    try:
        joint_pos = [0.1, 0.0, 0.0]
        joint_vel = [0.5, 0.0, 0.0]
        ee_pose = [0.8, 0.0, 0.5, 0, 0, 0, 1]
        obstacles = [[1.0, 0.0, 0.5]]
        safety_margins = {'joint_margin': 0.1,
                          'velocity_factor': 0.9, 'collision_margin': 0.2}
        penalty, violations = rfp.problem40(
            joint_pos, joint_vel, ee_pose, obstacles, safety_margins)
        assert isinstance(penalty, (int, float)
                          ) and isinstance(violations, list)
        _pass(counter, "Robotics")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Robotics", e)
        counter += 1

    # Problem 101 (A*)
    try:
        n = 101
        g = np.zeros((5, 5), dtype=int)
        g[2, 1:4] = 1
        path = rfp.rb_problem101_grid_astar(g, (0, 0), (4, 4))
        assert isinstance(path, list)
        _pass(n, "Robotics")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(101, "Robotics", e)

    # Problem 102 (RRT-Connect)
    try:
        n = 102
        def sampler(): return np.random.randn(2)
        def steer(a, b): return (
            a + (b-a)/max(1.0, float(np.linalg.norm(b-a))))

        def collision_free(a, b): return True
        wp = rfp.rb_problem102_rrt_connect(
            np.zeros(2), np.ones(2), sampler, steer, collision_free)
        assert isinstance(wp, list)
        _pass(n, "Robotics")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(102, "Robotics", e)

    # Problem 103 (PD tune)
    try:
        n = 103
        kp, kd = rfp.rb_problem103_pd_tune(1.0, 0.1, 0.0, zeta=0.7, wn=5.0)
        assert isinstance(kp, float) and isinstance(kd, float)
        _pass(n, "Robotics")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(103, "Robotics", e)

    # Problem 104 (EKF)
    try:
        n = 104
        x = np.zeros(2)
        P = np.eye(2)
        u = np.zeros(1)
        z = np.zeros(1)
        x2, P2 = rfp.rb_problem104_ekf_step(x, P, u, z)
        assert x2.shape == (2,) and P2.shape == (2, 2)
        _pass(n, "Robotics")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(104, "Robotics", e)

    # Problem 105 (SE3 pose from points)
    try:
        n = 105
        Pw = np.random.randn(3, 5)
        R = np.eye(3)
        t = np.array([0.1, -0.2, 0.3])
        Pc = R @ Pw + t[:, None]
        Rout, tout = rfp.rb_problem105_se3_pose_from_points(Pw, Pc)
        assert Rout.shape == (3, 3) and tout.shape == (3,)
        _pass(n, "Robotics")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(105, "Robotics", e)
