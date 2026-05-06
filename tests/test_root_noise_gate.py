import equinox as eqx
import jax
import jax.numpy as jnp

from connectzero import batched, single


class FixedPolicyModel(eqx.Module):
    logits: jnp.ndarray

    def __call__(self, _x, state):
        return (self.logits, jnp.array([0.0], dtype=jnp.float32)), state


def test_single_puct_root_noise_is_opt_in():
    model = FixedPolicyModel(
        jnp.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    )
    board_state = jnp.zeros((6, 7), dtype=jnp.int32)

    tree = single.SearchTree.init(N=50, A=7)
    _, action, _, sample = single.run_mcts_search(
        tree,
        board_state,
        num_simulations=1,
        c_term=jnp.sqrt(2),
        key=jax.random.PRNGKey(0),
        model=(model, None),
        temperature=1e-8,
        temperature_depth=0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=1.0,
        add_root_noise=False,
    )

    assert action == 6
    assert jnp.array_equal(sample.policy_target, jnp.array([0, 0, 0, 0, 0, 0, 1]))

    tree = single.SearchTree.init(N=50, A=7)
    _, action, _, sample = single.run_mcts_search(
        tree,
        board_state,
        num_simulations=1,
        c_term=jnp.sqrt(2),
        key=jax.random.PRNGKey(0),
        model=(model, None),
        temperature=1e-8,
        temperature_depth=0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=1.0,
        add_root_noise=True,
    )

    assert action == 4
    assert jnp.array_equal(sample.policy_target, jnp.array([0, 0, 0, 0, 1, 0, 0]))


def test_batched_puct_root_noise_is_opt_in():
    model = FixedPolicyModel(
        jnp.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    )
    board_state = jnp.zeros((2, 6, 7), dtype=jnp.int32)

    tree = batched.BatchedSearchTree.init(B=2, N=50, A=7)
    _, action, _, sample = batched.run_mcts_search(
        tree,
        board_state,
        num_simulations=1,
        c_term=jnp.sqrt(2),
        key=jax.random.PRNGKey(0),
        model=(model, None),
        temperature=1e-8,
        temperature_depth=0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=1.0,
        add_root_noise=False,
    )

    assert jnp.array_equal(action, jnp.array([6, 6]))
    assert jnp.array_equal(
        sample.policy_target,
        jnp.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]),
    )

    tree = batched.BatchedSearchTree.init(B=2, N=50, A=7)
    _, action, _, sample = batched.run_mcts_search(
        tree,
        board_state,
        num_simulations=1,
        c_term=jnp.sqrt(2),
        key=jax.random.PRNGKey(0),
        model=(model, None),
        temperature=1e-8,
        temperature_depth=0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=1.0,
        add_root_noise=True,
    )

    assert jnp.array_equal(action, jnp.array([4, 5]))
    assert jnp.array_equal(
        sample.policy_target,
        jnp.array([[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0]]),
    )
