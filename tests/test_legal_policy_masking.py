import jax
import jax.numpy as jnp

from connectzero import batched, single
from connectzero.game import (
    masked_policy_softmax,
    normalize_legal_policy,
    visit_count_logits,
)


def test_masked_policy_softmax_renormalizes_over_legal_moves_only():
    logits = jnp.array([[10.0, 1.0, 1.0, 1.0, 1.0, 1.0, -5.0]])
    legal_moves = jnp.array([[False, True, True, True, True, True, False]])

    policy = masked_policy_softmax(logits, legal_moves, axis=1)

    assert policy[0, 0] == 0.0
    assert policy[0, 6] == 0.0
    assert jnp.isclose(jnp.sum(policy), 1.0)
    assert jnp.allclose(policy[0, 1:6], jnp.full((5,), 0.2))


def test_normalize_legal_policy_removes_illegal_dirichlet_mass():
    noise = jnp.array([[0.7, 0.1, 0.1, 0.05, 0.025, 0.025, 0.0]])
    legal_moves = jnp.array([[False, True, True, True, False, False, False]])

    policy = normalize_legal_policy(noise, legal_moves, axis=1)

    assert jnp.all(policy[0, jnp.array([0, 4, 5, 6])] == 0.0)
    assert jnp.isclose(jnp.sum(policy), 1.0)
    assert jnp.allclose(policy[0, 1:4], jnp.array([0.4, 0.4, 0.2]))


def test_single_dirichlet_noise_keeps_zero_mass_on_full_columns():
    key = jax.random.PRNGKey(0)
    tree = single.SearchTree.init(N=10, A=7)
    tree = tree._replace(
        children_priors=tree.children_priors.at[tree.root_index].set(
            jnp.ones((7,), dtype=jnp.float32) / 7
        )
    )
    board_state = jnp.zeros((6, 7), dtype=jnp.int32)
    board_state = board_state.at[:, 0].set(1)
    board_state = board_state.at[:, 6].set(2)

    tree = single.add_dirichlet_noise(tree, key, board_state, alpha=0.3, epsilon=0.25)
    priors = tree.children_priors[tree.root_index]

    assert priors[0] == 0.0
    assert priors[6] == 0.0
    assert jnp.isclose(jnp.sum(priors), 1.0)


def test_batched_dirichlet_noise_keeps_zero_mass_on_full_columns_per_board():
    key = jax.random.PRNGKey(0)
    tree = batched.BatchedSearchTree.init(B=2, N=10, A=7)
    tree = tree._replace(
        children_priors=tree.children_priors.at[:, 0, :].set(
            jnp.ones((2, 7), dtype=jnp.float32) / 7
        )
    )
    board_state = jnp.zeros((2, 6, 7), dtype=jnp.int32)
    board_state = board_state.at[0, :, 0].set(1)
    board_state = board_state.at[1, :, 6].set(2)

    tree = batched.add_dirichlet_noise(tree, key, board_state, alpha=0.3, epsilon=0.25)
    priors = tree.children_priors[jnp.arange(2), tree.root_index]

    assert priors[0, 0] == 0.0
    assert priors[1, 6] == 0.0
    assert jnp.allclose(jnp.sum(priors, axis=1), jnp.ones((2,)))


def test_visit_count_logits_give_zero_visit_actions_zero_probability():
    visits = jnp.array([10, 0, 5, 0, 0, 0, 0], dtype=jnp.int32)
    legal_moves = jnp.array([True, True, True, False, True, False, False])

    logits = visit_count_logits(visits, legal_moves, temperature=1.0)
    policy = jax.nn.softmax(logits)

    assert policy[1] == 0.0
    assert policy[3] == 0.0
    assert policy[4] == 0.0
    assert policy[5] == 0.0
    assert policy[6] == 0.0
    assert jnp.allclose(policy[jnp.array([0, 2])], jnp.array([2 / 3, 1 / 3]))


def test_visit_count_logits_fall_back_to_uniform_legal_policy_when_all_visits_zero():
    visits = jnp.zeros((2, 7), dtype=jnp.int32)
    legal_moves = jnp.array(
        [
            [False, True, True, False, False, False, False],
            [False, False, False, True, True, True, False],
        ]
    )

    logits = visit_count_logits(visits, legal_moves, temperature=1.0, axis=1)
    policy = jax.nn.softmax(logits, axis=1)

    assert jnp.allclose(policy[0], jnp.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))
    assert jnp.allclose(
        policy[1],
        jnp.array([0.0, 0.0, 0.0, 1 / 3, 1 / 3, 1 / 3, 0.0]),
    )
