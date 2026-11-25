import dataclasses
from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi.models import cfg as _cfg
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.shared.nnx_utils import module_jit

# Fixed routing example used across tests (see MD §3): labels [T, T, F, F] with
# uniforms [0.5, 0.05, 0.5, 0.05] and p = 0.1.
_LABELS = [True, True, False, False]
_RANDOM_VALUES = [0.5, 0.05, 0.5, 0.05]
_P = 0.1


def _routing(*, positive_only: bool, labels=_LABELS, random_values=_RANDOM_VALUES, p=_P):
    return _cfg.compute_routing_masks(
        jnp.asarray(labels),
        jnp.asarray(random_values),
        positive_only_conditional=positive_only,
        unconditional_prob=p,
    )


def test_positive_only_routing_table():
    routing = _routing(positive_only=True)

    # MD §3 table, cell by cell: only positive samples with r > p route to the
    # conditional branch; negatives always stay on the original prompt.
    np.testing.assert_array_equal(routing["positive_mask"], [True, True, False, False])
    np.testing.assert_array_equal(routing["negative_mask"], [False, False, True, True])
    np.testing.assert_array_equal(routing["positive_conditional_mask"], [True, False, False, False])
    np.testing.assert_array_equal(routing["positive_unconditional_mask"], [False, True, False, False])
    np.testing.assert_array_equal(routing["negative_conditional_mask"], [False, False, False, False])
    np.testing.assert_array_equal(routing["negative_unconditional_mask"], [False, False, True, True])
    np.testing.assert_array_equal(routing["conditional_mask"], [True, False, False, False])


def test_full_mode_routing_guidance_kept():
    # Full mode: both labels route when r > p.
    routing = _routing(positive_only=False, labels=[True, False], random_values=[0.5, 0.5])

    np.testing.assert_array_equal(routing["positive_conditional_mask"], [True, False])
    np.testing.assert_array_equal(routing["negative_conditional_mask"], [False, True])
    np.testing.assert_array_equal(routing["conditional_mask"], [True, True])
    np.testing.assert_array_equal(routing["positive_unconditional_mask"], [False, False])
    np.testing.assert_array_equal(routing["negative_unconditional_mask"], [False, False])


def test_full_mode_routing_guidance_dropped():
    # Full mode with r <= p: everything falls back to the unconditional branch.
    routing = _routing(positive_only=False, labels=[True, False], random_values=[0.05, 0.05])

    np.testing.assert_array_equal(routing["positive_conditional_mask"], [False, False])
    np.testing.assert_array_equal(routing["negative_conditional_mask"], [False, False])
    np.testing.assert_array_equal(routing["conditional_mask"], [False, False])
    np.testing.assert_array_equal(routing["positive_unconditional_mask"], [True, False])
    np.testing.assert_array_equal(routing["negative_unconditional_mask"], [False, True])


def _synthetic_observation(batch_size: int = 4, seq_len: int = 5) -> _model.Observation:
    return _model.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((batch_size, 2), jnp.float32),
        tokenized_prompt=jnp.full((batch_size, seq_len), 11, jnp.int32),
        tokenized_prompt_mask=jnp.zeros((batch_size, seq_len), bool),
        tokenized_positive_guidance_prompt=jnp.full((batch_size, seq_len), 22, jnp.int32),
        tokenized_positive_guidance_prompt_mask=jnp.ones((batch_size, seq_len), bool),
        tokenized_negative_guidance_prompt=jnp.full((batch_size, seq_len), 33, jnp.int32),
        tokenized_negative_guidance_prompt_mask=jnp.zeros((batch_size, seq_len), bool),
    )


def test_select_guidance_tokens_positive_only():
    obs = _synthetic_observation()
    routing = _routing(positive_only=True)

    tokens, masks = _cfg.select_guidance_tokens(obs, routing, positive_only_conditional=True)

    # Row 0 is positive-conditional -> positive guidance (22); all others -> original (11).
    np.testing.assert_array_equal(tokens[0], np.full((5,), 22, np.int32))
    np.testing.assert_array_equal(tokens[1:], np.full((3, 5), 11, np.int32))
    # Masks stay in sync with the tokens: row 0 uses the positive mask (ones), the rest zeros.
    np.testing.assert_array_equal(masks[0], np.ones((5,), bool))
    np.testing.assert_array_equal(masks[1:], np.zeros((3, 5), bool))


def test_select_guidance_tokens_full_mode():
    obs = _synthetic_observation()
    # labels [T, T, F, F], r [0.5, 0.05, 0.5, 0.05] with p = 0.1
    # -> guidance_mask [T, F, T, F] -> conditional [T, F, T, F].
    routing = _routing(positive_only=False)

    tokens, masks = _cfg.select_guidance_tokens(obs, routing, positive_only_conditional=False)

    # Guidance is picked by label first, then swapped in where conditional:
    # row 0 pos+cond -> 22, row 1 pos+uncond -> 11, row 2 neg+cond -> 33, row 3 neg+uncond -> 11.
    expected = np.full((4, 5), 11, np.int32)
    expected[0] = 22  # positive, conditional
    expected[2] = 33  # negative, conditional
    np.testing.assert_array_equal(tokens, expected)
    # Masks route the same way: row 0 picks the positive mask (ones); rows 1-3 keep the
    # original mask (zeros), with row 2 taking the negative mask (zeros in this fixture).
    expected_masks = np.zeros((4, 5), bool)
    expected_masks[0] = True
    np.testing.assert_array_equal(masks, expected_masks)


def test_select_guidance_tokens_missing_fields():
    obs = _model.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((2, 2), jnp.float32),
        tokenized_prompt=jnp.zeros((2, 5), jnp.int32),
        tokenized_prompt_mask=jnp.ones((2, 5), bool),
    )
    routing = _routing(positive_only=True, labels=[True, False], random_values=[0.5, 0.5])

    with pytest.raises(ValueError, match="guidance"):
        _cfg.select_guidance_tokens(obs, routing, positive_only_conditional=True)


def test_compute_cfg_metrics_and_ratios():
    # q = 0.3: 30 positives, 27 of them routed conditional (r > 0.1), 70 negatives.
    num_positive, num_conditional = 30, 27
    labels = [True] * num_positive + [False] * (100 - num_positive)
    random_values = [0.5] * num_conditional + [0.05] * (num_positive - num_conditional) + [0.5] * (100 - num_positive)
    routing = _routing(positive_only=True, labels=labels, random_values=random_values)

    # Group-constant losses so derived ratios are easy to check by hand.
    per_sample_loss = jnp.where(
        routing["positive_conditional_mask"], 1.0, jnp.where(routing["positive_unconditional_mask"], 2.0, 3.0)
    )

    metrics = _cfg.compute_cfg_metrics(routing, per_sample_loss)
    assert set(metrics) == set(_cfg.CFG_RAW_KEYS)

    assert float(metrics["conditional_count"]) == 27.0
    assert float(metrics["unconditional_count"]) == 73.0
    assert float(metrics["positive_label_count"]) == 30.0
    assert float(metrics["negative_label_count"]) == 70.0
    assert float(metrics["positive_conditional_count"]) == 27.0
    assert float(metrics["positive_unconditional_count"]) == 3.0
    assert float(metrics["negative_conditional_count"]) == 0.0
    assert float(metrics["negative_unconditional_count"]) == 70.0
    assert float(metrics["conditional_loss_sum"]) == pytest.approx(27.0)
    assert float(metrics["unconditional_loss_sum"]) == pytest.approx(3 * 2.0 + 70 * 3.0)
    assert float(metrics["negative_conditional_loss_sum"]) == 0.0

    derived = _cfg.derive_cfg_ratios({k: float(v) for k, v in metrics.items()})
    assert derived["conditional_ratio"] == pytest.approx(0.27)
    assert derived["unconditional_ratio"] == pytest.approx(0.73)
    assert derived["positive_label_ratio"] == pytest.approx(0.3)
    assert derived["negative_label_ratio"] == pytest.approx(0.7)
    # The headline RECAP checks (MD §5): effective conditional ratio ~ p kept (0.9),
    # negatives never conditional in positive-only mode.
    assert derived["positive_effective_conditional_ratio"] == pytest.approx(0.9)
    assert derived["positive_effective_unconditional_ratio"] == pytest.approx(0.1)
    assert derived["negative_effective_conditional_ratio"] == pytest.approx(0.0)
    assert derived["negative_effective_unconditional_ratio"] == pytest.approx(1.0)
    assert derived["conditional_loss"] == pytest.approx(1.0)
    assert derived["unconditional_loss"] == pytest.approx((3 * 2.0 + 70 * 3.0) / 73.0)
    assert derived["negative_unconditional_loss"] == pytest.approx(3.0)
    # Zero-count groups are skipped rather than producing NaN/inf.
    assert "negative_conditional_loss" not in derived


def test_derive_cfg_ratios_zero_denominators():
    # No positive labels and no conditional samples: those keys are skipped, no division by zero.
    derived = _cfg.derive_cfg_ratios(
        {
            "conditional_count": 0.0,
            "unconditional_count": 4.0,
            "negative_label_count": 4.0,
            "negative_unconditional_count": 4.0,
        }
    )
    assert derived["conditional_ratio"] == pytest.approx(0.0)
    assert derived["negative_effective_unconditional_ratio"] == pytest.approx(1.0)
    assert "positive_effective_conditional_ratio" not in derived
    assert "conditional_loss" not in derived
    assert _cfg.derive_cfg_ratios({}) == {}


def test_draw_routing_random_determinism_and_range():
    rng = jax.random.key(42)
    draw_a = _cfg.draw_routing_random(rng, 64)
    draw_b = _cfg.draw_routing_random(rng, 64)
    np.testing.assert_array_equal(draw_a, draw_b)

    assert bool(jnp.all((draw_a >= 0.0) & (draw_a < 1.0)))

    # fold_in(rng, 3) must not collide with the three split(rng, 3) streams used for
    # preprocess/noise/time — otherwise routing uniforms would correlate with the noise.
    for stream in jax.random.split(rng, 3):
        other = jax.random.uniform(stream, (64,))
        assert not np.allclose(np.asarray(draw_a), np.asarray(other))


def test_non_cfg_loss_bit_identical():
    # Same seed + dummy variants: a CFG model with p = 1.0 and all-negative labels must
    # reproduce the non-CFG loss exactly (routing always selects the original prompt).
    # This is the cheap half of MD §8.1-2's "non-CFG configs unchanged" regression.
    key = jax.random.key(0)
    batch_size = 2
    base_config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    cfg_config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        cfg_guidance=True,
        unconditional_prob=1.0,
    )

    # Use the CFG spec's observation so both models see identical inputs; the base
    # model simply ignores the extra guidance fields.
    observation = cfg_config.fake_obs(batch_size)
    actions = cfg_config.fake_act(batch_size)
    advantage = jnp.zeros((batch_size,), jnp.bool_)

    base_model = base_config.create(key)
    cfg_model = cfg_config.create(key)

    base_loss = base_model.compute_loss(key, observation, actions, train=False)
    cfg_loss = cfg_model.compute_loss(key, observation, actions, train=False, advantage=advantage)

    assert base_loss.shape == (batch_size, base_config.action_horizon)
    np.testing.assert_allclose(base_loss, cfg_loss, rtol=1e-7, atol=1e-7)


def test_cfg_model_requires_advantage():
    key = jax.random.key(0)
    batch_size = 2
    config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", cfg_guidance=True)
    model = config.create(key)
    observation = config.fake_obs(batch_size)
    actions = config.fake_act(batch_size)

    with pytest.raises(ValueError, match="advantage"):
        model.compute_loss(key, observation, actions, train=False)


# Inference regression tests use the actual samplers with lightweight nonlinear backbones.
class _JaxBackbone(nnx.Module):
    def __call__(self, embs, *, mask, positions, kv_cache=None, adarms_cond=None):
        prefix, suffix = embs
        if suffix is None:
            return (prefix, None), prefix[:, :1, :]
        # A nonlinear velocity makes mixing completed action chunks detectably wrong.
        velocity = kv_cache * (1 + 0.1 * suffix**2) + adarms_cond[1][:, None, None]
        velocity += positions[:, :1, None] * 0.01
        return (None, velocity), None


class _JaxSampler(Pi0):
    def __init__(self, config):
        self.cfg_guidance = config.cfg_guidance
        self.positive_only_conditional = config.positive_only_conditional
        self.guidance_type = config.guidance_type
        self.cfgrl_guidance_scale = config.cfgrl_guidance_scale
        self.action_dim = 1
        self.action_horizon = 2
        self.PaliGemma = nnx.Dict(llm=_JaxBackbone())
        self.action_out_proj = lambda x: x

    def embed_prefix(self, obs):
        return obs.tokenized_prompt[..., None].astype(jnp.float32), obs.tokenized_prompt_mask, jnp.zeros(2, bool)

    def embed_suffix(self, obs, actions, time):
        return actions, jnp.ones(actions.shape[:2], bool), jnp.array([True, False]), time


class _TorchBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace()))
        self.gemma_expert = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))
        self.prefixes = []
        self.calls = []

    def forward(self, *, inputs_embeds, position_ids, past_key_values, use_cache, adarms_cond=None, **kwargs):
        prefix, suffix = inputs_embeds
        if suffix is None:
            assert use_cache
            assert past_key_values is None
            cache = prefix[:, :1, :].clone()
            self.prefixes.append(cache)
            return (prefix, None), cache
        assert not use_cache
        self.calls.append((past_key_values, suffix.clone(), adarms_cond[1].clone()))
        velocity = past_key_values * (1 + 0.1 * suffix**2) + adarms_cond[1][:, None, None]
        velocity += position_ids[:, :1, None] * 0.01
        return (None, velocity), None


class _TorchSampler(PI0Pytorch):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.config = config
        self.paligemma_with_expert = _TorchBackbone()
        self.action_out_proj = torch.nn.Identity()

    def _preprocess_observation(self, obs, *, train=True):
        assert not train
        return (
            [],
            [],
            obs.tokenized_prompt,
            obs.tokenized_prompt_mask,
            obs.tokenized_positive_guidance_prompt,
            obs.tokenized_positive_guidance_prompt_mask,
            obs.tokenized_negative_guidance_prompt,
            obs.tokenized_negative_guidance_prompt_mask,
            obs.state,
        )

    def embed_prefix(self, images, masks, tokens, token_masks):
        return tokens[..., None].float(), token_masks, torch.zeros_like(token_masks)

    def embed_suffix(self, state, actions, time):
        return actions, torch.ones(actions.shape[:2], dtype=torch.bool), torch.tensor([[True, False]]), time


def _observation(framework):
    convert = jnp.asarray if framework == "jax" else torch.tensor
    return _model.Observation(
        images={},
        image_masks={},
        state=convert([[0.0]]),
        tokenized_prompt=convert([[1, 1]]),
        tokenized_prompt_mask=convert([[True, False]]),
        tokenized_positive_guidance_prompt=convert([[2, 2]]),
        tokenized_positive_guidance_prompt_mask=convert([[True, True]]),
        tokenized_negative_guidance_prompt=convert([[3, 3]]),
        tokenized_negative_guidance_prompt_mask=convert([[True, True]]),
    )


def _reference(scale, *, code=2, guided=True):
    x = np.array([[[0.25], [-0.5]]], dtype=np.float32)
    for i in range(4):
        time = 1 - i / 4
        original = 1 + 0.1 * x**2 + time + 0.01
        conditional = code * (1 + 0.1 * x**2) + time + 0.02
        velocity = (1 - scale) * original + scale * conditional if guided else original
        x = x - velocity / 4
    return x


@pytest.fixture
def _skip_image_preprocessing(monkeypatch):
    monkeypatch.setattr(_model, "preprocess_observation", lambda rng, observation, **kwargs: observation)


def _sample(framework, config, observation=None, *, jit=False):
    observation = observation if observation is not None else _observation(framework)
    noise = np.array([[[0.25], [-0.5]]], dtype=np.float32)
    if framework == "jax":
        sampler = _JaxSampler(config)
        method = module_jit(sampler.sample_actions) if jit else sampler.sample_actions
        output = method(jax.random.key(0), observation, num_steps=4, noise=jnp.asarray(noise))
    else:
        sampler = _TorchSampler(config)
        output = sampler.sample_actions("cpu", observation, num_steps=4, noise=torch.tensor(noise))
    return np.asarray(output), sampler


@pytest.mark.usefixtures("_skip_image_preprocessing")
@pytest.mark.parametrize("framework", ["jax", "torch"])
@pytest.mark.parametrize("guided", [False, True])
def test_statistical_constants_remain_zero_during_sampling(framework, guided):
    config = Pi0Config(
        action_dim=2, action_horizon=2, cfg_guidance=guided, cfgrl_guidance_scale=1.5, pytorch_compile_mode=None
    )
    convert = jnp.asarray if framework == "jax" else torch.tensor
    obs = dataclasses.replace(_observation(framework), action_loss_mask=convert([[False, True]]))
    noise = convert([[[999.0, 0.25], [-999.0, -0.5]]])
    if framework == "jax":
        sampler = _JaxSampler(config)
        sampler.action_dim = 2
        result = module_jit(sampler.sample_actions)(jax.random.key(0), obs, noise=noise, num_steps=4)
    else:
        sampler = _TorchSampler(config)
        result = sampler.sample_actions("cpu", obs, noise=noise, num_steps=4)
        for _, x_t, _ in sampler.paligemma_with_expert.calls:
            np.testing.assert_array_equal(x_t[..., 0].numpy(), 0.0)
    np.testing.assert_array_equal(np.asarray(result)[..., 0], 0.0)
    np.testing.assert_allclose(np.asarray(result)[..., 1:], _reference(1.5, guided=guided), atol=1e-6)


class _JaxTrainingBackbone(nnx.Module):
    def __call__(self, embs, **kwargs):
        return embs, None


class _JaxFixedVelocity(nnx.Module):
    def __init__(self):
        self.velocity = nnx.Param(jnp.asarray([1.0, 100.0]))

    def __call__(self, x):
        return jnp.broadcast_to(self.velocity.value, x.shape)


class _TorchTrainingBackbone(_TorchBackbone):
    def __init__(self):
        super().__init__()
        self.paligemma.language_model.layers = [
            SimpleNamespace(self_attn=SimpleNamespace(q_proj=torch.nn.Linear(1, 1)))
        ]

    def forward(self, *, inputs_embeds, **kwargs):
        self.suffix_input = inputs_embeds[1]
        return inputs_embeds, None


class _TorchFixedVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = torch.nn.Parameter(torch.tensor([1.0, 100.0]))

    def forward(self, x):
        return self.velocity.expand_as(x)


class _TorchTrainingSampler(_TorchSampler):
    def __init__(self, config):
        super().__init__(config)
        self.paligemma_with_expert = _TorchTrainingBackbone()
        self.action_out_proj = _TorchFixedVelocity()

    def _preprocess_observation(self, obs, *, train=True):
        return super()._preprocess_observation(obs, train=False)

    def _apply_checkpoint(self, func, *args):
        return func(*args)


@pytest.mark.usefixtures("_skip_image_preprocessing")
@pytest.mark.parametrize("framework", ["jax", "torch"])
@pytest.mark.parametrize("active", [[True, False], [False, False], [True, True]])
def test_constant_action_loss_and_gradients(framework, active):
    config = Pi0Config(action_dim=2, action_horizon=2, pytorch_compile_mode=None)
    convert = jnp.asarray if framework == "jax" else torch.tensor
    obs = dataclasses.replace(_observation(framework), action_loss_mask=convert([active]))
    actions = convert([[[0.5, 888.0], [0.1, -777.0]]])
    if framework == "jax":
        sampler = _JaxSampler(config)
        sampler.action_dim = 2
        sampler.PaliGemma = nnx.Dict(llm=_JaxTrainingBackbone())
        sampler.action_out_proj = _JaxFixedVelocity()
        key = jax.random.key(0)
        noise = jax.random.normal(jax.random.split(key, 3)[1], actions.shape)
        loss, grads = nnx.value_and_grad(lambda m: m.compute_loss(key, obs, actions).mean())(sampler)
        gradient = np.asarray(grads.action_out_proj.velocity.value)
    else:
        sampler = _TorchTrainingSampler(config)
        noise = convert([[[0.3, 123.0], [0.7, -999.0]]])
        loss = sampler(obs, actions, noise=noise, time=convert([0.5])).mean()
        loss.backward()
        gradient = sampler.action_out_proj.velocity.grad.numpy()
        np.testing.assert_array_equal(
            sampler.paligemma_with_expert.suffix_input.detach().numpy()[..., ~np.array(active)], 0
        )
    errors = (np.array([1.0, 100.0]) - (np.asarray(noise) - np.asarray(actions))) ** 2
    expected = np.where(active, errors, 0).sum(axis=-1).mean() / max(sum(active), 1)
    np.testing.assert_allclose(float(loss), expected, rtol=1e-6)
    np.testing.assert_array_equal(gradient[~np.array(active)], 0)


def test_observation_preserves_action_mask():
    obs = _observation("jax")
    obs = dataclasses.replace(obs, images={"test": jnp.zeros((1, 2, 2, 3))}, action_loss_mask=jnp.array([[False]]))
    restored = _model.Observation.from_dict(obs.to_dict())
    processed = _model.preprocess_observation(None, restored, image_keys=("test",), image_resolution=(2, 2))
    np.testing.assert_array_equal(processed.action_loss_mask, [[False]])


@pytest.mark.parametrize("framework", ["jax", "pytorch"])
@pytest.mark.parametrize("scale", [0.0, 1.0, 1.5, 2.0])
@pytest.mark.usefixtures("_skip_image_preprocessing")
def test_cfg_mixes_velocities_at_each_step(framework, scale):
    config = Pi0Config(cfg_guidance=True, cfgrl_guidance_scale=scale, pytorch_compile_mode=None)
    output, sampler = _sample(framework, config)
    np.testing.assert_allclose(output, _reference(scale), rtol=1e-6, atol=1e-6)
    if framework == "pytorch":
        backbone = sampler.paligemma_with_expert
        assert len(backbone.prefixes) == 2
        assert len(backbone.calls) == 8
        for original, conditional in zip(backbone.calls[::2], backbone.calls[1::2], strict=True):
            assert original[0] is backbone.prefixes[0]
            assert conditional[0] is backbone.prefixes[1]
            torch.testing.assert_close(original[1], conditional[1])
            torch.testing.assert_close(original[2], conditional[2])
        torch.testing.assert_close(backbone.prefixes[0], torch.tensor([[[1.0]]]))
        torch.testing.assert_close(backbone.prefixes[1], torch.tensor([[[2.0]]]))


@pytest.mark.usefixtures("_skip_image_preprocessing")
def test_jax_policy_jit_supports_cfg():
    config = Pi0Config(cfg_guidance=True, cfgrl_guidance_scale=1.5, pytorch_compile_mode=None)
    output, _ = _sample("jax", config, jit=True)
    np.testing.assert_allclose(output, _reference(1.5), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("framework", ["jax", "pytorch"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.usefixtures("_skip_image_preprocessing")
def test_original_route_without_guidance(framework, enabled):
    config = Pi0Config(cfg_guidance=enabled, guidance_type="no_guide", pytorch_compile_mode=None)
    obs = dataclasses.replace(
        _observation(framework), tokenized_positive_guidance_prompt=None, tokenized_positive_guidance_prompt_mask=None
    )
    output, sampler = _sample(framework, config, obs)
    np.testing.assert_allclose(output, _reference(1, guided=False), rtol=1e-6, atol=1e-6)
    if framework == "pytorch":
        assert len(sampler.paligemma_with_expert.prefixes) == 1


@pytest.mark.parametrize("framework", ["jax", "pytorch"])
@pytest.mark.usefixtures("_skip_image_preprocessing")
def test_missing_guidance_fails(framework):
    config = Pi0Config(cfg_guidance=True, pytorch_compile_mode=None)
    obs = dataclasses.replace(
        _observation(framework), tokenized_positive_guidance_prompt=None, tokenized_positive_guidance_prompt_mask=None
    )
    with pytest.raises(ValueError, match="guidance prompt"):
        _sample(framework, config, obs)


@pytest.mark.parametrize("framework", ["jax", "pytorch"])
@pytest.mark.usefixtures("_skip_image_preprocessing")
def test_negative_guidance_only_with_full_conditioning(framework):
    config = Pi0Config(cfg_guidance=True, guidance_type="negative", pytorch_compile_mode=None)
    with pytest.raises(ValueError, match="positive_only"):
        _sample(framework, config)
    config = dataclasses.replace(config, positive_only_conditional=False, cfgrl_guidance_scale=1.5)
    output, _ = _sample(framework, config)
    np.testing.assert_allclose(output, _reference(1.5, code=3), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("framework", ["jax", "pytorch"])
@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -1.0])
@pytest.mark.usefixtures("_skip_image_preprocessing")
def test_invalid_guidance_scale(framework, scale):
    config = Pi0Config(cfg_guidance=True, cfgrl_guidance_scale=scale, pytorch_compile_mode=None)
    with pytest.raises(ValueError, match="cfgrl_guidance_scale"):
        _sample(framework, config)
