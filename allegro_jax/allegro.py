from typing import List, Optional
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp


def normalized_bessel(d: jnp.ndarray, n: int) -> jnp.ndarray:
    with jax.ensure_compile_time_eval():
        r = jnp.linspace(0.0, 1.0, 1000, dtype=d.dtype)
        b = e3nn.bessel(r, n)
        mu = jax.scipy.integrate.trapezoid(b, r, axis=0)
        sig = jax.scipy.integrate.trapezoid((b - mu) ** 2, r, axis=0) ** 0.5
    return (e3nn.bessel(d, n) - mu) / sig


def u(d: jnp.ndarray, p: int) -> jnp.ndarray:
    return e3nn.poly_envelope(p - 1, 2)(d)


def filter_layers(layer_irreps: List[e3nn.Irreps], max_ell: int) -> List[e3nn.Irreps]:
    layer_irreps = list(layer_irreps)
    filtered = [e3nn.Irreps(layer_irreps[-1])]
    for irreps in reversed(layer_irreps[:-1]):
        irreps = e3nn.Irreps(irreps)
        irreps = irreps.filter(
            keep=e3nn.tensor_product(
                filtered[0],
                e3nn.Irreps.spherical_harmonics(lmax=max_ell),
            ).regroup()
        )
        filtered.insert(0, irreps)
    return filtered


def allegro_layer_call(
    Linear,
    MultiLayerPerceptron,
    output_irreps: e3nn.Irreps,
    self,
    vectors: e3nn.IrrepsArray,  # [n_edges, 3]
    x: jnp.ndarray,  # [n_edges, features]
    V: e3nn.IrrepsArray,  # [n_edges, irreps]
    senders: jnp.ndarray,  # [n_edges]
) -> e3nn.IrrepsArray:
    num_edges = vectors.shape[0]
    assert vectors.shape == (num_edges, 3)
    assert x.shape == (num_edges, x.shape[-1])
    assert V.shape == (num_edges, V.irreps.dim)
    assert senders.shape == (num_edges,)

    irreps_out = e3nn.Irreps(output_irreps)

    w = MultiLayerPerceptron((V.irreps.mul_gcd,), act=None)(x)
    Y = e3nn.spherical_harmonics(range(self.max_ell + 1), vectors, True)
    wY = e3nn.scatter_sum(
        w[:, :, None] * Y[:, None, :], dst=senders, map_back=True
    ) / jnp.sqrt(self.avg_num_neighbors)
    assert wY.shape == (num_edges, V.irreps.mul_gcd, wY.irreps.dim)

    V = e3nn.tensor_product(
        wY, V.mul_to_axis(), filter_ir_out="0e" + irreps_out
    ).axis_to_mul()

    if "0e" in V.irreps:
        x = jnp.concatenate([x, V.filter(keep="0e").array], axis=1)
        V = V.filter(drop="0e")

    x = MultiLayerPerceptron(
        (self.mlp_n_hidden,) * self.mlp_n_layers,
        self.mlp_activation,
        output_activation=False,
    )(x)
    lengths = e3nn.norm(vectors).array
    x = u(lengths, self.p) * x
    assert x.shape == (num_edges, self.mlp_n_hidden)

    V = Linear(irreps_out)(V)
    assert V.shape == (num_edges, V.irreps.dim)

    return (x, V)


def allegro_call(
    Linear,
    MultiLayerPerceptron,
    self,
    node_attrs: jnp.ndarray,  # jax.nn.one_hot(z, num_species)
    vectors: e3nn.IrrepsArray,  # [n_edges, 3]
    senders: jnp.ndarray,  # [n_edges]
    receivers: jnp.ndarray,  # [n_edges]
    edge_feats: Optional[e3nn.IrrepsArray] = None,  # [n_edges, irreps]
) -> e3nn.IrrepsArray:
    num_edges = vectors.shape[0]
    num_nodes = node_attrs.shape[0]
    assert vectors.shape == (num_edges, 3)
    assert node_attrs.shape == (num_nodes, node_attrs.shape[-1])
    assert senders.shape == (num_edges,)
    assert receivers.shape == (num_edges,)

    assert vectors.irreps in ["1o", "1e"]
    irreps = e3nn.Irreps(self.irreps)
    irreps_out = e3nn.Irreps(self.output_irreps)

    irreps_layers = [irreps] * self.num_layers + [irreps_out]
    irreps_layers = filter_layers(irreps_layers, self.max_ell)

    vectors = vectors / self.radial_cutoff

    d = e3nn.norm(vectors).array.squeeze(1)
    x = jnp.concatenate(
        [
            normalized_bessel(d, self.n_radial_basis),
            node_attrs[senders],
            node_attrs[receivers],
        ],
        axis=1,
    )
    assert x.shape == (num_edges, self.n_radial_basis + 2 * node_attrs.shape[-1])

    # Protection against exploding dummy edges:
    x = jnp.where(d[:, None] == 0.0, 0.0, x)

    x = MultiLayerPerceptron(
        (
            self.mlp_n_hidden // 8,
            self.mlp_n_hidden // 4,
            self.mlp_n_hidden // 2,
            self.mlp_n_hidden,
        ),
        self.mlp_activation,
        output_activation=False,
    )(x)
    x = u(d, self.p)[:, None] * x
    assert x.shape == (num_edges, self.mlp_n_hidden)

    irreps_Y = irreps_layers[0].filter(
        keep=lambda mir: vectors.irreps[0].ir.p ** mir.ir.l == mir.ir.p
    )
    V = e3nn.spherical_harmonics(irreps_Y, vectors, True)

    if edge_feats is not None:
        V = e3nn.concatenate([V, edge_feats])
    w = MultiLayerPerceptron((V.irreps.num_irreps,), act=None)(x)
    V = w * V
    assert V.shape == (num_edges, V.irreps.dim)

    for irreps in irreps_layers[1:]:
        y, V = allegro_layer_call(
            Linear,
            MultiLayerPerceptron,
            irreps,
            self,
            vectors,
            x,
            V,
            senders,
        )

        alpha = 0.5
        x = (x + alpha * y) / jnp.sqrt(1 + alpha**2)

    x = MultiLayerPerceptron((128,), act=None)(x)

    xV = Linear(irreps_out)(e3nn.concatenate([x, V]))

    if xV.irreps != irreps_out:
        raise ValueError(f"output_irreps {irreps_out} is not reachable")

    return xV
