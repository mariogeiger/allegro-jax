from typing import Callable, Optional

import e3nn_jax as e3nn
import flax
import jax
import jax.numpy as jnp


def normalized_bessel(d: jnp.ndarray, n: int) -> jnp.ndarray:
    with jax.ensure_compile_time_eval():
        r = jnp.linspace(0.0, 1.0, 1000, dtype=d.dtype)
        b = e3nn.bessel(r, n)
        mu = jnp.trapz(b, r, axis=0)
        sig = jnp.trapz((b - mu) ** 2, r, axis=0) ** 0.5
    return (e3nn.bessel(d, n) - mu) / sig


def u(d: jnp.ndarray, p: int) -> jnp.ndarray:
    return e3nn.poly_envelope(p - 1, 2)(d)


class AllegroLayer(flax.linen.Module):
    avg_num_neighbors: float
    max_ell: int = 3
    output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e")
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 3
    p: int = 6

    @flax.linen.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        x: jnp.ndarray,  # [n_edges, features]
        V: e3nn.IrrepsArray,  # [n_edges, n, irreps]
        senders: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        irreps_out = e3nn.Irreps(self.output_irreps)
        n = V.shape[1]

        w = e3nn.flax.MultiLayerPerceptron((n,))(x)  # (edge, n)
        Y = e3nn.spherical_harmonics(
            range(self.max_ell + 1), vectors, True
        )  # (edge, irreps)
        wY = w[:, :, None] * Y[:, None, :]  # (edge, n, irreps)
        wY = e3nn.scatter_sum(wY, dst=senders, map_back=True) / jnp.sqrt(
            self.avg_num_neighbors
        )  # (edge, n, irreps)

        V = e3nn.tensor_product(
            wY, V, filter_ir_out="0e" + irreps_out
        )  # (edge, n, irreps)

        if "0e" in V.irreps:
            x = jnp.concatenate([x, V.filter(keep="0e").axis_to_mul().array], axis=1)
            V = V.filter(drop="0e")

        x = e3nn.flax.MultiLayerPerceptron(
            (self.mlp_n_hidden,) * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False,
        )(
            x
        )  # (edge, features_out)
        lengths = e3nn.norm(vectors).array
        x = u(lengths, self.p) * x  # (edge, features_out)

        V = V.axis_to_mul()  # (edge, n * irreps)
        V = e3nn.flax.Linear(irreps_out)(V)  # (edge, n_out * irreps_out)
        V = V.mul_to_axis()  # (edge, n_out, irreps_out)

        return (x, V)


class Allegro(flax.linen.Module):
    avg_num_neighbors: float
    max_ell: int = 3
    irreps: e3nn.Irreps = 128 * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e")
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    mlp_n_hidden: int = 1024
    mlp_n_layers: int = 3
    p: int = 6
    n_radial_basis: int = 8
    radial_cutoff: float = 1.0
    output_irreps: e3nn.Irreps = e3nn.Irreps("0e")
    num_layers: int = 3

    @flax.linen.compact
    def __call__(
        self,
        node_attrs: jnp.ndarray,  # jax.nn.one_hot(z, num_species)
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        edge_feats: Optional[e3nn.IrrepsArray] = None,  # [n_edges, irreps]
    ) -> e3nn.IrrepsArray:
        assert vectors.irreps in ["1o", "1e"]
        irreps = e3nn.Irreps(self.irreps)
        irreps_out = e3nn.Irreps(self.output_irreps)

        assert vectors.shape == senders.shape + (3,)

        vectors = vectors / self.radial_cutoff

        d = e3nn.norm(vectors).array.squeeze(1)  # (edge,)
        x = jnp.concatenate(
            [
                normalized_bessel(d, self.n_radial_basis),
                node_attrs[senders],
                node_attrs[receivers],
            ],
            axis=1,
        )

        # Protect against exploding dummy edges
        x = jnp.where(d[:, None] == 0.0, 0.0, x)  # (edge, features)

        x = e3nn.flax.MultiLayerPerceptron(
            (
                self.mlp_n_hidden // 8,
                self.mlp_n_hidden // 4,
                self.mlp_n_hidden // 2,
                self.mlp_n_hidden,
            ),
            self.mlp_activation,
            output_activation=False,
        )(
            x
        )  # (edge, features)
        x = u(d, self.p)[:, None] * x  # (edge, features)

        V = e3nn.spherical_harmonics(
            range(irreps.lmax + 1), vectors, True
        )  # (edge, irreps)

        if edge_feats is not None:
            V = e3nn.concatenate([V, edge_feats])  # (edge, irreps)

        w = e3nn.flax.MultiLayerPerceptron((irreps.mul_gcd,))(x)  # (edge, n)
        V = w[:, :, None] * V[:, None, :]  # (edge, n, irreps)

        for _ in range(self.num_layers):
            y, V = AllegroLayer(
                avg_num_neighbors=self.avg_num_neighbors,
                max_ell=self.max_ell,
                output_irreps=self.irreps,
                mlp_activation=self.mlp_activation,
                mlp_n_hidden=self.mlp_n_hidden,
                mlp_n_layers=self.mlp_n_layers,
                p=self.p,
            )(
                vectors,
                x,
                V,
                senders,
            )

            alpha = 0.5
            x = (x + alpha * y) / jnp.sqrt(1 + alpha**2)

        x = e3nn.flax.MultiLayerPerceptron((128,))(x)  # (edge, 128)

        xV = e3nn.concatenate([e3nn.as_irreps_array(x), V.axis_to_mul()])
        xV = e3nn.flax.Linear(irreps_out)(xV)  # (edge, irreps_out)

        return xV
