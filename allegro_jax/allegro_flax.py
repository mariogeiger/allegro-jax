from typing import Callable, Optional

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp

from .allegro import allegro_call, allegro_layer_call


class AllegroLayer(nn.Module):
    avg_num_neighbors: float
    max_ell: int = 3
    output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e")
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 3
    p: int = 6

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        x: jnp.ndarray,  # [n_edges, features]
        V: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        return allegro_layer_call(
            e3nn.flax.Linear,
            e3nn.flax.MultiLayerPerceptron,
            self,
            vectors,
            x,
            V,
            senders,
        )


class Allegro(nn.Module):
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

    @nn.compact
    def __call__(
        self,
        node_attrs: jnp.ndarray,  # jax.nn.one_hot(z, num_species)
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        edge_feats: Optional[e3nn.IrrepsArray] = None,  # [n_edges, irreps]
    ) -> e3nn.IrrepsArray:

        return allegro_call(
            e3nn.flax.Linear,
            e3nn.flax.MultiLayerPerceptron,
            self,
            node_attrs,
            vectors,
            senders,
            receivers,
            edge_feats,
        )
