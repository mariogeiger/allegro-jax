from typing import Callable, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from .allegro_flax import  allegro_call, allegro_layer_call

        
class AllegroHaikuLayer(hk.Module):

    def __init__(
        self,
        avg_num_neighbors: float,
        max_ell: int = 3,
        output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e"),
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
        mlp_n_hidden: int = 64,
        mlp_n_layers: int = 3,
        p: int = 6,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.max_ell = max_ell
        self.output_irreps = output_irreps
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.p = p

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


class AllegroHaiku(hk.Module):

    def __init__(
        self,
        avg_num_neighbors: float,
        max_ell: int = 3,
        irreps: e3nn.Irreps = 128 * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e"),
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
        mlp_n_hidden: int = 1024,
        mlp_n_layers: int = 3,
        p: int = 6,
        n_radial_basis: int = 8,
        radial_cutoff: float = 1.0,
        output_irreps: e3nn.Irreps = e3nn.Irreps("0e"),
        num_layers: int = 3,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.max_ell = max_ell
        self.irreps = irreps
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.p = p
        self.n_radial_basis = n_radial_basis
        self.radial_cutoff = radial_cutoff
        self.output_irreps = output_irreps
        self.num_layers = num_layers

    def __call__(
        self,
        node_attrs: jnp.ndarray,  # jax.nn.one_hot(z, num_species)
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        edge_feats: Optional[e3nn.IrrepsArray] = None,  # [n_edges, irreps]
    ) -> e3nn.IrrepsArray:
        return allegro_call(
            e3nn.haiku.Linear,
            e3nn.haiku.MultiLayerPerceptron,
            self,
            node_attrs,
            vectors,
            senders,
            receivers,
            edge_feats,
        )
