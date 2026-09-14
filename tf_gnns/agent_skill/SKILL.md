---
name: tf-gnns-user-docs
description: Use when writing code against tf_gnns graph data structures, GraphNet/MPNN layers, Sparse GCN layers, or Keras backend-portable graph tensor dictionaries.
---

# tf_gnns User Docs

## Fast Retrieval

- List topics: `python -m tf_gnns.agent_docs list`
- Read a topic (full page): `python -m tf_gnns.agent_docs get quickstart`
- **Read concepts first** (How It Works, Decision Guide, Parameter Key Facts, Boundaries): `python -m tf_gnns.agent_docs get graphnets --concepts`
- Search with compact results: `python -m tf_gnns.agent_docs search "GraphTuple|SparseGCN" --limit 2`
- Avoid `--examples` by default; fall back only when concepts + search are not enough.
- Install into a worktree: `python -m tf_gnns.agent_docs install-opencode-skill --project-root .`

## Available Topics

- `quickstart`: read when creating a small GraphTuple and running one model.
- `graphs`: read when building `Node`, `Edge`, `Graph`, and `GraphTuple` inputs.
- `graphnets`: read when using `GraphNet` factories, MPNN layers, globals, and aggregators.
- `gcn`: read when using `SparseGCNConv`, `SparseGCN`, or `GCNv2` on tensor dictionaries.
- `backends`: read when choosing TensorFlow, Torch, or JAX Keras backends.

## Agent Workflow

- Read the concept sections of the matching page first: `get <topic> --concepts` (How It Works, Decision Guide, Parameter Key Facts, Boundaries & Gotchas).
- Derive code from the rules, shape table, and parameter key facts. Do not read `--examples` by default; prefer `search "<keyword>" --limit 2` for the exact signature or output shape. Keep code lean and uncommented.
- Prefer `python -m tf_gnns.agent_docs` over reading library source. The docs cover every public task this suite tests.
- Use `search` for signatures and API cards before guessing argument names; keep each query narrow and its results small.
- Every reference page has an `## API Signatures (authoritative)` section listing exact constructors, kwargs, and defaults. Never use Python `inspect` or read `tf_gnns` source to learn a signature — the docs page already lists it.
- Each reference page also documents an **Output Contract** (which keys and shapes are produced); trust it instead of probing with trial scripts.
- Derive code from the rules, shape table, and parameter key facts; write the solution file first, then validate and iterate; keep edits inside the allowed file.

## Environment

- Run Python scripts and tests with `uv run python <script>` and `uv run pytest ...`.
- The project dependencies (TensorFlow, Keras 3) are already available in the project uv environment; never hunt for system Python installs or create new virtualenvs in `/tmp`.
- The worktree root is `tf_gnns`; import the package directly (`from tf_gnns import ...`).
- Backend selection happens at application level with `tf.function`; do not modify tf_gnns internals.

## Import Rules

- Core graph data: `from tf_gnns import Node, Edge, Graph, GraphTuple, make_graph_tuple_from_graph_list`
- GraphNet factories: `from tf_gnns.graphnet_utils import make_mlp_graphnet_functions, make_full_graphnet_functions, make_graph_indep_graphnet_functions, make_mpnn_graphnet_noglobal_functions`
- Layer wrappers: `from tf_gnns.models.graphnet import GNCellMLP, GraphNetMLP, GraphIndep, GraphNetMPNN_MLP`
- GCN layers: `from tf_gnns.models.gcn import SparseGCNConv, SparseGCN, GCNv2`
- Backend facade: `from tf_gnns import backend_ops`
- `GCNv2` is intentionally not exported from the package root; import it from `tf_gnns.models.gcn`.

## Core Usage Rules

- Tensor dictionaries use keys `nodes`, `edges`, `senders`, `receivers`, `n_nodes`, `n_edges`, `n_graphs`, `global_attr`, `global_reps_for_nodes`, and `global_reps_for_edges`.
- `senders` and `receivers` are flattened node indices over the whole graph batch.
- Object `Node` feature tensors must have rank at least 2; single-object graph features usually use shape `[1, feature_dim]`.
- `make_graph_tuple_from_graph_list` expects object graph node and edge tensors with first dimension `1`.
- GraphNet edge/node/global model inputs are routed by Keras input names, not positional order.

## Boundaries

- Do not use removed `safe` or `batched` `GraphNet.graph_eval` modes.
- Do not expect Torch eager paths to outperform PyG; use them for Keras backend portability.
- Do not enable both `batchnorm` and `layernorm` on GCN layers.
- `GraphNet.save()` serializes node/edge/aggregation functions only: reloaded blocks lose global updates and cannot re-run message passing. Use a no-global graph-independent block for a faithful roundtrip.
- Do not rely on package-root exports for every model; use submodule imports shown above.

## Minimal Runnable Pattern

```python
import tensorflow as tf
from tf_gnns import Edge, Graph, Node, make_graph_tuple_from_graph_list
from tf_gnns.models.graphnet import GraphNetMPNN_MLP

n0 = Node(tf.constant([[1.0, 0.0]], dtype=tf.float32))
n1 = Node(tf.constant([[0.0, 1.0]], dtype=tf.float32))
e01 = Edge(tf.constant([[0.5, 0.2]], dtype=tf.float32), n0, n1)
graph = Graph([n0, n1], [e01])

gt = make_graph_tuple_from_graph_list([graph])
td = gt.to_tensor_dict()
model = GraphNetMPNN_MLP(units=8, core_steps=1)
out = model(td)
print(out["nodes"].shape)
```
