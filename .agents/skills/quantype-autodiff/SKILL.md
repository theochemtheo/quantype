---
name: quantype-autodiff
description: Maintain Quantype JAX jit/vmap/pytree integration and JAX or Torch differentiation that preserves raw values, unit systems and physical derivative kinds. Use for gradient, Hessian, graph preservation or transformation bugs.
---

# Quantype Autodiff

Parent: read [numerical backends](../quantype-backends/SKILL.md), including its shared guidance. Read [core](../quantype-core/SKILL.md) if changing derivative-kind algebra.

## Physical derivatives

Differentiate raw numerical values. Derivative semantics use `core.result_kind("div", output_kind, input_kind)`; Hessians divide a second time. Energy differentiated with respect to length gives `Force`, then `ForceConstant`. These adapters return positive derivatives; physical force is explicitly the negative gradient. Changing the display/input unit must not change the derivative.

Output and input must share a unit system; the derivative is then already in that system's coherent units, with no rescaling. The adapters raise `TypeError` for an output in another system rather than relying on the invariant silently. Static signatures are generic over the system (`[S: UnitSystem]`).

Current differentiation scope is one quantity input and scalar quantity output. Preserve scalar-output failures. Do not claim general multi-input, JVP or VJP support unless implementing and verifying it as requested. Model code using `.value` and `from_value` asserts the raw model's physical meaning in its unit system; a quantity-algebra example instead needs physically typed coefficients.

## JAX

`src/quantype/ujax.py` registers the base quantity and generated quantity classes as single-leaf pytrees on adapter import. The leaf is the raw `.value`; metadata holds semantic identity and unit system, never the display unit (it would make presentation part of tree structure and break `lax.cond`/`scan`). Unflattening must restore both without coercing leaves, including sentinels JAX supplies while manipulating trees.

`_raw_function` carries output metadata through `has_aux=True` so kind discovery does not evaluate the user's function separately. Preserve that behavior under `grad`, `hessian`, `jit` and `vmap`. Reuse physical algebra rather than deriving kinds from raw dimensions.

Generated application adapters register their own classes through `codegen/_package.py`. Registration and nominal identity checks belong in isolated generated-project subprocesses; see [catalogue](../quantype-catalogue/SKILL.md).

## Torch

`src/quantype/utorch.py` differentiates the existing scalar output tensor with respect to the input quantity's raw tensor. Preserve `create_graph` and `retain_graph` semantics so higher derivatives remain possible. A source tensor must already have `requires_grad=True`; do not recreate tensors or detach them during arithmetic, unit conversion or differentiation.

Host extraction such as `.detach().cpu().numpy()` belongs to an explicit consumer or serialization boundary. Both adapters must remain usable when the other backend is absent.

## Validate behavior and signatures

Use `tests/runtime/test_autodiff.py` for all-class single-leaf registration, metadata, JIT/vmap, `scan`/`cond` and derivatives in every built-in and one custom system, single model evaluation and nonscalar-output failures; `test_scenarios_jax.py` covers the prototype scenarios. Use `test_torch.py` for graphs, conversion invariance and higher derivatives; `test_optional_backends.py` for isolation.

`ujax.pyi` and `utorch.pyi` are generated. For changed public signatures or derivative relations, update their owning renderers/definitions and read [typing](../quantype-typing/SKILL.md). Verify generated consumers when the change affects application adapters.
