---
name: quantype-autodiff
description: Maintain Quantype JAX jit/vmap/pytree integration and JAX or Torch differentiation that preserves canonical values and physical derivative kinds. Use for gradient, Hessian, graph preservation or transformation bugs.
---

# Quantype Autodiff

Parent: read [numerical backends](../quantype-backends/SKILL.md), including its shared guidance. Read [core](../quantype-core/SKILL.md) if changing derivative-kind algebra.

## Physical derivatives

Differentiate canonical numerical values. Derivative semantics use `core.result_kind("div", output_kind, input_kind)`; Hessians divide a second time. Energy differentiated with respect to length gives `Force`, then `ForceConstant`. These adapters return positive derivatives; physical force is explicitly the negative gradient. Changing the display/input unit must not change the canonical derivative.

Current differentiation scope is one quantity input and scalar quantity output. Preserve scalar-output failures. Do not claim general multi-input, JVP or VJP support unless implementing and verifying it as requested. Model code using `.value` and `from_canonical` asserts the raw model's canonical physical meaning; a quantity-algebra example instead needs physically typed coefficients.

## JAX

`src/quantype/ujax.py` registers the base quantity and generated quantity classes as single-leaf pytrees on adapter import. The leaf is canonical `.value`; metadata holds semantic identity and display unit. Unflattening must preserve both without coercing leaves, including sentinels JAX supplies while manipulating trees.

`_raw_function` carries output metadata through `has_aux=True` so kind discovery does not evaluate the user's function separately. Preserve that behavior under `grad`, `hessian`, `jit` and `vmap`. Reuse physical algebra rather than deriving kinds from raw dimensions.

Generated application adapters register their own classes through `codegen/_package.py`. Registration and nominal identity checks belong in isolated generated-project subprocesses; see [catalogue](../quantype-catalogue/SKILL.md).

## Torch

`src/quantype/utorch.py` differentiates the existing scalar output tensor with respect to the input quantity's canonical tensor. Preserve `create_graph` and `retain_graph` semantics so higher derivatives remain possible. A source tensor must already have `requires_grad=True`; do not recreate tensors or detach them during arithmetic, unit conversion or differentiation.

Host extraction such as `.detach().cpu().numpy()` belongs to an explicit consumer or serialization boundary. Both adapters must remain usable when the other backend is absent.

## Validate behavior and signatures

Use `tests/runtime/test_autodiff.py` for all-class single-leaf registration, metadata, JIT/vmap, conversion-invariant derivatives, single model evaluation and nonscalar-output failures. Use `test_torch.py` for graphs, conversion invariance and higher derivatives; `test_optional_backends.py` for isolation.

`ujax.pyi` and `utorch.pyi` are generated. For changed public signatures or derivative relations, update their owning renderers/definitions and read [typing](../quantype-typing/SKILL.md). Verify generated consumers when the change affects application adapters.
