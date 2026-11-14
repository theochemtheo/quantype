# How-to guides

Choose a task once you know the basics from [Getting started](../tutorials/getting-started.md).

| Task | Use it to |
| --- | --- |
| [Work with arrays](arrays.md) | Index, reduce, and transform quantities without dropping physical meaning |
| [Test quantities](test-quantities.md) | Compare results in different units, including temperature tolerances |
| [Validate data](validate-data.md) | Turn external configuration into typed quantities |
| [Serialize quantities](serialize.md) | Save values as JSON, NPZ, or NPY without pickle |
| [Differentiate with JAX](autodiff-jax.md) | Compute physical derivatives and compile a model; install `quantype[jax]` |
| [Differentiate with Torch](autodiff-torch.md) | Preserve derivative kinds through autograd; install `quantype[torch]` |
| [Work with temperatures](temperatures.md) | Convert points, shift them by differences, and compute thermal energies |
| [Define units](define-units.md) | Add a local unit or calibration for an existing kind |
| [Choose constants](choose-constants.md) | Calculate with physical constants and select a CODATA edition |
| [Define systems](define-systems.md) | Match the raw units of a numerical kernel or simulator |
| [Generate a catalogue](custom-catalogues.md) | Add kinds and relations in a separate nominal API; requires Ruff |

For an exact signature or restriction, use [Reference](../reference/index.md).
For a failure you are already seeing, use [Troubleshooting](../troubleshooting.md).
