# llama-engine

A C++ LLM inference engine built from scratch.

The goal of this project is to understand how modern LLM inference
works at a low level by implementing the core components ourselves.

## Current Progress

- [x] Tensor storage
- [x] Tensor shapes
- [x] Strides
- [x] Multi-dimensional indexing
- [x] Tensor filling
- [x] Element-wise addition
- [x] Shape validation
- [x] Basic tests

## Planned

- [x] Element-wise multiplication
- [x] Matrix multiplication
- [ ] Softmax
- [ ] Reshape / transpose
- [ ] Normalization
- [ ] Transformer components
- [ ] Model loading
- [ ] Llama inference
- [ ] KV cache
- [ ] Performance optimizations

----------------------------------------------------------------------------------------------------------------
# C++ Llama Inference Engine
## Inference Engine Requirements, API Contracts & Implementation Roadmap

**Document status:** Project specification / source of truth  
**Last updated:** 13 November 2026  
**Project:** `llama-engine`  
**Primary implementation language:** C++  
**Initial execution backend:** CPU  
**Target model family:** Small autoregressive, decoder-only Transformer (Llama-style architecture)

---

# 1. Project objective

Build a real, standalone C++ inference runtime capable of loading the parameters and metadata of a small autoregressive Transformer and generating text without relying on a Python inference framework at runtime.

The project is both a functioning software system and a deep systems-engineering exercise. Correctness, explicit contracts, numerical behaviour, memory layout, and the ability to validate each layer of the system take priority over prematurely optimising or adding features.

The intended model-development workflow is:

1. Train or obtain a small decoder-only Transformer using PyTorch.
2. Export its learned parameters, model configuration, and tokenizer/vocabulary metadata.
3. Load those artifacts in the C++ runtime.
4. Execute a forward pass and produce next-token logits.
5. Generate tokens autoregressively, using a KV cache so previously computed attention keys and values do not need to be recomputed at every decoding step.
6. Profile and optimise the runtime while preserving numerical correctness.

The initial practical target is a lightweight model in approximately the 15M–50M parameter range, subject to the capabilities of the development machine. The runtime should be designed so that the model dimensions and configuration are not hard-coded into individual operators.

## 1.1 Explicit non-goals for the first working runtime

The first working inference engine does not need:

- A GPU/CUDA backend.
- Distributed inference or tensor parallelism.
- A production HTTP server or hosted API.
- Training or backpropagation in C++.
- Every possible PyTorch operator.
- Every checkpoint format from day one.
- Quantisation before a correct floating-point implementation exists.

These are possible later extensions, not prerequisites for the first end-to-end CPU inference milestone.

---

# 2. Engineering principles

These principles govern implementation decisions throughout the project.

1. **Requirements before implementation.** Define the operation's mathematical behaviour, accepted shapes, output shape, validation rules, and mutation semantics before coding.
2. **Preserve working code.** Do not rewrite a completed component merely because a broader feature has been identified. Prefer a justified extension that preserves existing behaviour.
3. **Separate correctness from optimisation.** First establish a correct reference implementation; then measure and optimise it.
4. **Make tensor semantics explicit.** Shape, strides, layout, broadcasting, dtype, and axis semantics must be documented rather than inferred.
5. **Avoid accidental restrictions.** An operator should not be restricted to rank-2 tensors unless its mathematical/API contract genuinely requires that restriction.
6. **Do not overgeneralise without a use case.** Support the generality required by the target Transformer and define deliberate boundaries for other cases.
7. **Test numerical results, not only shapes.** Tests must verify values, validation failures, and mutation guarantees.
8. **Keep layers of responsibility separate.** Tensor storage, mathematical operators, neural-network layers, model architecture, file loading, and generation/runtime orchestration are distinct responsibilities.
9. **No silent scope changes.** Any proposed change to a previously agreed contract must explain the requirement, compatibility impact, implementation cost, and whether existing work can be retained.

---

# 3. Target model architecture

The first end-to-end target is a decoder-only autoregressive Transformer using Llama-style components.

A typical model consists of:

1. Token embedding table.
2. A stack of Transformer decoder blocks.
3. Final RMSNorm.
4. Language-model output projection (LM head).
5. Logits processing and token sampling.

Each decoder block contains:

1. Pre-attention RMSNorm.
2. Query, key, and value projections.
3. Rotary positional embeddings (RoPE) applied to queries and keys.
4. Causal self-attention.
5. Output projection.
6. Residual connection.
7. Pre-feed-forward RMSNorm.
8. Feed-forward projections using the SwiGLU activation pattern.
9. Residual connection.

The model configuration must specify dimensions such as vocabulary size, hidden size, number of layers, attention-head count, key/value-head count where applicable, intermediate/feed-forward size, maximum context length, RoPE parameters, and numerical dtype.

The architecture must be driven by model configuration and loaded weights, not by constants embedded throughout the implementation.

---

# 4. Tensor core

## 4.1 Tensor representation

The Tensor abstraction is responsible for:

- Owning its data.
- Storing an arbitrary-rank shape.
- Storing or deriving row-major strides.
- Mapping multidimensional coordinates to linear storage offsets.
- Exposing controlled const and mutable data access.
- Validating shape and coordinate operations.
- Reporting rank, shape, element count, and strides.

Initial storage layout: contiguous, row-major, `float32`.

The design should leave room for future non-contiguous views and additional dtypes, but those should not be added until their requirements are defined.

## 4.2 Shape and indexing rules

For a contiguous row-major tensor with shape `[D0, D1, ..., Dn-1]`, strides are:

`stride[i] = product(shape[j]) for j > i`

The linear offset for coordinate `[i0, i1, ..., in-1]` is:

`offset = sum(i[j] * stride[j])`

Coordinate indexing must validate:

- Coordinate rank equals tensor rank.
- Every coordinate is within its corresponding dimension.
- The computed offset is within storage.

Zero-sized dimensions and zero-rank/scalar tensors need explicit, consistent handling. Operations must not accidentally call `front()`, `back()`, or index a dimension that does not exist.

## 4.3 Tensor API capabilities

The tensor core should provide or support:

- Construction from a shape.
- Construction from shape and initial values, with element-count validation.
- `rank()` / number of dimensions.
- `get_shape()`.
- `get_strides()`.
- `numel()` / element count.
- Const and mutable data access.
- Checked `at(coordinates)` access.
- `fill(value)`.
- Copy semantics with independent owned storage.
- Clearly defined move semantics.

A future view/reshape system may need shared storage and offset/stride metadata. Do not assume that the current owning contiguous Tensor already supports views.

---

# 5. Operator API contracts

This section defines the target mathematical behaviour. Exact C++ names may follow the existing codebase, but semantics should remain consistent.

## 5.1 Elementwise addition

For the initial API:

- Accept two or more tensors of identical shape.
- Return a tensor of that same shape.
- Do not mutate inputs.
- Reject incompatible shapes.

Broadcasting for general elementwise addition is a separate extension. It must not be accidentally implied by the word “addition.”

Residual connections require shape-compatible elementwise addition. Identical-shape addition is sufficient for the initial Transformer implementation.

## 5.2 Elementwise multiplication

Initial contract:

- Inputs have identical shapes.
- Multiply corresponding elements.
- Output shape equals input shape.
- Inputs remain unchanged.

This operator is needed for gated activations such as SwiGLU.

## 5.3 Matrix multiplication (`matmul`)

The target API should implement standard tensor `matmul` semantics, rather than a matrix-only operation.

Supported operand categories:

| Left operand A | Right operand B | Result |
|---|---|---|
| `[K]` | `[K]` | Scalar (rank-0 result) |
| `[K]` | `[K, N]` | `[N]` |
| `[M, K]` | `[K]` | `[M]` |
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[..., M, K]` | `[..., K, N]` | `[..., M, N]`, subject to batch broadcasting |

For N-dimensional operands, the final two dimensions are matrix dimensions. Leading dimensions are batch dimensions and must be broadcast according to standard right-aligned broadcasting rules:

- Dimensions are compatible when they are equal or one of them is `1`.
- Missing leading dimensions are treated as `1`.
- The output batch dimension is the maximum compatible dimension.
- Incompatible batch dimensions must produce a clear error.

Vector operands are treated specially:
- A rank-1 left operand is promoted conceptually to a row vector for multiplication, then its synthetic matrix dimension is removed from the result.
- A rank-1 right operand is promoted conceptually to a column vector, then its synthetic matrix dimension is removed.
- If both operands are rank 1, the result is a scalar (rank 0).

The contraction dimensions must match.

### Implementation guidance

The existing batched matrix-matrix implementation is valuable and must be preserved. Vector support and batch broadcasting should be added as carefully scoped extensions. Do not discard or replace the existing batch-coordinate/indexing work without evidence that it is incorrect.

The implementation may use distinct internal paths for vector-vector, vector-matrix, matrix-vector, and matrix/batched-matrix cases, provided the public API has consistent semantics.

The `iterate()` helper's responsibility must be documented. If it currently iterates matrix rows/columns or batch coordinates under assumptions that exclude vectors, adapt it deliberately or introduce a narrowly scoped helper. Do not change a helper blindly without reviewing its callers and invariants.

### Required validation

- Both operands have rank at least 1.
- Contracting dimensions match.
- Batch dimensions are broadcast-compatible.
- Output shape follows the rules above.
- Invalid shapes fail predictably.
- Inputs are not mutated.

## 5.4 Bias addition (`add_bias`)

Mathematical operation: add one rank-1 bias vector along the final dimension of an input tensor.

Contract:

- Input rank is at least 1.
- Bias rank is exactly 1.
- `input.shape.back() == bias.shape[0]`.
- Output shape is exactly the input shape.
- For every input coordinate, add the bias element indexed by the input coordinate's final component.
- The same bias vector is reused across all leading dimensions.
- Neither input nor bias is mutated.

Examples:

- `[N] + [N] -> [N]`
- `[M, N] + [N] -> [M, N]`
- `[B, S, N] + [N] -> [B, S, N]`
- `[D1, D2, ..., N] + [N] -> [D1, D2, ..., N]`

This is feature-axis broadcasting. It does not create additional features or dimensions.

The implementation should work for arbitrary leading dimensions, not special-case rank 2. A straightforward contiguous implementation can use the final dimension size as the bias-cycle length. If the Tensor representation later supports non-contiguous views, the implementation must be revisited to respect strides rather than assuming flat contiguous storage.

## 5.5 Linear layer

Mathematical operation:

`Y = XW + b`

Target contract:

- Input `X` has rank at least 1 and shape `[..., K]`.
- Weights `W` have exactly rank 2 and shape `[K, N]`.
- Bias `b` has exactly rank 1 and shape `[N]` when the layer is configured with bias.
- Output shape is `[..., N]`.
- The final input dimension must equal the first weight dimension.
- Bias length must equal the second weight dimension.
- Input, weights, and bias are not mutated.
- The layer owns or otherwise has a clearly documented lifetime/ownership relationship with its parameters.

A single feature vector `[K]` must produce `[N]`. A batch `[B,K]` produces `[B,N]`. A Transformer input `[B,S,K]` produces `[B,S,N]`.

The layer should compose `matmul` and `add_bias`, rather than duplicating their mathematics.

Whether bias is optional should be an explicit API decision. The first implementation may require bias if that matches the current class design; bias-free projections can be added through a documented constructor/configuration option when required by the target model.

## 5.6 Activation functions

Required for the target architecture:

- ReLU may remain as an elementary operator and test of elementwise behaviour.
- SiLU/Swish: `x * sigmoid(x)`.
- GELU if needed by the selected model configuration or experiments.
- SwiGLU composition: `silu(xW_gate) * (xW_up)`, followed by the down projection.

Activation operators should preserve shape and not mutate inputs unless explicitly named as in-place operations.

## 5.7 Softmax

Required for attention and optionally output sampling.

Contract:

- Accept arbitrary-rank input.
- Operate along an explicitly specified axis/dimension.
- Preserve input shape.
- Produce normalised values along that axis.
- Use a numerically stable algorithm: subtract the maximum value along the axis before exponentiation.
- Define behaviour for invalid axes, empty dimensions, infinities, and NaNs.
- Tests should verify sums near 1 within a dtype-appropriate tolerance.

A softmax implementation that only handles a flattened tensor is not sufficient for attention.

## 5.8 Shape and layout operations

Required capabilities:

- Reshape, with element-count preservation.
- Permute/transpose dimensions.
- Contiguous conversion if non-contiguous views are introduced.
- Explicit axis/dimension validation.

Attention requires transformations between layouts such as `[B,S,H*D]`, `[B,S,H,D]`, and `[B,H,S,D]`. The API must not confuse a reshape (which changes shape interpretation) with a transpose/permutation (which changes dimension order).

If the implementation only supports contiguous owning tensors initially, document and enforce that limitation.

## 5.9 Reductions

At minimum, support reductions needed by normalisation and attention:

- Sum along an axis.
- Mean along an axis.
- Maximum along an axis.

Reduction APIs must define whether the reduced dimension is retained (`keepdims`) and must validate axes.

## 5.10 Normalisation

Required: RMSNorm.

For a vector `x` of hidden features:

`RMSNorm(x) = (x / sqrt(mean(x * x) + epsilon)) * weight`

Requirements:

- Normalise over the final hidden dimension.
- Support arbitrary leading dimensions.
- Learnable scale vector shape `[hidden_size]`.
- Configurable epsilon.
- Stable accumulation behaviour.
- No mutation of input or parameters unless explicitly documented.

LayerNorm is optional unless the selected model architecture requires it.

## 5.11 Rotary positional embeddings (RoPE)

Required for the selected Llama-style architecture.

Requirements:

- Apply rotary position information to query and key tensors.
- Respect the model's configured head dimension and RoPE parameters.
- Define the rotary pairing convention (for example, adjacent-pair or split-half) and match the checkpoint/model implementation exactly.
- Support position offsets so decoding can continue at the correct token position.
- Test known small cases and consistency between full-sequence prefill and incremental decoding.

## 5.12 Embedding lookup

Required:

- Token IDs index rows of an embedding table.
- Validate token IDs against vocabulary size.
- Return embeddings with the token dimensions preserved and hidden size appended.
- Support batches and sequences of token IDs.
- Define the integer type used for token IDs separately from floating-point tensor data.

## 5.13 Attention primitives

Required functionality:

- Query, key, and value tensor handling.
- Scaled dot-product attention.
- Causal masking so a position cannot attend to future positions.
- Stable softmax along the key/time axis.
- Weighted sum of values.
- Correct head and batch dimension handling.
- Support the selected model's attention layout and head configuration.

The implementation must document tensor layouts at every stage. For example, distinguish `[B,S,H,D]` from `[B,H,S,D]`; do not rely on ambiguous variable names.

Grouped-query attention (GQA) or multi-query attention must be supported if the selected model configuration uses it. The implementation should not assume the number of query heads always equals the number of key/value heads.

## 5.14 Residual connections

Use shape-compatible elementwise addition. For the initial model, both residual operands should have identical shapes. Residual addition must not mutate either input.

---

# 6. Transformer layer and model composition

## 6.1 Decoder block

Implement a decoder block by composing already-tested operators:

1. Save residual input.
2. Apply pre-attention RMSNorm.
3. Compute Q, K, V projections.
4. Reshape and arrange heads.
5. Apply RoPE to Q and K.
6. Perform causal self-attention, using cached K/V when decoding.
7. Merge attention heads.
8. Apply attention output projection.
9. Add the attention residual.
10. Save the new residual.
11. Apply pre-feed-forward RMSNorm.
12. Compute gate and up projections.
13. Apply SiLU to the gate branch and multiply by the up branch.
14. Apply down projection.
15. Add the feed-forward residual.

The exact order and projection details must match the chosen architecture/checkpoint.

## 6.2 Full model

The model should:

1. Accept token IDs.
2. Perform embedding lookup.
3. Execute decoder blocks in configured order.
4. Apply final RMSNorm.
5. Apply the language-model head.
6. Return logits for the next-token distribution.

The model should not assume a fixed batch size, sequence length, vocabulary size, or hidden dimension.

---

# 7. Model artifacts, configuration, and loading

## 7.1 Required artifacts

- Model weights.
- Model configuration.
- Tokenizer model or vocabulary/merge data, depending on tokenizer type.
- Special-token metadata.
- Optional generation defaults.

## 7.2 Checkpoint/export strategy

Initial strategy: define a simple, versioned export format from the PyTorch model to the C++ runtime.

The format should include:

- Magic identifier.
- Format version.
- Tensor count.
- For each tensor: name, rank, shape, dtype, data length, and payload.
- Explicit byte order.
- Validation of lengths and bounds.
- A way to identify missing, duplicate, or unexpected parameter names.

Model configuration should be stored in a separate documented file or in a versioned metadata section.

The exporter must preserve exact tensor names and architecture conventions expected by the C++ loader.

Later, add compatibility with established formats such as GGUF if it materially advances the project. Do not make GGUF support a prerequisite for the first working end-to-end model.

## 7.3 Loader requirements

- Read and validate file headers and versions.
- Reject truncated or malformed files safely.
- Validate tensor dimensions and payload sizes.
- Map parameter names to model components.
- Check model configuration against loaded parameter shapes.
- Report missing, unexpected, and incompatible tensors clearly.
- Avoid silently accepting corrupt or mismatched weights.

---

# 8. Tokenizer and text generation

## 8.1 Tokenizer

The runtime needs a tokenizer compatible with the model used to produce the weights.

Requirements:

- Encode input text into token IDs.
- Decode generated token IDs into text.
- Handle special tokens.
- Match the training/export tokenizer exactly.
- Validate vocabulary bounds.

A tokenizer mismatch can make an otherwise numerically correct model unusable.

## 8.2 Generation

Required generation loop:

1. Encode prompt.
2. Run prompt prefill.
3. Obtain logits for the next token.
4. Apply configured logits processing.
5. Select/sample a token.
6. Append token to output.
7. Stop on EOS or configured stopping condition.
8. Feed the new token through incremental decoding using the KV cache.

Initial sampling options:

- Greedy decoding.
- Temperature.
- Top-k sampling.
- Top-p/nucleus sampling.
- Reproducible random seed.

Sampling parameters must have validated ranges and deterministic tests where applicable.

---

# 9. KV cache and execution modes

KV caching is a core inference requirement for efficient autoregressive generation.

## 9.1 Prefill

Process the prompt sequence, computing keys and values for all prompt positions and storing them in the cache.

## 9.2 Incremental decode

For each newly generated token:

- Compute its query, key, and value.
- Append the new key/value to the appropriate cache.
- Attend to cached positions up to the current position.
- Do not recompute K/V projections for all previous tokens.

## 9.3 Cache requirements

- Cache storage per layer and per key/value head.
- Correct position tracking.
- Configurable maximum context length.
- Bounds checks and clear overflow behaviour.
- Correct handling of batch dimensions.
- Memory-size calculations based on model configuration and dtype.
- Tests comparing cached incremental output against an uncached reference path within numerical tolerance.

Cache layout should be selected deliberately for efficient writes and attention reads. Do not optimise its layout before a correct reference implementation exists.

---

# 10. Runtime architecture

Keep responsibilities separated.

Suggested conceptual components:

- `Tensor`: storage, shape, strides, indexing.
- `operations`: mathematical tensor operators.
- `Linear` and normalisation layers: reusable neural-network layers.
- `Attention`: attention-specific composition.
- `TransformerBlock`: decoder block.
- `Model`: full network composition and parameter ownership.
- `Tokenizer`: text/token conversion.
- `ModelLoader` / checkpoint reader: model artifact loading.
- `Generation` / sampler: autoregressive loop and sampling.
- `KVCache`: cached attention state.
- CLI/application entry point: user-facing execution and configuration.

The exact directory structure can evolve. Do not create abstractions solely for aesthetic reasons; each component should have a clear responsibility and testable contract.

---

# 11. Numerical correctness and data types

## 11.1 Initial dtype

Start with `float32` for weights, activations, and accumulations unless a specific operation requires a different type. Token IDs and indices must use integer types.

## 11.2 Numerical requirements

- Avoid unstable softmax.
- Use an appropriate epsilon in normalisation.
- Avoid unintended integer overflow in shape/element-count calculations.
- Validate dimensions before allocation.
- Use tolerance-based floating-point comparisons in tests.
- Document accumulation behaviour.
- Detect non-finite values during debugging where useful.

## 11.3 Later dtype support

After the float32 CPU path works:

- FP16 and/or BF16 storage and computation.
- Quantised weight formats such as INT8 or lower-bit schemes.
- Mixed-precision accumulation rules.

Quantisation is not just changing the tensor dtype. It requires defined scaling, zero-point/grouping metadata where applicable, quantise/dequantise or quantised kernels, and numerical evaluation.

---

# 12. Testing and validation strategy

Testing is required at every layer; passing a single end-to-end example is not enough.

## 12.1 Tensor tests

- Shape and rank.
- Stride calculation.
- Element count.
- Coordinate-to-offset mapping.
- Bounds errors.
- Copy/move behaviour.
- Empty and degenerate dimensions.
- Const and mutable access.

## 12.2 Operator tests

For every operator:

- Valid small examples with manually calculated expected values.
- Shape correctness.
- Numerical correctness.
- Invalid shape/rank/axis rejection.
- Input immutability.
- Edge cases, including rank-1 and higher-rank inputs where supported.
- Batch broadcasting cases for `matmul`.
- Tolerance-based comparisons.

## 12.3 Layer tests

- Linear vector, matrix, and higher-rank input cases.
- RMSNorm known-value cases.
- RoPE position and layout cases.
- Attention masking and head-layout cases.
- SwiGLU composition.
- Residual shape and value checks.

## 12.4 Model tests

- Tiny deterministic model with known parameters.
- Compare C++ intermediate outputs/logits against a PyTorch reference.
- Compare full-sequence prefill against token-by-token execution.
- Compare cached and uncached decoding.
- Verify EOS and stopping behaviour.
- Verify malformed model files fail safely.

## 12.5 Integration acceptance test

A first end-to-end milestone is achieved when the C++ program can:

1. Load a compatible exported model and tokenizer.
2. Tokenize a prompt.
3. Run the model forward pass.
4. Produce finite next-token logits.
5. Generate at least one token.
6. Decode generated tokens into text.
7. Repeat generation using the KV cache.
8. Match a PyTorch reference within documented numerical tolerances.

A model producing plausible text is not by itself proof of correctness; reference comparisons are required.

---

# 13. Performance and systems roadmap

Optimisation begins only after a correct reference path exists and is measurable.

## 13.1 Profiling

Measure:

- Prompt prefill latency.
- Per-token decode latency.
- Tokens per second.
- Operator-level time.
- Memory usage and allocation frequency.
- KV-cache memory and access costs.
- Model loading time.

Record model configuration, hardware, compiler, build flags, and benchmark methodology.

## 13.2 CPU optimisation progression

Potential work, ordered by evidence and value:

1. Reduce unnecessary allocations and copies.
2. Improve memory locality and tensor layout.
3. Add efficient matrix multiplication kernels.
4. Use compiler optimisation and inspect generated code.
5. Add SIMD paths appropriate to supported hardware (for example AVX2/AVX-512 or ARM NEON).
6. Add threading/parallel execution where benchmarks justify it.
7. Fuse operations to reduce memory traffic and intermediate tensors.
8. Improve KV-cache layout and attention kernels.
9. Add batching where it improves throughput without unacceptable latency.

Do not assume a SIMD or multithreaded implementation is faster; benchmark it against the reference.

## 13.3 Later extensions

After reliable CPU inference:

- Quantised model loading and kernels.
- GPU/CUDA backend.
- More advanced batching and serving.
- Additional checkpoint formats.
- Performance portability across CPU architectures.

---

# 14. Implementation progression and milestones

This is the project sequence. A milestone is complete only when its acceptance criteria are met.

## Milestone 0 — Requirements and contracts
- Maintain this specification.
- Confirm target model architecture and export path.
- Define shape, broadcasting, dtype, mutation, and error semantics for each operator.

**Exit criteria:** no ambiguous contract for an operator before implementation begins.

## Milestone 1 — Tensor foundation
- Complete and validate arbitrary-rank Tensor storage.
- Verify row-major strides and coordinate indexing.
- Confirm ownership, copy, and bounds behaviour.

**Exit criteria:** Tensor tests cover ordinary, higher-rank, and degenerate cases.

## Milestone 2 — Core tensor operations
- Addition.
- Multiplication.
- Existing batched matrix multiplication.
- Extend `matmul` to vector cases and broadcast-compatible batch dimensions according to the contract.
- Generalise `add_bias` to rank-1 and higher input tensors.
- Complete `Linear` for input rank >= 1, rank-2 weights, and rank-1 bias.

**Exit criteria:** numerical, shape, validation, broadcasting, and immutability tests pass. Existing batched matmul behaviour remains intact.

## Milestone 3 — Shape, reduction, and activation primitives
- Reshape and dimension permutation/transpose.
- Sum, mean, and max reductions.
- Stable axis-based softmax.
- SiLU and required activation composition.

**Exit criteria:** operations support the axes/layouts required by attention and normalisation.

## Milestone 4 — Transformer layers
- RMSNorm.
- RoPE.
- Embedding lookup.
- Causal mask.
- Scaled dot-product attention.
- Head reshape/transpose.
- SwiGLU.
- Residual composition.

**Exit criteria:** each component passes isolated tests and matches a small reference calculation.

## Milestone 5 — Transformer block and model
- Compose a decoder block.
- Compose the full decoder-only model.
- Drive architecture from configuration.
- Compare intermediate activations and logits against PyTorch.

**Exit criteria:** a tiny model forward pass matches the reference within tolerance.

## Milestone 6 — Export, loading, and tokenizer
- Implement PyTorch export.
- Define versioned weights format and model configuration.
- Load and validate weights in C++.
- Load a compatible tokenizer and special-token metadata.

**Exit criteria:** C++ loads the exported model without Python at inference time.

## Milestone 7 — End-to-end generation
- Prompt encoding.
- Prefill.
- Logits processing and sampling.
- Autoregressive decode.
- EOS/stopping conditions.

**Exit criteria:** the runtime generates decoded text from a prompt.

## Milestone 8 — KV cache
- Cache K/V per layer.
- Implement incremental decode.
- Handle positions, batches, and context limits.
- Compare cached and uncached execution.

**Exit criteria:** cached generation matches the reference path and avoids recomputing prior-token K/V.

## Milestone 9 — Correctness hardening and usability
- Error handling for invalid files/configurations/inputs.
- Reproducible tests and example model.
- CLI arguments and useful diagnostics.
- Documentation of model compatibility and limitations.

**Exit criteria:** another developer can build the project, load the documented model artifact, and run a prompt.

## Milestone 10 — Profiling and optimisation
- Establish baseline benchmarks.
- Optimise allocation and memory movement.
- Improve matmul and attention kernels.
- Add SIMD and threading based on measurements.
- Re-run correctness tests after every optimisation.

**Exit criteria:** measurable improvements with unchanged correctness within tolerance.

## Milestone 11 — Advanced backends and formats
- Quantisation.
- GGUF or other external model formats.
- GPU backend.
- Advanced batching/serving.

**Exit criteria:** each extension has its own requirements, tests, and benchmark evidence.

---

# 15. Current project status (13 November 2026)

The following reflects the known project state and should be updated as work progresses.

## Existing foundation

- Repository/project: `llama-engine`.
- `Tensor` supports arbitrary-rank shape metadata, row-major data, strides, checked indexing, and const/mutable data access.
- Elementwise addition has been implemented with same-shape validation.
- Elementwise multiplication has been started; verify its completion and tests.
- `matmul` has a working batched matrix-matrix implementation using batch-coordinate iteration. It currently handles matching batch dimensions; vector cases and batch broadcasting need to be assessed/extended against the contract in this document.
- ReLU has been implemented as a non-mutating operation.
- `add_bias` has been implemented for rank-2 input and rank-1 bias. The planned extension is rank-1-or-higher input with bias applied along the final dimension.
- `Linear` has been introduced with owned weights and bias, constructor validation, and a `forward()` that composes `matmul` and `add_bias`. Its validation must be aligned with the final contract.
- CMake test target `tensor_tests` exists.
- The project has not yet reached a complete Transformer forward pass or standalone text generation.

## Immediate next work

1. Preserve the current `matmul` implementation.
2. Review its existing public contract and helper invariants (`iterate()` included).
3. Extend it incrementally to the agreed vector and batch-broadcasting semantics, only where required by the target API.
4. Generalise `add_bias` to input rank >= 1.
5. Align `Linear` validation with input rank >= 1, rank-2 weights, and rank-1 bias.
6. Add/update tests for all newly agreed cases before moving to the next operator family.

Do not begin unrelated model/runtime components until these core contracts are implemented and tested.

---

# 16. Definition of project completion

The first substantial project completion target is not merely a collection of tensor operators.

It is a standalone C++ CPU inference runtime that:

- Loads a documented, compatible small decoder-only Transformer checkpoint.
- Loads the matching tokenizer and model configuration.
- Runs a numerically validated forward pass.
- Performs prompt prefill.
- Generates autoregressively.
- Uses a correct KV cache for incremental decoding.
- Produces decoded text.
- Has tests comparing its outputs with a trusted PyTorch reference.
- Has documented build/run instructions and reproducible baseline benchmarks.

Performance optimisation, quantisation, and GPU execution are subsequent engineering stages, not substitutes for this correctness milestone.

---

# 17. Change-control rule

This specification is the shared project contract.

When a new requirement is proposed:

1. Identify the real inference use case that needs it.
2. Determine whether the current contract already supports it.
3. Identify affected APIs, helpers, tests, and callers.
4. Explain whether the change is an extension, a breaking change, or an optional feature.
5. Estimate what existing implementation work can be preserved.
6. Update this specification before implementing a material scope change.

The developer/learner should retain ownership of implementation decisions. Guidance should explain trade-offs and help reason through designs; it should not silently redirect the project or replace the learner's implementation with unsolicited full solutions.

**Guiding rule: build the system we specified, preserve sound work, and change direction only for an explicit engineering reason.**