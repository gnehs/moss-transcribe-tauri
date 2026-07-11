# Generated MOSS parity fixtures

Only the schema and this guide are versioned. Put a locally generated fixture
under `generated/<fixture-name>/`; it must contain `metadata.json` and
`tensors.npz`. The source audio must exceed 30 seconds at 16 kHz after
conversion. See [`scripts/parity/README.md`](../../../scripts/parity/README.md)
for generation and execution instructions.
