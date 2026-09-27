# Tiny synthetic example

A complete run of the model on a small synthetic data folder: 12
municipalities of 4 buurten, a skim store for the 4 buurten of one
municipality, synthetic jobs and household tables, and the reference files
of the repository. It takes a few seconds.

    python examples/tiny/make_data.py examples/tiny/data
    python -m ikob2.cli.accessibility --data-root examples/tiny/data \
        --study tiny --run example --modes car bike pt

The run writes `examples/tiny/data/outputs/runs/example/` (the same products
as a real run; see docs/data_specification.md). `expected/accessibility.csv`
is its `accessibility.csv`; `tests/test_example.py` rebuilds the folder,
reruns the model and compares, so it doubles as an end-to-end test.

What the example shows: 480 rows (4 origins x 3 modes x 40 segments).
Decile 1 carries the atom of the cost margin, so it reaches only the free
intrazonal car trips and nothing by public transport; the bicycle, which has
no cost, is the same for all deciles up to their job pools. All numbers are
synthetic: they illustrate the mechanics, not the Utrecht case.

`make_data.py` documents how every input is made. Rebuild the stored output
after an intended change of results with the two commands above and
`cp examples/tiny/data/outputs/runs/example/accessibility.csv examples/tiny/expected/`.
