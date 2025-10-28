"""Convert MixtureAgentsModels MATLAB exports into the `.npz` bundle consumed by
`series_hmm_rnn.run_real_data_pipeline`.

The script only runs when SciPy is available.  It prints a warning and exits
otherwise because the hosted evaluation environment cannot install optional
packages.  When running locally ensure SciPy is installed and point `--input`
at one of the behavioural `.mat` files shown in the MixtureAgentsModels
repository screenshot supplied by the user.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scipy.io import loadmat  # type: ignore
except Exception as exc:  # pragma: no cover - handled by CLI guard
    loadmat = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

import numpy as np


def extract_arrays(mat_payload):
    """Extract trial-wise arrays from the MATLAB payload.

    The MATLAB files ship as nested structs.  Different releases of the dataset
    occasionally rename the fields, so this function checks a few aliases and
    raises an informative error when required entries are missing.
    """

    candidates = [
        ("actions", ("choice", "actions", "choice_stay")),
        ("rewards", ("reward", "rewards")),
        ("transitions", ("transition", "transitions", "trans")),
    ]
    extracted = {}
    for target, aliases in candidates:
        for alias in aliases:
            if alias in mat_payload:
                extracted[target] = np.asarray(mat_payload[alias]).astype(np.int64)
                break
        else:  # no break
            raise KeyError(f"missing {target} field (tried {aliases})")
    if "phases" in mat_payload:
        extracted["phases"] = np.asarray(mat_payload["phases"]).astype(np.int64)
    return extracted


def normalise_shape(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected rank-1/2 array, found shape {arr.shape}")
    return arr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to MATLAB .mat file")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file")
    args = parser.parse_args()

    if loadmat is None:
        raise SystemExit(
            "scipy is required to read MATLAB files. Install scipy locally and rerun this script."  # noqa: E501
        ) from _IMPORT_ERROR

    payload = loadmat(args.input, squeeze_me=True, struct_as_record=False)
    arrays = extract_arrays(payload)
    normalised = {k: normalise_shape(v) for k, v in arrays.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **normalised)
    print(f"saved {args.output} with keys: {sorted(normalised)}")


if __name__ == "__main__":
    main()
