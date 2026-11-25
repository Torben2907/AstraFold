"""
Writes a copy of a metadata CSV without the chains that resemble a set of held-out
RNAs, e.g. to pretrain on RNA3DB without the CASP15 test targets (RNA3DB contains
several of them, e.g. 7qr4_B = R1107) or the RNAs of a validation split.

A chain is removed if, compared with any held-out RNA, it
  - comes from the same PDB entry (held-out CSV rows only), or
  - shares most of its k-mers (overlap coefficient |A & B| / min(|A|, |B|) >=
    --kmer-overlap on the sets of k-mers), the criterion scripts/make_val_split.py
    uses to keep near-identical RNAs on one side of a split.

The input CSV is not modified.

Usage:
    python scripts/exclude_similar.py --csv metadata/RNA3DB_train_split.csv \\
        --exclude-fasta benchmarks/CASP15/CASP15_test.fasta \\
        --exclude-csv metadata/CASP15_val.csv \\
        --out metadata/RNA3DB_train_split_noCASP15.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from make_val_split import kmers, read_sequence


def read_fasta(path: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    name = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            sequences[name] = ""
        elif name is not None:
            sequences[name] += line.upper().replace("T", "U")
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-fasta", nargs="*", default=[])
    parser.add_argument("--exclude-csv", nargs="*", default=[])
    parser.add_argument("--kmer-size", type=int, default=5)
    parser.add_argument("--kmer-overlap", type=float, default=0.7)
    args = parser.parse_args()
    if Path(args.out).resolve() == Path(args.csv).resolve():
        parser.error("--out must differ from --csv")

    held_out: dict[str, str] = {}  # name -> sequence
    held_out_pdb_ids: dict[str, str] = {}  # PDB id -> name
    for fasta in args.exclude_fasta:
        held_out.update(read_fasta(fasta))
    for csv in args.exclude_csv:
        for row in pd.read_csv(csv).itertuples():
            held_out[row.pdb_name] = read_sequence(row.processed_path)
            held_out_pdb_ids[row.pdb_name[:4].lower()] = row.pdb_name
    held_out_kmers = {
        name: kmers(seq, args.kmer_size) for name, seq in held_out.items()
    }

    df = pd.read_csv(args.csv)
    reasons = []
    for row in df.itertuples():
        pdb_id = row.pdb_name[:4].lower()
        if pdb_id in held_out_pdb_ids:
            reasons.append(f"same PDB entry as {held_out_pdb_ids[pdb_id]}")
            continue
        chain_kmers = kmers(read_sequence(row.processed_path), args.kmer_size)
        reason = ""
        for name, other in held_out_kmers.items():
            overlap = len(chain_kmers & other) / min(len(chain_kmers), len(other))
            if overlap >= args.kmer_overlap:
                reason = f"k-mer overlap {overlap:.2f} with {name}"
                break
        reasons.append(reason)

    removed = pd.Series(reasons, index=df.index) != ""
    df.loc[~removed].to_csv(args.out, index=False)

    print(f"{args.csv}: {len(df)} chains, {removed.sum()} removed")
    print(f"{(~removed).sum()} -> {args.out}")
    for pdb_name, reason in zip(df.loc[removed, "pdb_name"], pd.Series(reasons)[removed]):
        print(f"  {pdb_name}: {reason}")


if __name__ == "__main__":
    main()
