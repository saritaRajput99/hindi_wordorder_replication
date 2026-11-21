"""
Loader module for UD_Hindi-HDTB (TRAIN ONLY)


"""

from pathlib import Path
from typing import List, Dict, Optional


def parse_feats(feats_str: str) -> Dict[str, str]:
    """
    Parse a CoNLL-U feats string into a dict.

    Examples:
        "Case=Acc|Gender=Masc|Number=Sing"
          → {"Case": "Acc", "Gender": "Masc", "Number": "Sing"}
        "_"  →  {}
        ""   →  {}
    """
    if not feats_str or feats_str == "_":
        return {}
    result = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def feats_to_str(feats: Dict[str, str]) -> str:
    """
    Serialize a feats dict back to CoNLL-U string format.

    Example:  {"Case": "Acc", "Gender": "Masc"}  →  "Case=Acc|Gender=Masc"
    Empty dict → "_"
    """
    if not feats:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))


class HUTBLoader:
    """Load UD_Hindi-HDTB data (train split only)."""

    # Both naming conventions used for the same file
    CANDIDATE_FILENAMES = [
        "hi_hdtb-ud-train.conllu",   # official UD HDTB name
        "train.conllu",               # user-renamed shorthand
    ]

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    # ──────────────────────────────────────────
    # INTERNAL: resolve filename
    # ──────────────────────────────────────────

    def _find_train_file(self) -> Path:
        """
        Try both filename conventions and return the first that exists.
        Raises FileNotFoundError if neither is found.
        """
        for name in self.CANDIDATE_FILENAMES:
            p = self.data_path / name
            if p.exists():
                return p
        tried = " | ".join(self.CANDIDATE_FILENAMES)
        raise FileNotFoundError(
            f"Could not find train file in {self.data_path}.\n"
            f"Tried: {tried}\n"
            f"Download hi_hdtb-ud-train.conllu from "
            f"https://github.com/UniversalDependencies/UD_Hindi-HDTB "
            f"and place it in {self.data_path}"
        )

    # ──────────────────────────────────────────
    # LOAD
    # ──────────────────────────────────────────

    def load_train(self) -> List[Dict]:
        """
        Load UD_Hindi-HDTB train split.

        Returns list of sentence dicts:
            {
                "sent_id" : str,       # from '# sent_id = ...'
                "text"    : str,       # from '# text = ...'   ← FIX 2
                "tokens"  : [          # one dict per token
                    {
                        "id"      : int,
                        "form"    : str,
                        "lemma"   : str,
                        "upostag" : str,
                        "xpostag" : str,
                        "feats"   : dict,   # ← FIX 1: parsed, not raw string
                        "head"    : int,
                        "deprel"  : str,
                        "deps"    : str,
                        "misc"    : str,
                    }, ...
                ]
            }
        """
        filepath = self._find_train_file()
        print(f"Loading: {filepath}")

        sentences        = []
        current_tokens   = []
        current_sent_id  = None
        current_text     = None           # FIX 2

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # ── Sentence boundary ──────────────────
                if not line:
                    if current_tokens:
                        sentences.append({
                            "sent_id" : current_sent_id,
                            "text"    : current_text,   # FIX 2
                            "tokens"  : current_tokens,
                        })
                        current_tokens  = []
                        current_sent_id = None
                        current_text    = None
                    continue

                # ── Comment lines ──────────────────────
                if line.startswith("#"):
                    if "sent_id" in line:
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            current_sent_id = parts[1].strip()
                    elif line.startswith("# text"):        # FIX 2
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            current_text = parts[1].strip()
                    continue

                # ── Token lines ────────────────────────
                parts = line.split("\t")

                if len(parts) < 10:          # malformed
                    continue
                if "-" in parts[0]:          # multiword token  (e.g. 3-4)
                    continue
                if "." in parts[0]:          # empty node       (e.g. 1.1)
                    continue

                try:
                    token = {
                        "id"      : int(parts[0]),
                        "form"    : parts[1],
                        "lemma"   : parts[2],
                        "upostag" : parts[3],
                        "xpostag" : parts[4],
                        "feats"   : parse_feats(parts[5]),   # FIX 1
                        "head"    : int(parts[6]),
                        "deprel"  : parts[7],
                        "deps"    : parts[8],
                        "misc"    : parts[9],
                    }
                    current_tokens.append(token)

                except ValueError:
                    continue   # skip tokens with non-integer id/head

        # Last sentence if file doesn't end with a blank line
        if current_tokens:
            sentences.append({
                "sent_id" : current_sent_id,
                "text"    : current_text,
                "tokens"  : current_tokens,
            })

        print(f"Loaded {len(sentences)} sentences from {filepath.name}")
        return sentences

    # ──────────────────────────────────────────
    # SAVE
    # ──────────────────────────────────────────

    def save_conllu(self, sentences: List[Dict], output_path: str):
        """Save sentences back to CoNLL-U format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, sent in enumerate(sentences):
                sent_id = sent.get("sent_id", f"sent_{i+1}")
                tokens  = sent["tokens"]

                f.write(f"# sent_id = {sent_id}\n")
                f.write(
                    f"# text = "
                    + (sent.get("text") or " ".join(t["form"] for t in tokens))
                    + "\n"
                )

                for t in tokens:
                    # feats: serialize dict back to CoNLL-U string
                    feats_val = t.get("feats", {})
                    if isinstance(feats_val, dict):
                        feats_str = feats_to_str(feats_val)
                    else:
                        feats_str = feats_val or "_"

                    line = (
                        f"{t['id']}\t{t['form']}\t{t['lemma']}\t"
                        f"{t['upostag']}\t{t.get('xpostag','_')}\t"
                        f"{feats_str}\t{t['head']}\t"
                        f"{t['deprel']}\t{t.get('deps','_')}\t"
                        f"{t.get('misc','_')}\n"
                    )
                    f.write(line)

                f.write("\n")

        print(f"Saved {len(sentences)} sentences to {output_path}")
