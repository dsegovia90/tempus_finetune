"""`cactus finetune` without its closing pass that generates an answer for every validation
example (about 45 minutes on ml for 1,129 examples); validation loss is still printed per epoch.

    python finetune_unscored.py data.jsonl --epochs 6 --out adapter.safetensors
"""

import sys

from cactus.cli.parser import parser
from cactus.cli.logging import configure
from cactus.model.finetune import finetune_local

if __name__ == "__main__":
    configure()
    args = parser().parse_args(["finetune", *sys.argv[1:]])
    args.score = False  # read by finetune_local; the CLI has no flag for it
    finetune_local(args)
