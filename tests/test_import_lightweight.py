# SPDX-License-Identifier: MIT
"""Import-hygiene regression: a text-only deploy must be able to `import superl8serve`
without torchvision present. The model registry eagerly imports the VLM family
(models/__init__.py -> vlm -> multimodal -> preprocess), so a top-level torchvision
import there would hard-require it even to serve a text model. The live deploy
installs torchvision with `|| true`, so a failed install would silently kill the
server on restart. The VLM image path must import torchvision lazily.
"""

import subprocess
import sys


def test_import_superl8serve_does_not_eagerly_import_torchvision():
    """Importing superl8serve must NOT pull torchvision as a side effect."""
    code = "import superl8serve, sys; sys.exit(1 if 'torchvision' in sys.modules else 0)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        "torchvision was imported at `import superl8serve` time — a text-only deploy "
        f"would then hard-require it. stderr:\n{r.stderr}"
    )


def test_preprocess_module_imports_without_torchvision_at_top_level():
    """The preprocess module itself must import without torchvision installed."""
    code = (
        "import sys, types;"
        # simulate torchvision being absent
        "sys.modules['torchvision'] = None;"
        "import importlib;"
        "m = importlib.import_module('superl8serve.multimodal.preprocess');"
        "sys.exit(0 if hasattr(m, 'resize_image') else 2)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"preprocess module failed to import with torchvision absent:\n{r.stderr}"
    )
