# lbmpy-atmo: atmospheric flows with lbmpy

Make your life easier when using [lbmpy](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/index.html) for atmospheric flows!

**WARNING**: single GPU only for now

## Installation on Kraken

The installation must happen on a GPU-enabled node, you cannot perform it directly on the login nodes.
To aquire a GPU-enabled node, you can *e.g.* run:

```bash
salloc -p gpua30 --gres=gpu:4
```

and then `ssh` into the A30 GPU node you were allocated.

### Environment

Once on the node you need to load the necessary modules:

 1. A modern version of python (3.9 is fine, untested below)
 2. The correct CUDA driver. Tested with 11.2 on the A30 nodes
 3. A gcc compiler

**These must be loaded _before_ you perform the installation commands.**
If you don't know where to start, just use these commands:

```bash
module load python/tf2.6-cuda11.2-py39
module load nvidia/cuda/11.2
module load compiler/gcc/9.4.0
```

### Package install

For now, we recommend a "development" install. We also recommend setting up a dedicated virtual environment for this.

```bash
python -m venv env_atmo
source env_atmo/bin/activate
git clone https://gitlab.com/cerfacs/lbmpy-atmo.git
cd lbmpy-atmo
pip install cupy-cuda11x
pip install -e .
```

This installs everything you need. **However**, CuPy needs to be installed
[specifically for your version of Cuda](https://cupy.dev/), so adapt the line
appropriately.

## Running on Kraken

For convenience, once everything is installed you can add this to your `.bashrc` (adapt as needed):

```bash
function boot_atmo {
    module load python/tf2.6-cuda11.2-py39
    module load nvidia/cuda/11.2
    module load compiler/gcc/9.4.0
    cd <path-to-your-lbmpy-atmo-work>
    source <path-to-your-environment>
}
```
