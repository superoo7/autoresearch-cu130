FROM nvcr.io/nvidia/pytorch:25.11-py3

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV CUDA_PATH=$CUDA_HOME
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

# Install triton from source for latest blackwell support
RUN git clone https://github.com/triton-lang/triton.git && \
    cd triton && \
    git checkout c5d671f91d90f40900027382f98b17a3e04045f6 && \
    pip install -r python/requirements.txt && \
    pip install . && \
    cd ..

# Install xformers from source for blackwell support
RUN git clone -b v0.0.33 --depth=1 https://github.com/facebookresearch/xformers --recursive && \
    cd xformers && \
    export TORCH_CUDA_ARCH_LIST="12.1" && \
    python setup.py install && \
    cd ..

# Install unsloth and deps for Qwen3.5 (transformers 5.x)
# --no-deps on unsloth to preserve CUDA torch
RUN pip install --no-deps bitsandbytes==0.48.0 && \
    pip install "transformers>=5.2.0" "trl>=0.18.0" "peft>=0.15.0" "accelerate>=1.0.0" datasets && \
    pip install --no-cache-dir --no-deps \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
# Patch: unsloth exec()s transformers config source code in globals().
# transformers 5.x configs use new symbols (auto_docstring, strict, etc.)
# that aren't in unsloth's globals. Inject all needed imports.
RUN python3 << 'PATCH'
import glob
files = glob.glob("/usr/local/lib/**/unsloth/models/_utils.py", recursive=True)
if not files:
    print("WARNING: _utils.py not found")
else:
    path = files[0]
    with open(path) as f:
        code = f.read()
    # Inject imports for all symbols that transformers 5.x config classes need
    patch = """
# --- Patch for transformers 5.x compatibility ---
try:
    from transformers.utils import auto_docstring
except ImportError:
    auto_docstring = lambda *a, **k: (lambda f: f)
try:
    from huggingface_hub.dataclasses import strict
except ImportError:
    try:
        from dataclasses import dataclass as strict
    except:
        strict = lambda *a, **k: (lambda f: f)
# --- End patch ---
"""
    if "# --- Patch for transformers 5.x" not in code:
        code = code.replace("import inspect\n", "import inspect\n" + patch, 1)
        with open(path, "w") as f:
            f.write(code)
        print(f"Patched {path}")
    else:
        print("Already patched")
PATCH

# Launch the shell
CMD ["/bin/bash"]
