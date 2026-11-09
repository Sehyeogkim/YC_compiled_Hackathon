# Native GPU setup guide

This guide is for the canonical `control-ui/` native GPU app. For the current presentation UI at `5173/?local=1`, see [the local presentation guide](LOCAL_DEMO.md).

Run all commands from the repository root unless a step says otherwise.

## Prerequisites

| Where | Required setup | Purpose |
| --- | --- | --- |
| Local machine | Node.js 20.19+ or 22.12+, npm, a WebGL2-capable browser, and an SSH client | Run the operator interface and connect to the GPU |
| RunPod GPU Pod | Linux, NVIDIA GPU, CUDA-compatible driver, and EGL/OpenGL libraries | Run SmolVLA inference and native LIBERO simulation |
| GPU Python environment | Python 3.12, the pinned LeRobot/LIBERO runtime, Git, and `uv` for the clean setup script | Install the model and simulator dependencies |
| Downloads and storage | Network access to GitHub and Hugging Face; storage under `/workspace` for models and recordings | Fetch pinned assets and retain episode data |

The project was verified on **RunPod with an NVIDIA A40**. A GPU is needed for the live inference path even when you only want to run the pretrained model. Fine-tuning is a separate, optional workflow. Other GPU configurations require their own validation; the A40 is the tested configuration rather than a measured minimum requirement.

See [`prerequisites.txt`](../prerequisites.txt) for the complete setup checklist. [`requirements.txt`](../requirements.txt) installs the API packages into an **already prepared native runtime**; the full framework and GPU dependencies are installed by the pinned runtime setup below. Browser dependencies are managed separately in `control-ui/package.json`.

## Quick start

### Browser interface

Requires Node.js **20.19+ or 22.12+** and npm. From the repository root:

```sh
cd control-ui
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

For a GPU-free preview, expand **Connection settings** and choose **Offline demo**. This mode uses a local scripted conveyor scene. Included factory assets load without API credentials.

To build and preview the production bundle:

```sh
npm run build
npm run preview
```

### Native GPU mode on RunPod

Prepare a RunPod GPU Pod with SSH access and the prerequisites above, then place the repository under `/workspace`. For a first-time clean runtime setup, run these commands from the repository root **on the Pod**:

```sh
bash training/runtime/setup_pinned.sh /workspace/werescue-runtime
source /workspace/werescue-runtime/env.sh
export WERESCUE_POLICY_PATH=/workspace/werescue-runtime/models/P
```

The setup requires `uv`, Git, and working NVIDIA/EGL libraries. `WERESCUE_POLICY_PATH` reuses the pinned model downloaded by the setup. If your Pod already has a compatible pinned runtime, activate it instead and use its pinned model path; see the [native runtime guide](../training/runtime/README.md).

Check GPU access with `nvidia-smi`. In the activated Python environment, `python -c "import torch; print(torch.cuda.is_available())"` should print `True` before starting the server.

Install the API packages and start the inference service from the repository root on the Pod:

```sh
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export WERESCUE_DEVICE=cuda
export WERESCUE_HORIZON=1000
python -m inference.server
```

The interactive horizon is 1,000 native steps; benchmark evaluation uses a separate protocol. See the [inference guide](../inference/README.md) for exact model pins and runtime configuration.

Forward the API to your local machine in another terminal, replacing `GPU_HOST` with your SSH host or configured alias:

```sh
ssh -N -L 8000:127.0.0.1:8000 GPU_HOST
```

The Vite proxy forwards `/robot-api` to `127.0.0.1:8000`. In the browser, select **GPU · actual VLA**, leave the endpoint as `/robot-api`, and click **Connect GPU**, then **Start VLA**.

With the SSH tunnel running, `http://127.0.0.1:8000/health` reports loading and readiness. `ready=true` means the model is loaded; `inference_verified=true` is set after an actual policy step.


[Back to RG Service Center](../README.md)
