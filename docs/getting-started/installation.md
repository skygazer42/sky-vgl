# 安装指南

## 系统要求

| 依赖 | 最低版本 |
|------|----------|
| Python | 3.10+ |
| PyTorch | 2.4+ |

VGL 以纯 Python 实现，无需编译。只要满足 PyTorch 依赖即可安装。

## Environment Snapshot / English Summary

The CI matrix currently exercises Python 3.10–3.12 with PyTorch 2.4+ (CPU + CUDA builds), plus nightly smoke coverage for optional extras. For DGL/PyG interop installs, run `python scripts/interop_smoke.py --backend dgl` or `python scripts/interop_smoke.py --backend pyg` for the extras you installed; use `--backend all` only when both are present. Follow the [support matrix](../support-matrix.md) to align your local setup with a verified combo.

## Optional Extra Verification Matrix

| Extra | Install | Local verification | CI coverage |
|------|----------|--------------------|-------------|
| `networkx` | `pip install "sky-vgl[networkx]"` | `python scripts/extras_smoke.py --extras networkx` | main `ci.yml` extras smoke |
| `scipy` | `pip install "sky-vgl[scipy]"` | `python scripts/extras_smoke.py --extras scipy` | main `ci.yml` extras smoke |
| `tensorboard` | `pip install "sky-vgl[tensorboard]"` | `python scripts/extras_smoke.py --extras tensorboard` | main `ci.yml` extras smoke |
| `pyg` | `pip install "sky-vgl[pyg]"` | `python scripts/interop_smoke.py --backend pyg` | nightly/manual `interop-smoke.yml` |
| `dgl` | `pip install "sky-vgl[dgl]"` | `python scripts/interop_smoke.py --backend dgl` | nightly/manual `interop-smoke.yml` |
| `full` | `pip install "sky-vgl[full]"` | `python scripts/extras_smoke.py --extras networkx scipy tensorboard` and `python scripts/interop_smoke.py --backend all` | combined CI + interop smoke |

When a DGL or PyG smoke run fails because the backend is missing, `scripts/interop_smoke.py` now echoes the matching extra install command in the error message.

## 安装

### 基础安装

```bash
pip install sky-vgl
```

基础安装包含核心图学习功能，依赖 `torch>=2.4` 和 `typing_extensions>=4.12`。

### 可选依赖

根据需要安装额外的集成功能：

=== "NetworkX"

    ```bash
    pip install "sky-vgl[networkx]"
    ```

    启用 NetworkX 图格式的双向转换。

=== "SciPy"

    ```bash
    pip install "sky-vgl[scipy]"
    ```

    启用 SciPy 稀疏矩阵导出。

=== "TensorBoard"

    ```bash
    pip install "sky-vgl[tensorboard]"
    ```

    启用 TensorBoard 训练日志记录。

=== "DGL"

    ```bash
    pip install "sky-vgl[dgl]"
    ```

    启用与 DGL 框架的双向互操作。

=== "PyG"

    ```bash
    pip install "sky-vgl[pyg]"
    ```

    启用与 PyTorch Geometric 的双向互操作。

=== "全部安装"

    ```bash
    pip install "sky-vgl[full]"
    ```

    安装全部可选依赖（NetworkX + SciPy + TensorBoard + DGL + PyG）。

## PyTorch 安装

VGL 需要 PyTorch 2.4 或更高版本。请先按照 [PyTorch 官方指南](https://pytorch.org/get-started/locally/) 安装适合你硬件的 PyTorch 版本：

=== "CPU"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

=== "CUDA 11.8"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    ```

=== "CUDA 12.1"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

## 验证安装

```python
import vgl
print(vgl.__version__)  # installed release version

from vgl.graph import Graph
import torch

g = Graph.homo(
    edge_index=torch.tensor([[0, 1], [1, 0]]),
    x=torch.randn(2, 4),
)
print(g)  # 验证 Graph 对象正常创建
```

## 开发安装

如果你想贡献代码或使用最新开发版本：

```bash
git clone https://github.com/skygazer42/sky-vgl.git
cd sky-vgl
pip install -e ".[dev]"
```

开发依赖包括 `pytest`、`ruff`、`mypy` 等工具。

## 下一步

安装完成后，前往 [5 分钟快速入门](quickstart.md) 开始你的第一个图学习任务。
