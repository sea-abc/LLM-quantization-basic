# LLM 量化示例项目

这个项目展示了如何使用不同的量化技术在 GPU 上高效运行大型语言模型 (LLM)，特别是针对资源受限的环境。

## 项目结构

```
├── llm_int8_demo.py      # 8bit 量化示例代码
├── llm_QLoRA_int4.py     # 4bit 量化 (QLoRA) 示例代码
├── 量化基础.ipynb         # 量化技术基础知识和代码示例：包括对称动态/静态量化，量化感知
└── README.md             # 文档说明
```

## 量化技术介绍

### 1. 8bit 量化 (LLM.int8)
- 使用 `bitsandbytes` 库实现
- 将模型权重从 16bit/32bit 降低到 8bit
- 减少显存占用约 50%
- 适用于中等大小的模型

### 2. 4bit 量化 (QLoRA)
- 基于 LoRA (Low-Rank Adaptation) 技术
- 进一步将模型权重降低到 4bit
- 支持 NF4 (Normal Float 4) 数据类型，专为量化设计
- 启用双量化 (double quantization) 进一步减少显存占用
- 适用于更大的模型，在有限显存环境中表现优秀

## 环境配置

### 推荐使用autodl云端配置

**此环境是在autodl中租赁下创建的，使用的配置为：**
```
镜像
PyTorch  2.5.1                 #  这三项务必一致
Python  3.12(ubuntu22.04)      #  这三项务必一致
CUDA  12.4                     #  这三项务必一致
GPU
RTX 3090(24GB) * 1
CPU
14 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
```

下载此仓库
```bash
# 克隆到临时文件夹（避免目录冲突）
git clone https://github.com/sea-abc/LLM-quantization-basic.git temp_repo

# 将临时文件夹中的内容移动到当前目录
mv temp_repo/* .

# 删除临时文件夹（可选）
rm -rf temp_repo
```


1. 克隆相应的环境过来

```bash
mkdir -p /root/autodl-tmp/quantization
conda create --clone /root/miniconda3 -p /root/autodl-tmp/quantization --yes

conda init
source activate /root/autodl-tmp/quantization
```

2. 安装核心依赖：

```bash
pip install accelerate==0.34.2
pip install \
transformers==4.52.1 \
bitsandbytes==0.43.3 \
huggingface-hub==0.30.1 \
tokenizers==0.21.4 \
ipykernel==6.29.5 \
-i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

3. 如果想在jupyter notebook中使用上述环境，需要安装对应的ipykernel
```bash
python -m ipykernel install --user --name=quantization --display-name "Python (quantization)"
```
之后打开一个ipynb文件后，右上角切换内核至 Python (quantization) 即可。

## 运行示例

### 运行 8bit 量化示例

```bash
python llm_int8_demo.py
```

### 运行 4bit 量化 (QLoRA) 示例

```bash
python llm_QLoRA_int4.py
```

## 代码说明

### 量化基础教程 (`量化基础.ipynb`)

这个 Jupyter Notebook 提供了量化技术的基础知识和实践示例：

- 对称动态/静态量化、量化感知的代码实现
- 量化前后参数变化的直观展示
- 推理性能对比
- 详细的注释和解释，适合初学者学习量化技术原理

### 8bit 量化示例 (`llm_int8_demo.py`)

该脚本演示了如何使用 `bitsandbytes` 库对 LLM 进行 8bit 量化：

1. 设置环境变量以优化 CUDA 性能
2. 配置量化参数
3. 加载量化后的模型
4. 进行文本生成测试

### 4bit 量化示例 (`llm_QLoRA_int4.py`)

该脚本演示了如何使用 QLoRA 技术对 LLM 进行 4bit 量化：

1. 设置环境变量以优化 CUDA 性能
2. 配置 QLoRA 量化参数：
   - `load_in_4bit=True`：启用 4bit 量化
   - `bnb_4bit_use_double_quant=True`：启用双量化
   - `bnb_4bit_quant_type="nf4"`：使用 NF4 数据类型
   - `bnb_4bit_compute_dtype=torch.bfloat16`：计算使用 bfloat16
3. 加载量化后的模型
4. 进行文本生成测试


## 常见问题

### 1. 模型加载失败

- 确保网络连接正常（首次运行需要下载模型）
- 检查 CUDA 版本是否与 PyTorch 兼容
- 确认 `bitsandbytes` 库已正确安装

### 2. 生成速度慢

- 确保模型在 GPU 上运行 (`device_map="auto"`)
- 尝试使用更快的 CUDA 架构
- 减少生成的最大长度

## 参考资料

- 哔哩哔哩UP主：RethinkFun量化系列课程
