"""
LLM QLoRA 4bit量化示范程序
这个程序展示了如何使用bitsandbytes库的QLoRA技术将大语言模型压缩为4bit精度
进一步减少内存占用，同时通过Double Quant和NF4量化类型保持较好的模型性能
"""

# 1. 配置环境（必须在导入transformers库之前设置，否则不生效）
# 导入操作系统模块，用于设置环境变量
import os
# 设置Hugging Face国内镜像地址，这样下载模型会更快
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 关闭bitsandbytes库的欢迎信息，让输出更简洁
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

# 2. 导入所需的Python库
# 导入PyTorch，这是一个深度学习框架，用于模型的训练和推理
import torch
# 从transformers库导入自动加载模型和分词器的工具
# AutoModelForCausalLM用于加载生成式语言模型
# AutoTokenizer用于将文本转换为模型可理解的数字格式
from transformers import AutoModelForCausalLM, AutoTokenizer
# 从transformers库导入量化配置工具，用于设置4bit量化参数
from transformers import BitsAndBytesConfig


def main():
    """主函数：演示如何使用QLoRA将语言模型量化为4bit并进行文本生成"""

    # 第一步：配置QLoRA 4bit量化参数
    # 创建量化配置对象，设置QLoRA特有的4bit量化参数
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 关键参数：开启4bit量化，比8bit更节省内存
        bnb_4bit_use_double_quant=True,  # 启用Double Quant技术：对量化后的权重再进行一次量化
                                         # 可以进一步减少内存占用并提高量化精度
        bnb_4bit_quant_type="nf4",  # 使用NF4（Normalized Float 4bit）量化类型
                                     # 这是专门为语言模型权重设计的量化类型，能更好地保留模型性能
        bnb_4bit_compute_dtype=torch.bfloat16  # 计算时使用bfloat16数据类型
                                               # 平衡计算效率和数值精度
    )

    # 第二步：选择要使用的模型
    # 使用一个轻量级的模型TinyLlama进行演示，参数只有11亿
    # 这个模型足够小，可以在大多数GPU上运行，同时能展示量化效果
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # 第三步：加载量化后的模型
    print("\n===== 🚀 开始加载QLoRA 4bit量化模型 =====")
    # 从Hugging Face下载模型，并应用我们之前设置的QLoRA量化配置
    model = AutoModelForCausalLM.from_pretrained(
        model_id,  # 指定要加载的模型ID
        device_map="auto",  # 自动将模型分配到可用的设备（优先使用GPU）
        quantization_config=bnb_config,  # 应用QLoRA 4bit量化配置
        trust_remote_code=True,  # 信任模型中的自定义代码
        torch_dtype=torch.float16,  # 模型参数使用16bit浮点数
        low_cpu_mem_usage=True,  # 减少CPU内存使用，适合大模型加载
        use_cache=True  # 使用KV缓存加速推理过程
    )

    # 第四步：查看模型加载情况
    # 打印模型加载在哪个设备上（应该是cuda:0，即第一个GPU）
    print(f"✅ 模型加载位置: {next(model.parameters()).device}")
    # 检查模型是否成功进行了4bit量化（应该显示True）
    print(f"✅ 是否4bit量化: {model.is_loaded_in_4bit}")
    # 查看当前模型占用的GPU显存大小（MB）
    print(f"✅ GPU显存占用: {torch.cuda.memory_allocated(0)/1024/1024:.0f} MB")

    # 第五步：加载分词器
    # 加载与模型配套的分词器，用于处理文本输入
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 设置padding token与结束token相同，避免在批量处理时出现警告
    tokenizer.pad_token = tokenizer.eos_token

    # 第六步：使用量化模型生成文本
    print("\n===== 📝 开始文本生成测试 =====")
    # 定义我们要问模型的问题
    prompt = "请用一句话介绍人工智能"
    # 1. 调用分词器tokenizer处理文本prompt：将自然语言转换为模型可识别的token序列
    # 2. 指定return_tensors="pt"：要求分词器返回PyTorch格式的张量，便于后续PyTorch框架下的模型计算
    # 3. 调用.to("cuda:0")：将分词器返回的所有张量（如input_ids、attention_mask）移动到编号为0的GPU设备上
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    # 使用模型生成回答
    outputs = model.generate(
        **inputs,  # 传入处理后的输入数据
        max_new_tokens=50,  # 限制生成的文本长度最多50个token
        temperature=0.7,  # 控制生成文本的随机性（0-2之间，越大越随机）
        do_sample=True,  # 启用采样模式，生成更自然的文本
        pad_token_id=tokenizer.eos_token_id,  # 设置padding token的ID
        repetition_penalty=1.1  # 惩罚重复内容，值越大越不容易重复
    )

    # 将生成的数字格式转换回人类可读的文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 打印我们的问题和模型的回答
    print(f"输入: {prompt}")
    print(f"输出: {response}")

    # 第七步：清理资源
    # 释放未使用的GPU内存，避免资源浪费
    torch.cuda.empty_cache()
    # 告诉用户程序已经运行完成
    print("\n===== ✨ 运行完成 =====")


if __name__ == "__main__":
    """程序入口：当直接运行这个脚本时，会执行main函数"""
    # 使用try-except来捕获可能出现的错误，这样程序出错时不会直接崩溃
    try:
        # 执行主函数
        main()
    except Exception as e:
        # 如果出现错误，打印错误信息
        print(f"\n❌ 程序运行出错: {e}")
        print("\n📌 请检查你的环境配置或网络连接")