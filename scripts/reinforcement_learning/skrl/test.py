import torch
import time
import argparse
import numpy as np
from pathlib import Path

def test_torchscript_model(model_path, input_shape=(1,3,224,224), device="cpu", 
                          test_iters=100, rtol=1e-5, atol=1e-8):
    """
    综合测试TorchScript模型的工具函数
    参数：
        model_path: .pt文件路径
        input_shape: 测试输入形状，默认(1,3,224,224)
        device: 运行设备，支持cpu/cuda
        test_iters: 性能测试迭代次数
        rtol/atol: 输出验证容忍度
    """
    try:
        # 加载TorchScript模型[6](@ref)
        ts_model = torch.jit.load(model_path, map_location=device)
        ts_model.eval()
        print(f"✅ 成功加载TorchScript模型：{Path(model_path).name}")
        
        # 输入数据生成
        # input_tensor = torch.randn(input_shape).to(device)
        input_tensor = torch.tensor(
        [-0.3133,  0.0474,  0.0357, -0.0465,  0.1358, -0.3102,  0.7947,  1.3074,
        -0.4235,  0.2052,  0.0311, -0.0176,  3.2304, -2.6063,  0.3000, -0.7626,
         2.0400,  1.4007, -2.2374,  1.9474, -1.2490, -0.1005,  0.1889, -0.0183,
         0.8296,  0.9835,  0.1714,  0.0571, -0.0600,  0.0116,  0.9981,  1.2591,
         0.0855,  0.2676,  0.3049, -0.0312, -1.4769,  0.0365, -0.1870, -0.4155,
         0.1210,  0.1918, -0.3950, -0.3530, -0.1386, -0.5340, -0.0357,  0.1257,
        -0.7681],
        dtype=torch.float32  # 显式指定数据类型（可选）
        ).to(device)
        # input_tensor = {"states": input_tensor1.to(device)}
        # 基础信息检查[7](@ref)
        print("\n=== 模型基础信息 ===")
        print(f"输入形状要求: {input_shape}")
        print(f"模型结构摘要:\n{ts_model.code if hasattr(ts_model, 'code') else '无可用代码表示'}")

        # 前向推理测试
        # 输出类型验证
        with torch.no_grad():
            output = ts_model(input_tensor)
            
            if isinstance(output, tuple):
                print(f"输出元组包含 {len(output)} 个张量:")
                for i, tensor in enumerate(output):
                    print(f"  输出{i}形状: {tensor.shape} | 数据类型: {tensor.dtype}")
                    print(tensor)
            else:
                print(f"输出形状: {output.shape} | 数据类型: {output.dtype}")
                print(output[0,0:12])

        # 动态控制流检测（如果存在）[3](@ref)
        if "prim::Loop" in ts_model.graph.str() or "prim::If" in ts_model.graph.str():
            print("\n⚠️ 检测到动态控制流（循环/条件分支），建议进行多场景测试")

        # 性能基准测试[7](@ref)
        print(f"\n=== 性能测试 ({device}, {test_iters}次迭代) ===")
        timings = []
        with torch.no_grad():
            for _ in range(10):  # 预热
                _ = ts_model(input_tensor)
            
            start_time = time.time()
            for _ in range(test_iters):
                start_iter = time.time()
                _ = ts_model(input_tensor)
                timings.append(time.time() - start_iter)
        
        avg_time = np.mean(timings)*1000
        fps = 1/(avg_time/1000)
        print(f"平均推理时间: {avg_time:.2f}ms | 吞吐量: {fps:.1f} FPS")
        print(f"总耗时: {time.time()-start_time:.2f}s (含{test_iters}次推理)")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TorchScript模型测试工具')
    parser.add_argument('--model', required=True, help='TorchScript模型路径 (.pt)')
    parser.add_argument('--input_shape', type=int, nargs='+', 
                       default=[1,3,224,224], help='输入形状，例如 1 3 224 224')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='运行设备')
    parser.add_argument('--iters', type=int, default=100,
                       help='性能测试迭代次数')
    args = parser.parse_args()

    test_torchscript_model(
        model_path=args.model,
        input_shape=tuple(args.input_shape),
        device=args.device,
        test_iters=args.iters
    )