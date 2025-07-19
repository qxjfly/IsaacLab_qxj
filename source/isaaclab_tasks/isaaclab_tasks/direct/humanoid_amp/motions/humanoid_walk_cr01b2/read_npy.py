import numpy as np
import os

def print_npy_file(file_path, max_rows=30, max_cols=30, print_full=False):
    """
    读取并打印 .npy 文件的内容
    
    参数:
        file_path (str): .npy 文件的路径
        max_rows (int): 最大显示行数（默认10）
        max_cols (int): 最大显示列数（默认10）
        print_full (bool): 是否打印完整内容（默认False）
    
    返回:
        None
    """
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误：文件 '{file_path}' 不存在")
            return
        
        # 检查文件扩展名
        if not file_path.lower().endswith('.npy'):
            print(f"警告：文件 '{file_path}' 不是 .npy 文件")
        
        # 加载 .npy 文件
        data = np.load(file_path, allow_pickle=True)
        
        # 打印基本信息
        print("\n" + "="*50)
        print(f"文件: {file_path}")
        print(f"数组形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"元素总数: {data.size}")
        print(f"维度数: {data.ndim}")
        print("="*50 + "\n")
        
        # 处理不同维度的数组
        if data.ndim == 0:
            # 标量
            print("标量值:")
            print(data.item())
        
        elif data.ndim == 1:
            # 一维数组
            print("一维数组内容:")
            if print_full or len(data) <= max_rows:
                print(data)
            else:
                print(f"前 {max_rows} 个元素:")
                print(data[:max_rows])
                print(f"... (共 {len(data)} 个元素)")
        
        elif data.ndim == 2:
            # 二维数组
            print("二维数组内容:")
            if print_full or (data.shape[0] <= max_rows and data.shape[1] <= max_cols):
                print(data)
            else:
                # 打印部分内容
                print(f"显示前 {min(max_rows, data.shape[0])} 行, 前 {min(max_cols, data.shape[1])} 列:")
                with np.printoptions(threshold=np.inf, linewidth=200):
                    for i in range(min(max_rows, data.shape[0])):
                        row = data[i]
                        if len(row) > max_cols:
                            print(f"行 {i}: {row[:max_cols]} ...")
                        else:
                            print(f"行 {i}: {row}")
                
                print(f"... (共 {data.shape[0]} 行, {data.shape[1]} 列)")
        
        elif data.ndim >= 3:
            # 高维数组
            print(f"{data.ndim}维数组内容:")
            print("显示前几个切片:")
            
            # 显示前几个切片
            for i in range(min(3, data.shape[0])):
                print(f"\n切片 {i}:")
                slice_data = data[i]
                
                if slice_data.ndim == 1:
                    if len(slice_data) > max_cols:
                        print(slice_data[:max_cols], "...")
                    else:
                        print(slice_data)
                
                elif slice_data.ndim == 2:
                    if slice_data.shape[0] > max_rows or slice_data.shape[1] > max_cols:
                        print(f"显示前 {min(max_rows, slice_data.shape[0])} 行, 前 {min(max_cols, slice_data.shape[1])} 列:")
                        for j in range(min(max_rows, slice_data.shape[0])):
                            row = slice_data[j]
                            if len(row) > max_cols:
                                print(row[:max_cols], "...")
                            else:
                                print(row)
                        print(f"... (共 {slice_data.shape[0]} 行, {slice_data.shape[1]} 列)")
                    else:
                        print(slice_data)
                
                else:
                    print(f"形状: {slice_data.shape}")
            
            print(f"\n... (共 {data.shape[0]} 个切片)")
        
        # 打印数组统计信息
        if data.size > 0 and np.issubdtype(data.dtype, np.number):
            print("\n统计信息:")
            print(f"  最小值: {np.min(data)}")
            print(f"  最大值: {np.max(data)}")
            print(f"  平均值: {np.mean(data)}")
            print(f"  标准差: {np.std(data)}")
    
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    
    # 测试函数
    print("="*50)
    print("测试一维数组")
    print("="*50)
    print_npy_file("body_names.npy")
    
    print("\n" + "="*50)
    print("测试二维数组")
    print("="*50)
    print_npy_file("body_names.npy", max_rows=5, max_cols=5)
    
    print("\n" + "="*50)
    print("测试三维数组")
    print("="*50)
    print_npy_file("body_names.npy")
    
