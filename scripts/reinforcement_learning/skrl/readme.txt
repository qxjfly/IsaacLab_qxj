python export.py  将agent_xxxx.pt转换为torchscript格式的pt文件


python test.py   --model /home/ma/Learning/IsaacLab/IsaacLab_qxj/logs/skrl/humanoid_amp_walk/2025-05-26_13-28-27_amp_torch/checkpoints/exported/torchscript_gaussian_model.pt   --input_shape 1 49   --device cpu   --iters 500
为测试导出pt文件是否与skrl.play输出一致
