import torch
import torch.nn as nn
import numpy as np
import gymnasium
from gymnasium import spaces
from skrl.models.torch import Model, GaussianMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa
obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(49,))  # 根据实际维度修改
act_space = spaces.Box(low=-10000, high=10000, shape=(12,))  # 替换实际动作维度
obs_dim = 49
act_dim = 12
# 定义与原始模型完全一致的网络结构
class GaussianModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions,
                    clip_log_std, min_log_std, max_log_std, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net_container = nn.Sequential(
            # nn.LazyLinear(out_features=1024).to(device),
            nn.Linear(in_features=observation_space.shape[0], out_features=1024, bias=True).to(device),
            nn.ReLU(),
            # nn.LazyLinear(out_features=512).to(device),
            nn.Linear(in_features=1024, out_features=512, bias=True).to(device),
            nn.ReLU(),
            # nn.LazyLinear(out_features=self.num_actions).to(device),
            nn.Linear(in_features=512, out_features=action_space.shape[0], bias=True).to(device),
        )
        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), fill_value=-2.9,device=device), requires_grad=False)

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        # states = inputs.get("states")
        # print('states', states)
        # taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net_container(states)
        return output, self.log_std_parameter, {}

class GaussianModelWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad_(False)
        
    def forward(self, states):
        # 处理输入格式（假设states已经是展开后的张量）
        # 如果原始需要unflatten操作，这里需要具体实现
        # inputs = {"states": states}
        inputs = {"states": unflatten_tensorized_space(self.model.observation_space, states)}
        # 执行计算
        mean, log_std, _ = self.model.compute(inputs)
        # 扩展log_std维度以匹配mean
        log_std = log_std.unsqueeze(0)  # 从[action_dim]变为[1, action_dim]
        log_std = log_std.expand_as(mean)  # 扩展为与mean相同形状
        # 合并输出
        return torch.cat([mean, log_std], dim=-1)
    
# 1. 初始化模型参数（必须与训练时完全一致）
config = {
    "observation_space": obs_space,  # 替换实际观测空间
    "action_space": act_space,            # 替换实际动作空间
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "clip_actions": False,
    "clip_log_std": True,
    "min_log_std": -20,
    "max_log_std": 2
}

# 2. 创建并初始化模型
original_model = GaussianModel(**config)

# 3. 加载预训练权重
checkpoint = torch.load("/home/ma/Learning/IsaacLab/IsaacLab_qxj/logs/skrl/humanoid_amp_walk/2025-05-26_13-28-27_amp_torch/checkpoints/agent_40000.pt", weights_only=True)

# print(checkpoint)
# 5. 提取策略模型参数（根据实际检查点结构调整）
policy_state_dict = {
    k.replace("policy.", ""): v 
    for k, v in checkpoint["policy"].items()
}
# policy_state_dict = checkpoint["policy"]
# 验证参数匹配
missing_keys, unexpected_keys = original_model.load_state_dict(policy_state_dict, strict=False)
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")

original_model.eval()

# 4. 初始化LazyLinear层（关键步骤！）
with torch.no_grad():
    # dummy_input = torch.randn(1, obs_dim).to(config["device"])
    dummy_input = torch.tensor([-0.3932, -0.1583, -0.0114, -0.0112,  0.1535,  0.0943,  0.9568,  0.7750,
        -0.1410, -0.4991, -0.0165, -0.0541, -1.7885,  2.7503,  0.7256, -0.0302,
        -0.2077, -1.7453, -3.5475, -2.3353, -0.8671, -0.1158, -0.0249, -1.4260,
         0.8392,  0.9933, -0.0456,  0.1062, -0.1101, -0.0912,  0.9897,  1.1892,
         0.0754,  0.0676, -0.4977,  0.0942,  2.1875,  0.1055, -0.1309, -0.4194,
         0.2081,  0.1843, -0.3518, -0.0932, -0.1188, -0.7778,  0.0300,  0.1710,
        -0.7209]).to(config["device"])
    _ = original_model.compute({"states": dummy_input})

    actions, log_prob, outputsqxj = original_model.act({"states": dummy_input},role="")

    print("out",_)
    print("meanaction",outputsqxj)



# 5. 创建包装器并转换
wrapped_model = GaussianModelWrapper(original_model)
example_input = torch.randn(1, obs_dim).to(config["device"])

# 转换方法选择
traced_model = torch.jit.trace(wrapped_model, example_input)

# 6. 保存TorchScript模型
traced_model.save("/home/ma/Learning/IsaacLab/IsaacLab_qxj/logs/skrl/humanoid_amp_walk/2025-05-26_13-28-27_amp_torch/checkpoints/exported/torchscript_gaussian_model.pt")
# print("test")
print("save: ",traced_model.forward(dummy_input))