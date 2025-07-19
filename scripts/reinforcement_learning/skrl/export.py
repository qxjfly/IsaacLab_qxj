import torch
import torch.nn as nn
import numpy as np
import gymnasium
from gymnasium import spaces
from skrl.models.torch import Model, GaussianMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa
obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(110,))  # 根据实际维度修改
act_space = spaces.Box(low=-10000, high=10000, shape=(26,))  # 替换实际动作维度
obs_dim = 110
act_dim = 26
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
        # taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net_container(states)
        return output, self.log_std_parameter, {}

class GaussianModelWrapper(nn.Module):
    def __init__(self, original_model, running_mean, running_variance, epsilon=1e-8):
        super().__init__()
        self.model = original_model
        # 注册为buffer（非参数但会被保存）
        self.register_buffer('running_mean', running_mean.float())
        self.register_buffer('running_variance', running_variance.float())
        self.epsilon = epsilon
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad_(False)
        
    def forward(self, states):
        # 1. 状态标准化（关键步骤）
        normalized_states = (states - self.running_mean) / torch.sqrt(self.running_variance + self.epsilon)
        
        # 2. 处理输入格式
        inputs = {"states": unflatten_tensorized_space(self.model.observation_space, normalized_states)}
        
        # 3. 执行原始模型计算
        mean, log_std, _ = self.model.compute(inputs)
        
        # 4. 扩展log_std维度以匹配mean
        log_std = log_std.unsqueeze(0).expand_as(mean)
        
        # 5. 合并输出
        # return torch.cat([mean, log_std], dim=-1)
        return mean
    
# 1. 初始化模型参数（必须与训练时完全一致）
config = {
    "observation_space": obs_space,  # 替换实际观测空间
    "action_space": act_space,            # 替换实际动作空间
    "device": "cpu",
    "clip_actions": False,
    "clip_log_std": True,
    "min_log_std": -20,
    "max_log_std": 2
}

# 2. 创建并初始化模型
original_model = GaussianModel(**config)

# 3. 加载预训练权重
checkpoint = torch.load("/home/ma/Learning/IsaacLab/IsaacLab_qxj/logs/skrl/humanoid_amp_walk/2025-07-17_10-34-37_amp_torch/checkpoints/agent_100000.pt", weights_only=True)

state_mean = checkpoint["state_preprocessor"]["running_mean"]
state_variance = checkpoint["state_preprocessor"]["running_variance"]
print(state_mean)
print(state_variance)

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
    dummy_input = torch.tensor([-3.7957e-02,  7.5483e-02, -3.6718e-01,  1.5321e-02,  1.2162e-02,
         5.1858e-02, -3.7815e-02, -7.8463e-02,  3.5858e-01,  2.5631e-01,
         3.2330e-02, -1.3126e-01,  6.5065e-01, -3.2693e-01,  1.2462e+00,
        -2.6834e-01, -5.2528e-01, -1.7863e-02,  2.9055e-02, -3.5598e-02,
        -3.0928e-02, -2.0227e-03,  6.6492e-03, -1.9679e-02,  2.5263e-03,
         1.2423e-02,  4.2877e+00, -3.1308e+00, -3.1936e+00,  3.3097e+00,
        -6.9311e-01,  2.3186e-01,  2.6978e-01,  4.2808e-01,  3.1282e+00,
        -1.3757e+00,  6.3472e-01, -1.9725e+00, -4.9385e+00, -1.3982e+00,
        -7.6090e-01,  1.0917e+00,  1.0514e+00, -3.0212e-01, -2.7553e+00,
         9.3309e-03, -6.1172e-02, -7.4148e-01,  7.9651e-01,  4.3868e-01,
        -6.8948e-01,  4.1615e-02,  8.2586e-01,  9.9335e-01, -8.3650e-02,
         7.9113e-02, -8.2617e-02, -3.9285e-02,  9.9581e-01,  1.2304e+00,
        -7.8120e-02,  3.1711e-01,  2.4620e-01,  9.0132e-02, -2.5432e+00,
         1.0967e-03, -2.7393e-01,  1.6434e-02,  2.5129e-02,  2.6620e-01,
         3.9641e-02, -1.2567e-01, -1.5399e-01, -6.7196e-01, -1.3325e-01,
         1.2926e-01, -7.6466e-01,  1.2926e-01, -7.6466e-01, -7.6466e-01,
         -7.8120e-02,  3.1711e-01,  2.4620e-01,  9.0132e-02, -2.5432e+00,
         1.0967e-03, -2.7393e-01,  1.6434e-02,  2.5129e-02,  2.6620e-01,
         3.9641e-02, -1.2567e-01, -1.5399e-01, -6.7196e-01, -1.3325e-01,
         1.2926e-01, -7.6466e-01,  1.2926e-01, -7.6466e-01, -7.6466e-01,
         -7.8120e-02,  3.1711e-01,  2.4620e-01,  9.0132e-02, -2.5432e+00,
         1.0967e-03, -2.7393e-01,  1.6434e-02,  2.5129e-02,  2.6620e-01]).to(config["device"])
    input_states = (dummy_input - state_mean.float().to(config["device"])) / torch.sqrt(state_variance.float().to(config["device"]) + 1e-8)
    # _ = original_model.compute({"states": dummy_input})
    # [ 0.1895, -0.2297, -0.3467, -0.1349,  0.0215, -0.3763,  0.3311,  0.3805,
    #  -0.3586,  0.0353,  0.4035, -0.0625, -0.3288,  0.2595, -0.0823,  0.3742,
    #  -0.0437, -0.0059, -0.1628, -0.0072, -0.1121, -0.0183,  0.0077, -0.0015,
    #  -0.0083, -0.0064 ]
    _ = original_model.compute({"states": input_states})

    actions, log_prob, outputsqxj = original_model.act({"states": dummy_input},role="")

    print("out",_)
    print("meanaction",outputsqxj)



# 5. 创建包装器并转换
# 创建包装器（集成标准化）
wrapped_model = GaussianModelWrapper(
    original_model=original_model.to(config["device"]),
    running_mean=state_mean.to(config["device"]),
    running_variance=state_variance.to(config["device"])
)
example_input = torch.randn(1, obs_dim).to(config["device"])

# 转换方法选择
traced_model = torch.jit.trace(wrapped_model, example_input)

# 6. 保存TorchScript模型
traced_model.save("/home/ma/Learning/IsaacLab/IsaacLab_qxj/logs/skrl/humanoid_amp_walk/2025-07-17_10-34-37_amp_torch/checkpoints/exported/policy_amp_071702.pt")
# print("test")
print("save: ",traced_model.forward(dummy_input))


# # 加载检查点
# checkpoint = torch.load("policy_checkpoint.pth", map_location="cpu")

# # 初始化模型
# policy = CustomPolicy(env.observation_space, env.action_space, device="cpu")
# policy.load_state_dict(checkpoint["policy_state_dict"])

# # 恢复预处理器参数
# policy.state_preprocessor.running_mean = checkpoint["running_mean"]
# policy.state_preprocessor.running_var = checkpoint["running_var"]

# # 使用预处理后的状态生成动作
# raw_states = env.get_states()
# processed_states = policy.state_preprocessor(raw_states)
# actions, _, _ = policy.act({"states": processed_states})

# self.epsilon = 1e-8
# (x - self.running_mean.float()) / (torch.sqrt(self.running_variance.float()) + self.epsilon)