from torch import nn
import torch


class CasualLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden,filter_size, n_layer=1):
        super(CasualLSTMCell, self).__init__()
        # 设置卷积核大小
        filter_size = filter_size
        # 步幅
        stride = 1
        # 隐藏层单元数量
        self.num_hidden = num_hidden
        # 填充大小，确保卷积操作的输出尺寸不变
        self.padding = filter_size // 2

        self._forget_bias = 1.0
        self.cur_states = [None for _ in range(n_layer)]
        self.cell_state = [None for _ in range(n_layer)]
        self.memory_state = None
        self.n_layer = n_layer
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            # 确定当前层的输入通道数
            if i == 0:
                in_channel = in_channel  # 第一层使用外部传入的输入通道数
            else:
                in_channel = num_hidden  # 后续层使用前一层的输出通道数
            conv_gates = nn.ModuleDict({
                'x': nn.Sequential(
                    nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, padding=self.padding,
                              bias=False),
                   ),
                'h': nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,  padding=self.padding,
                              bias=False),
                    ),
                'm': nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,  padding=self.padding,
                              bias=False),
                    ),
                'c': nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, padding=self.padding,
                              bias=False),
                    ),
                'c2m': nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, padding=self.padding,
                              bias=False),
                    ),
                'om': nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, padding=self.padding,
                              bias=False),
                ),


                'o': nn.Sequential(
                    nn.Conv2d(num_hidden , num_hidden, kernel_size=filter_size, padding=self.padding,
                              bias=False),
                    )
            })
            self.layers.append(conv_gates)
        self.conv_last = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=num_hidden * 2 ,
                    out_channels=num_hidden,  # for candidate neural memory
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                )
                for _ in range(n_layer)
            ]
        )

    def init_hidden(self, batch_shape, device):
        b, _, h, w = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i]=torch.zeros((b, self.num_hidden, h, w), device=device)
            self.cell_state[i]=torch.zeros((b, self.num_hidden, h, w), device=device)
        self.memory_state = torch.zeros((b, self.num_hidden, h, w), device=device)

    def step_forward(self,input_tensor, index):
        conv_gates = self.layers[index]
        h_cur = self.cur_states[index]
        m_cur = self.memory_state
        c_cur = self.cell_state[index]
        assert h_cur is not None
        # 对输入、隐藏状态、记忆和细胞状态进行卷积和层归一化
        x_concat = conv_gates["x"](input_tensor)         # 输入的卷积结果
        h_concat = conv_gates["h"](h_cur)         # 隐藏状态的卷积结果
        m_concat = conv_gates["m"](m_cur)         # 记忆的卷积结果
        c_concat = conv_gates["c"](c_cur)         # 细胞状态的卷积结果
        # 将卷积结果按照数量划分成不同的部分
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1        # 将x_concat划分为i,f,g,i',f',g',o
        )
        i_h, f_h, g_h ,o_h= torch.split(h_concat, self.num_hidden, dim=1)   # 隐藏状态划分
        i_m_prime, f_m_prime, m_m = torch.split(
            m_concat, self.num_hidden, dim=1        # 记忆划分
        )
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)       # 细胞状态划分
        # 计算输入门、遗忘门和候选记忆
        i_t = torch.sigmoid(i_x + i_h + i_c)        # 输入门
        f_t = torch.sigmoid(f_x + f_h + f_c+self._forget_bias)        # 遗忘门
        g_t = torch.tanh(g_x + g_h + g_c)           # 候选记忆
        # 更新细胞状态
        self.cell_state[index] = f_t * c_cur + i_t * g_t       # 细胞状态更新公式
        # 对新的细胞状态进行卷积和层归一化
        c_concat_new = conv_gates["c2m"](self.cell_state[index])
        i_c_new, f_c_new, g_c_new,o_c = torch.split(c_concat_new, self.num_hidden, dim=1)       # 划分新细胞状态
        # 计算新的输入门、遗忘门和候选记忆（新细胞状态相关）
        i_t_prime = torch.sigmoid(i_x_prime + i_m_prime + i_c_new)      # 新输入门
        f_t_prime = torch.sigmoid(f_x_prime + f_m_prime + f_c_new+self._forget_bias)      # 新遗忘门
        g_t_prime = torch.tanh(g_x_prime + g_c_new)         # 新候选记忆
        # 更新记忆
        self.memory_state = f_t_prime * torch.tanh(m_m) + i_t_prime * g_t_prime
        o_m=conv_gates["om"](self.memory_state)
        # 计算输出
        o_t = torch.tanh(o_x + o_m+o_c+o_h)        # 输出状态
        o=conv_gates["o"](o_t)
        cell=torch.cat([self.cell_state[index], self.memory_state], dim=1)
        cell=self.conv_last[index](cell)
        h_new = o * torch.tanh(cell)   # 更新后的隐藏状态
        self.cur_states[index] = h_new
        return h_new

    def forward(self, input_tensor):
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i)
        return input_tensor
if __name__=='__main__':
    x=torch.rand(16,2,128,128).to('cuda:0')
    net=CasualLSTMCell(in_channel=2, num_hidden=64, filter_size=3).to('cuda:0')
    net.init_hidden((16,2,128,128),device='cuda:0')
    print(net(x).shape)
    print(sum([p.numel() for p in net.parameters()]))
