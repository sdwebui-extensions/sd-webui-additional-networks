'''
copied from lora
'''
import torch

from scripts import cp_lyco_helpers
from scripts import cp_network
from modules import devices


class ModuleTypeLora(cp_network.ModuleType):
    def create_module(self, net: cp_network.Network, weights: cp_network.NetworkWeights):
        if all(x in weights.w for x in ["lora_up.weight", "lora_down.weight"]):
            return NetworkModuleLora(net, weights)

        return None


class NetworkModuleLora(cp_network.NetworkModule):
    def __init__(self,  net: cp_network.Network, weights: cp_network.NetworkWeights):
        super().__init__(net, weights)

        self.up_model = self.create_module(weights.w, "lora_up.weight")
        self.down_model = self.create_module(weights.w, "lora_down.weight")
        self.mid_model = self.create_module(weights.w, "lora_mid.weight", none_ok=True)

        self.dim = weights.w["lora_down.weight"].shape[0]
        self.mask_area = None
        self.mask = None
        self.mask_dic = None

    def create_module(self, weights, key, none_ok=False):
        weight = weights.get(key)

        if weight is None and none_ok:
            return None

        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]

        if is_linear:
            weight = weight.reshape(weight.shape[0], -1)
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif is_conv and key == "lora_down.weight" or key == "dyn_up":
            if len(weight.shape) == 2:
                weight = weight.reshape(weight.shape[0], -1, 1, 1)

            if weight.shape[2] != 1 or weight.shape[3] != 1:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
            else:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif is_conv and key == "lora_mid.weight":
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
        elif is_conv and key == "lora_up.weight" or key == "dyn_down":
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            raise AssertionError(f'Lora layer {self.network_key} matched a layer with unsupported type: {type(self.sd_module).__name__}')

        with torch.no_grad():
            if weight.shape != module.weight.shape:
                weight = weight.reshape(module.weight.shape)
            module.weight.copy_(weight)

        module.to(device=devices.cpu, dtype=devices.dtype)
        module.weight.requires_grad_(False)

        return module

    def calc_updown(self, orig_weight, input=None):
        up = self.up_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        down = self.down_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [up.size(0), down.size(1)]
        if self.mask_dic:
            breakpoint()
        if self.mid_model is not None:
            # cp-decomposition
            mid = self.mid_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = cp_lyco_helpers.rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = cp_lyco_helpers.rebuild_conventional(up, down, output_shape, self.network.dyn_dim)

        # # FIXME(zhiying.xzy): ref lora_compvis.py for mask, output_shape here is actually weight_shape
        # if self.mask_dic:
        #     self.up_model.to(up.device)
        #     self.down_model.to(down.device)
        #     lx = self.up_model(self.down_model(input)) # lora activations

        #     if len(lx.size()) == 4:  # b,c,h,w
        #         area = lx.size()[2] * lx.size()[3] # N C 64 64, 4096
        #     else:
        #         area = lx.size()[1]  # b,seq,dim

        #     if self.mask is None or self.mask_area != area:
        #         # get mask
        #         # print(self.lora_name, x.size(), lx.size(), area)
        #         mask = self.mask_dic[area] # 4096, 1024; mask: 64 * 64; lORA A @ LORA B, OIhw
        #         if len(lx.size()) == 3:
        #             mask = torch.reshape(mask, (1, -1, 1))
        #         self.mask = mask
        #         self.mask_area = area
        # else:
        #     self.mask = 1

        orig_updown = self.finalize_updown(updown, orig_weight, output_shape)
        # return orig_updown * self.mask # wrong
        return orig_updown

    def forward(self, x, y):
        self.up_model.to(device=devices.device)
        self.down_model.to(device=devices.device)

        return y + self.up_model(self.down_model(x)) * self.multiplier() * self.calc_scale()

    def set_mask_dic(self, mask_dic):
        # called before every generation

        # check this module is related to h,w (not context and time emb)
        module_name = self.network_key # network_key is lora name, sd_key is the main branch module name
        if "attn2_to_k" in module_name or "attn2_to_v" in module_name or "emb_layers" in module_name:
            # print(f"LoRA for context or time emb: {self.lora_name}")
            self.mask_dic = None
        else:
            self.mask_dic = mask_dic

        self.mask = None


