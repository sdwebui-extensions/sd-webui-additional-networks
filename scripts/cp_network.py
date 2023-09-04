'''
copied from lora
'''
import os
from collections import namedtuple
import enum
import torch

from modules import sd_models, cache, errors, hashes, shared

NetworkWeights = namedtuple('NetworkWeights', ['network_key', 'sd_key', 'w', 'sd_module'])

metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}


class SdVersion(enum.Enum):
    Unknown = 1
    SD1 = 2
    SD2 = 3
    SDXL = 4


class NetworkOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.metadata = {}
        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        def read_metadata():
            metadata = sd_models.read_metadata_from_safetensors(filename)
            metadata.pop('ssmd_cover_images', None)  # those are cover images, and they are too big to display in UI as text

            return metadata

        if self.is_safetensors and (not shared.cmd_opts.no_read_lora_meta):
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "lora/" + self.name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.alias = self.metadata.get('ss_output_name', self.name)

        self.hash = None
        self.shorthash = None
        self.set_hash(
            self.metadata.get('sshs_model_hash') or
            hashes.sha256_from_cache(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or
            ''
        )

        self.sd_version = self.detect_version()

    def detect_version(self):
        if str(self.metadata.get('ss_base_model_version', "")).startswith("sdxl_"):
            return SdVersion.SDXL
        elif str(self.metadata.get('ss_v2', "")) == "True":
            return SdVersion.SD2
        elif len(self.metadata):
            return SdVersion.SD1

        return SdVersion.Unknown

    def set_hash(self, v):
        self.hash = v
        self.shorthash = self.hash[0:12]

        if self.shorthash:
            from scripts import cp_networks
            cp_networks.available_network_hash_lookup[self.shorthash] = self

    def read_hash(self):
        if not self.hash:
            self.set_hash(hashes.sha256(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or '')

    def get_alias(self):
        from scripts import cp_networks
        if shared.opts.lora_preferred_name == "Filename" or self.alias.lower() in cp_networks.forbidden_network_aliases:
            return self.name
        else:
            return self.alias


class Network:  # LoraModule
    def __init__(self, name, network_on_disk: NetworkOnDisk):
        self.name = name
        self.network_on_disk = network_on_disk
        self.te_multiplier = 1.0
        self.unet_multiplier = 1.0
        self.dyn_dim = None
        self.modules = {}
        self.mtime = None

        self.mentioned_name = None
        """the text that was used to add the network to prompt - can be either name or an alias"""
        self.latest_mask_info = None
        self.mask_dic = None
        self.mask_hash = None

    
    def set_mask(self, mask, height=None, width=None, hr_height=None, hr_width=None):
        if mask is None:
            # clear latest mask
            # print("clear mask")
            self.latest_mask_info = None
            self.mask_dic = None
            self.mask_hash = None
            return

        # check mask image and h/w are same
        if (
            self.latest_mask_info is not None
            and torch.equal(mask, self.latest_mask_info[0])
            and (height, width, hr_height, hr_width) == self.latest_mask_info[1:]
        ):
            # print("mask not changed")
            return

        self.latest_mask_info = (mask, height, width, hr_height, hr_width)
        breakpoint()
        org_dtype = mask.dtype
        if mask.dtype == torch.bfloat16:
            mask = mask.to(torch.float)

        mask_dic = {}
        mask = mask.unsqueeze(0).unsqueeze(1)  # b(1),c(1),h,w

        def resize_add(mh, mw):
            # print(mh, mw, mh * mw)
            m = torch.nn.functional.interpolate(mask, (mh, mw), mode="bilinear")  # doesn't work in bf16
            m = m.to(org_dtype)
            mask_dic[mh * mw] = m

        for h, w in [(height, width), (hr_height, hr_width)]:
            if not h or not w:
                continue

            h = h // 8
            w = w // 8
            for i in range(4):
                resize_add(h, w)
                if h % 2 == 1 or w % 2 == 1:  # add extra shape if h/w is not divisible by 2
                    resize_add(h + h % 2, w + w % 2)
                h = (h + 1) // 2
                w = (w + 1) // 2
        self.mask_dic = mask_dic
        self.mask_hash = mask.__hash__()
        return


class ModuleType:
    def create_module(self, net: Network, weights: NetworkWeights):
        return None


class NetworkModule:
    def __init__(self, net: Network, weights: NetworkWeights):
        self.network = net
        self.network_key = weights.network_key
        self.sd_key = weights.sd_key
        self.sd_module = weights.sd_module

        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        self.dim = None
        self.bias = weights.w.get("bias")
        self.alpha = weights.w["alpha"].item() if "alpha" in weights.w else None
        self.scale = weights.w["scale"].item() if "scale" in weights.w else None


    def multiplier(self):
        if 'transformer' in self.sd_key[:20]:
            return self.network.te_multiplier
        else:
            return self.network.unet_multiplier

    def calc_scale(self):
        if self.scale is not None:
            return self.scale
        if self.dim is not None and self.alpha is not None:
            return self.alpha / self.dim

        return 1.0

    def finalize_updown(self, updown, orig_weight, output_shape):
        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = updown.reshape(output_shape)

        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)

        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)

        return updown * self.calc_scale() * self.multiplier()

    def calc_updown(self, target):
        raise NotImplementedError()

    def forward(self, x, y):
        raise NotImplementedError()
