import os
import os.path
import re
import shutil
import json
import stat
import tqdm
import glob
from collections import OrderedDict
from multiprocessing.pool import ThreadPool as Pool
import torch
from loguru import logger

from modules import shared, sd_models, hashes
from scripts import safetensors_hack, model_util, util
from modules.paths_internal import models_path
import modules.scripts as scripts
import glob
from modules.timer import Timer


# MAX_MODEL_COUNT = shared.cmd_opts.addnet_max_model_count or 5
MAX_MODEL_COUNT = shared.cmd_opts.addnet_max_model_count if hasattr(shared.cmd_opts, "addnet_max_model_count") else 5
LORA_MODEL_EXTS = [".pt", ".ckpt", ".safetensors"]
re_legacy_hash = re.compile("\(([0-9a-f]{8})\)$")  # matches 8-character hashes, new hash has 12 characters
lora_models = {}  # "My_Lora(abcdef123456)" -> "C:/path/to/model.safetensors"
lora_model_names = {}  # "my_lora" -> "My_Lora(My_Lora(abcdef123456)"
legacy_model_names = {}
lora_models_dir = os.path.join(models_path, "Lora")
os.makedirs(lora_models_dir, exist_ok=True)

def move_blade_to(self, dev):
    # import gc
    used, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    logger.debug(f"[AddNet] before moving blade, cuda mem consumed: {used / 1024 ** 3 :.2f} GB, reserved mem: {reserved / 1024 ** 3 :.2f} GB")

    self.blade_input_blocks = self.blade_input_blocks.to(dev)
    self.blade_middle_block = self.blade_middle_block.to(dev)
    self.blade_output_blocks = self.blade_output_blocks.to(dev)

    # gc.collect()
    # if dev == devices.cpu:
    #     devices.torch_gc()
    used, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    logger.debug(f"[AddNet] after moving blade, cuda mem consumed: {used / 1024 ** 3 :.2f} GB, reserved mem: {reserved / 1024 ** 3 :.2f} GB")


def move_main_to(self, dev):
    # import gc
    used, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    logger.debug(f"[AddNet] before moving main, cuda mem consumed: {used / 1024 ** 3 :.2f} GB, reserved mem: {reserved / 1024 ** 3 :.2f} GB")
    self.main_input_blocks = self.main_input_blocks.to(dev)
    self.main_middle_block = self.main_middle_block.to(dev)
    self.main_output_blocks = self.main_output_blocks.to(dev)
    # gc.collect()
    # if dev == devices.cpu:
    #     devices.torch_gc()
    used, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    logger.debug(f"[AddNet] after moving main, cuda mem consumed: {used / 1024 ** 3 :.2f} GB, reserved mem: {reserved / 1024 ** 3 :.2f} GB")


def restore_main_branch_before_forward(self):
    # self.blade_input_blocks = self.input_blocks
    self.input_blocks = self.main_input_blocks

    # self.blade_middle_block = self.middle_block
    self.middle_block = self.main_middle_block

    # self.blade_output_blocks = self.output_blocks
    self.output_blocks = self.main_output_blocks


def is_safetensors(filename):
    return os.path.splitext(filename)[1] == ".safetensors"


def read_model_metadata(model_path, module):
    if model_path.startswith('"') and model_path.endswith('"'):  # trim '"' at start/end
        model_path = model_path[1:-1]
    if not os.path.exists(model_path):
        return None

    metadata = None
    if module == "LoRA":
        if os.path.splitext(model_path)[1] == ".safetensors":
            metadata = safetensors_hack.read_metadata(model_path)

    return metadata


def write_model_metadata(model_path, module, updates):
    if model_path.startswith('"') and model_path.endswith('"'):  # trim '"' at start/end
        model_path = model_path[1:-1]
    if not os.path.exists(model_path):
        return None

    from safetensors.torch import save_file

    back_up = shared.opts.data.get("additional_networks_back_up_model_when_saving", True)
    if back_up:
        backup_path = model_path + ".backup"
        if not os.path.exists(backup_path):
            print(f"[MetadataEditor] Backing up current model to {backup_path}")
            shutil.copyfile(model_path, backup_path)

    metadata = None
    tensors = {}
    if module == "LoRA":
        if os.path.splitext(model_path)[1] == ".safetensors":
            tensors, metadata = safetensors_hack.load_file(model_path, "cpu")

            for k, v in updates.items():
                metadata[k] = str(v)

            save_file(tensors, model_path, metadata)
            print(f"[MetadataEditor] Model saved: {model_path}")


def get_model_list(module, model, model_dir, sort_by):
    if model_dir == "":
        # Get list of models with same folder as this one
        model_path = lora_models.get(model, None)
        if model_path is None:
            return []
        model_dir = os.path.dirname(model_path)

    if not os.path.isdir(model_dir):
        return []

    found, _ = get_all_models([model_dir], sort_by, "")
    return list(found.keys())  # convert dict_keys to list


def traverse_all_files(curr_path, model_list):
    f_list = [(os.path.join(curr_path, entry.name), entry.stat()) for entry in os.scandir(curr_path)]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in LORA_MODEL_EXTS:
            model_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            model_list = traverse_all_files(fname, model_list)
    return model_list


def get_model_hash(metadata, filename, timer):
    if metadata is None:
        if shared.cmd_opts.no_hashing:
            model_hash = sd_models.model_hash_filename(filename.split('/')[-1])
            timer.record('get_model_hash:model_hash_filename')
        else:
            model_hash = hashes.calculate_sha256(filename)
            timer.record('get_model_hash:calculate_sha256')
        return model_hash

    if "sshs_model_hash" in metadata:
        return metadata["sshs_model_hash"]
    
    if shared.cmd_opts.no_hashing:
        model_hash = sd_models.model_hash_filename(filename.split('/')[-1])
        timer.record('get_model_hash:model_hash_filename')
        return model_hash
    model_hash = safetensors_hack.hash_file(filename)
    timer.record('get_model_hash:hash_file')
    return model_hash


def get_legacy_hash(metadata, filename, timer):
    if metadata is None:
        if shared.cmd_opts.no_hashing:
            model_hash = sd_models.model_hash_filename(filename.split('/')[-1])
            timer.record('get_legacy_hash:model_hash_filename')
        else:
            model_hash = sd_models.model_hash(filename)
            timer.record('get_legacy_hash:model_hash')
        return model_hash

    if "sshs_legacy_hash" in metadata:
        return metadata["sshs_legacy_hash"]
    
    model_hash = safetensors_hack.legacy_hash_file(filename)
    timer.record('get_legacy_hash:hash_file')
    return model_hash

import filelock

cache_filename = os.path.join(scripts.basedir(), "hashes.json")
cache_data = None


def cache(subsection):
    global cache_data

    if cache_data is None:
        with filelock.FileLock(cache_filename + ".lock"):
            if not os.path.isfile(cache_filename):
                cache_data = {}
            else:
                with open(cache_filename, "r", encoding="utf8") as file:
                    cache_data = json.load(file)

    s = cache_data.get(subsection, {})
    cache_data[subsection] = s

    return s


def dump_cache():
    with filelock.FileLock(cache_filename + ".lock"):
        with open(cache_filename, "w", encoding="utf8") as file:
            json.dump(cache_data, file, indent=4)


def get_model_rating(filename):
    if not model_util.is_safetensors(filename):
        return 0

    metadata = safetensors_hack.read_metadata(filename)
    return int(metadata.get("ssmd_rating", "0"))


def has_user_metadata(filename):
    if not model_util.is_safetensors(filename):
        return False

    metadata = safetensors_hack.read_metadata(filename)
    return any(k.startswith("ssmd_") for k in metadata.keys())


def hash_model_file(finfo):
    filename = finfo[0]
    stat = finfo[1]
    name = os.path.splitext(os.path.basename(filename))[0]

    # Prevent a hypothetical "None.pt" from being listed.
    timer = Timer()
    if name != "None":
        metadata = None

        cached = cache("hashes").get(filename, None)
        if cached is None or stat.st_mtime != cached["mtime"]:
            if metadata is None and model_util.is_safetensors(filename) and (not shared.cmd_opts.no_read_lora_meta):
                try:
                    metadata = safetensors_hack.read_metadata(filename)
                except Exception as ex:
                    return {"error": ex, "filename": filename}
            timer.record('read metadata')
            model_hash = get_model_hash(metadata, filename, timer)
            timer.record('get_model_hash')
            legacy_hash = get_legacy_hash(metadata, filename, timer)
            timer.record('get_legacy_hash')
            print(f'[AddNet] {timer.summary()} with {filename}')
        else:
            model_hash = cached["model"]
            legacy_hash = cached["legacy"]

    return {"model": model_hash, "legacy": legacy_hash, "fileinfo": finfo}


def get_all_models(paths, sort_by, filter_by):
    fileinfos = []
    for path in paths:
        if os.path.isdir(path):
            fileinfos += traverse_all_files(path, [])

    show_only_safetensors = shared.opts.data.get("additional_networks_show_only_safetensors", False)
    show_only_missing_meta = shared.opts.data.get("additional_networks_show_only_models_with_metadata", "disabled")

    if show_only_safetensors:
        fileinfos = [x for x in fileinfos if is_safetensors(x[0])]

    if show_only_missing_meta == "has metadata":
        fileinfos = [x for x in fileinfos if has_user_metadata(x[0])]
    elif show_only_missing_meta == "missing metadata":
        fileinfos = [x for x in fileinfos if not has_user_metadata(x[0])]

    print("[AddNet] Updating model hashes...")
    data = []
    thread_count = max(1, int(shared.opts.data.get("additional_networks_hash_thread_count", 1)))
    p = Pool(processes=thread_count)
    with tqdm.tqdm(total=len(fileinfos)) as pbar:
        for res in p.imap_unordered(hash_model_file, fileinfos):
            pbar.update()
            if "error" in res:
                print(f"Failed to read model file {res['filename']}: {res['error']}")
            else:
                data.append(res)
    p.close()

    cache_hashes = cache("hashes")

    res = OrderedDict()
    res_legacy = OrderedDict()
    filter_by = filter_by.strip(" ")
    if len(filter_by) != 0:
        data = [x for x in data if filter_by.lower() in os.path.basename(x["fileinfo"][0]).lower()]
    if sort_by == "name":
        data = sorted(data, key=lambda x: os.path.basename(x["fileinfo"][0]))
    elif sort_by == "date":
        data = sorted(data, key=lambda x: -x["fileinfo"][1].st_mtime)
    elif sort_by == "path name":
        data = sorted(data, key=lambda x: x["fileinfo"][0])
    elif sort_by == "rating":
        data = sorted(data, key=lambda x: get_model_rating(x["fileinfo"][0]), reverse=True)
    elif sort_by == "has user metadata":
        data = sorted(
            data, key=lambda x: os.path.basename(x["fileinfo"][0]) if has_user_metadata(x["fileinfo"][0]) else "", reverse=True
        )

    reverse = shared.opts.data.get("additional_networks_reverse_sort_order", False)
    if reverse:
        data = reversed(data)

    for result in data:
        finfo = result["fileinfo"]
        filename = finfo[0]
        stat = finfo[1]
        model_hash = result["model"]
        legacy_hash = result["legacy"]

        name = os.path.splitext(os.path.basename(filename))[0]

        # Commas in the model name will mess up infotext restoration since the
        # infotext is delimited by commas
        name = name.replace(",", "_")

        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            full_name = name + f"({model_hash[0:12]})"
            res[full_name] = filename
            res_legacy[legacy_hash] = full_name
            cache_hashes[filename] = {"model": model_hash, "legacy": legacy_hash, "mtime": stat.st_mtime}

    return res, res_legacy


def find_closest_lora_model_name(search: str):
    if not search or search == "None":
        return None

    # Match name and hash, case-sensitive
    # "MyModel-epoch00002(abcdef123456)"
    if search in lora_models:
        return search

    # Match model path, case-sensitive (from metadata editor)
    # "C:/path/to/mymodel-epoch00002.safetensors"
    if os.path.isfile(search):
        import json

        find = os.path.normpath(search)
        value = next((k for k in lora_models.keys() if lora_models[k] == find), None)
        if value:
            return value

    search = search.lower()

    # Match full name, case-insensitive
    # "mymodel-epoch00002"
    if search in lora_model_names:
        return lora_model_names.get(search)

    # Match legacy hash (8 characters)
    # "MyModel(abcd1234)"
    result = re_legacy_hash.search(search)
    if result is not None:
        model_hash = result.group(1)
        if model_hash in legacy_model_names:
            new_model_name = legacy_model_names[model_hash]
            return new_model_name

    # Use any model with the search term as the prefix, case-insensitive, sorted
    # by name length
    # "mymodel"
    applicable = [name for name in lora_model_names.keys() if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return lora_model_names[applicable[0]]


def update_models():
    global lora_models, lora_model_names, legacy_model_names
    paths = [lora_models_dir]
    if os.path.exists(shared.cmd_opts.lora_dir):
        paths.append(shared.cmd_opts.lora_dir)
    if os.path.exists(os.path.join(shared.cmd_opts.data_dir, 'models/Lora')) and os.path.isdir(os.path.join(shared.cmd_opts.data_dir, 'models/Lora')):
        paths.append(os.path.join(shared.cmd_opts.data_dir, 'models/Lora'))
    if shared.cmd_opts.uid is None:
        if os.environ.get('SERVICE_NAME', '') == '':
            for folder_path in glob.iglob(os.path.join(shared.cmd_opts.data_dir, 'users/*/models/Lora')):
                paths.append(folder_path)
        else:
            for folder_path in glob.iglob(os.path.join(shared.cmd_opts.data_dir, '*/models/Lora')):
                paths.append(folder_path)
    extra_lora_paths = util.split_path_list(shared.opts.data.get("additional_networks_extra_lora_path", ""))
    for path in extra_lora_paths:
        path = path.lstrip()
        if os.path.isdir(path):
            paths.append(path)
    paths = list(set(paths))
    print(paths)

    sort_by = shared.opts.data.get("additional_networks_sort_models_by", "name")
    filter_by = shared.opts.data.get("additional_networks_model_name_filter", "")
    res, res_legacy = get_all_models(paths, sort_by, filter_by)

    lora_models.clear()
    lora_models["None"] = None
    lora_models.update(res)

    for name_and_hash, filename in lora_models.items():
        if filename == None:
            continue
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        lora_model_names[name] = name_and_hash

    legacy_model_names = res_legacy
    dump_cache()


update_models()
