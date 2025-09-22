import os
import sys
import time
import requests
import queue
import platform
from tqdm import tqdm
from colorama import init, Fore, Style
import threading
import atexit
import json
from collections import defaultdict
from multiprocessing import current_process
DEFAULT_DOWNLOAD_PREFIX = "https://www.modelscope.cn/"
HF_DOWNLOAD_PREFIX = "https://huggingface.co/"
CURRENT_DOWNLOAD_PREFIX = os.getenv('CURRENT_DOWNLOAD_PREFIX', DEFAULT_DOWNLOAD_PREFIX)
current_source = "ModelScope魔搭国内源" if CURRENT_DOWNLOAD_PREFIX == DEFAULT_DOWNLOAD_PREFIX else "HuggingFace拥抱脸国外源"

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = script_dir

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print_colored(f"√创建目录: {directory}", Fore.GREEN)
        except Exception as e:
            print_colored(f"×创建目录失败: {directory}, 错误: {e}", Fore.RED)
            return False
    return True

def load_model_paths():
    global models_root

    # 直接使用默认路径，不再尝试加载配置文件
    models_root = os.path.normpath(os.path.join(script_dir, "models"))
    path_mapping = {
        "audio_encoders": [os.path.join(models_root, "audio_encoders")],
        "checkpoints": [os.path.join(models_root, "checkpoints")],
        "clip": [os.path.join(models_root, "clip")],
        "clip_gguf": [os.path.join(models_root, "clip_gguf")],
        "clip_vision": [os.path.join(models_root, "clip_vision")],
        "configs": [os.path.normpath(os.path.join(models_root, "configs"))],
        "controlnet": [os.path.join(models_root, "controlnet")],
        "diffusion_models": [os.path.join(models_root, "diffusion_models")],
        "diffusers": [os.path.join(models_root, "diffusers")],
        "embeddings": [os.path.join(models_root, "embeddings")],
        "face_parsing": [os.path.join(models_root, "face_parsing")],
        "foley": [os.path.join(models_root, "foley")],
        "gligen": [os.path.join(models_root, "gligen")],
        "hypernetworks": [os.path.join(models_root, "hypernetworks")],
        "inpaint": [os.path.join(models_root, "inpaint")],
        "insightface": [os.path.join(models_root, "insightface")],
        "ipadapter": [os.path.join(models_root, "ipadapter")],
        "layer_model": [os.path.join(models_root, "layer_model")],
        "LLM": [os.path.join(models_root, "LLM")],
        "loras": [os.path.join(models_root, "loras")],
        "model_patches": [os.path.join(models_root, "model_patches")],
        "onnx": [os.path.join(models_root, "onnx")],
        "photomaker": [os.path.join(models_root, "photomaker")],
        "prompt_generator": [os.path.join(models_root, "prompt_generator")],
        "pulid": [os.path.join(models_root, "pulid")],
        "rembg": [os.path.join(models_root, "rembg")],
        "sam2": [os.path.join(models_root, "sam2")],
        "sams": [os.path.join(models_root, "sams")],
        "style_models": [os.path.join(models_root, "style_models")],
        "text_encoders": [os.path.join(models_root, "text_encoders")],
        "transformers": [os.path.join(models_root, "transformers")],
        "unet": [os.path.join(models_root, "unet")],
        "upscale_models": [os.path.join(models_root, "upscale_models")],
        "vae": [os.path.join(models_root, "vae")],
        "vae_approx": [os.path.join(models_root, "vae_approx")],
        "yolo": [os.path.join(models_root, "yolo")],
        "pprompt_generator\\minicpm-v-4_5-int4": [os.path.join(models_root, "prompt_generator", "MiniCPM-V-4_5-int4")],
        "transformers\\TencentGameMate\\chinese-wav2vec2-base": [os.path.join(models_root, "transformers", "TencentGameMate", "chinese-wav2vec2-base")],
        "foley\\hunyuanvideo-foley-xxl": [os.path.join(models_root, "foley", "hunyuanvideo-foley-xxl")],
        "LLM\\CogFlorence-2.2-Large": [os.path.join(models_root, "LLM", "CogFlorence-2.2-Large")]
    }
    for key in path_mapping:
        path_mapping[key] = [
            os.path.abspath(p) if not os.path.isabs(p) else p
            for p in path_mapping[key]
        ]
        path_mapping[key] = list(set(path_mapping[key]))

    for key, paths in path_mapping.items():
        for path in paths:
            ensure_directory_exists(path)

    return path_mapping

def cleanup():
    if os.path.exists("downloadlist.txt"):
        os.remove("downloadlist.txt")
        print("已删除 'downloadlist.txt' 文件。")
atexit.register(cleanup)

init(autoreset=True)
class DownloadStatus:
    def __init__(self, filename, total_size):
        self.filename = filename
        self.total_size = total_size
        self.progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=filename,
            position=0,
            leave=True
        )

def print_colored(text, color=Fore.WHITE):
    print(f"{color}{text}{Style.RESET_ALL}")

def check_python_embedded():
    python_exe = sys.executable
    print(f"Python解析器路径: {python_exe}")

    if platform.system() =='Windows' and "python_embeded" not in python_exe.lower():
        print_colored("×当前 Python 解释器不在 python_embeded 目录中，请检查运行环境", Fore.RED)
        input("按任意键继续。")
        sys.exit(1)

def normalize_path(path):
    path_parts = path.split('/')
    if len(path_parts) < 2:
        return os.path.abspath(path)

    path_type = path_parts[0].lower()
    filename = '/'.join(path_parts[1:])
    
    # 直接使用固定的models目录，去掉多目录排序和选择逻辑
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", path_type)
    
    # 确保目录存在
    os.makedirs(base_dir, exist_ok=True)
    
    full_path = os.path.join(base_dir, filename)
    return os.path.abspath(full_path)

def typewriter_effect(text, delay=0.01):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
        
    print()


def get_unique_filename(file_path, extension=".corrupted"):
    base = file_path + extension
    counter = 1
    while os.path.exists(base):
        base = f"{file_path}{extension}_{counter}"
        counter += 1
    return base

def validate_files(packages):
    cleanup()
    path_mapping = load_model_paths()
    print_colored(f">>>>>>默认模型根目录为：{models_root}<<<<<<", Fore.YELLOW)
    print()
    download_files = {}
    missing_package_names = []
    package_percentages = {}
    package_sizes = {}
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    for package_key, package_info in packages.items():
        package_name = package_info["name"]
        package_note = package_info.get("note", "")
        files_and_sizes = package_info["files"]
        download_links = package_info["download_links"]

        # 适配新的三元组格式：(目录名, 下载URL, 文件大小)
        total_size = sum([size for _, _, size in files_and_sizes])
        total_size_gb = total_size / (1024 ** 3)
        non_missing_size = 0

        print(f"－－－－－－－", end='')
        time.sleep(0.1)
        print(f"校验{package_name}文件－－－－{package_note}")

        missing_files = []
        size_mismatch_files = []
        case_mismatch_files = []

        for dir_name, download_url, expected_size in files_and_sizes:
            # 从下载URL中提取文件名
            expected_filename = os.path.basename(download_url)
            
            # 构建完整的目标路径
            target_dir = os.path.join(models_root, dir_name)
            target_path = os.path.join(target_dir, expected_filename)
            
            # 确保目录存在，不存在则创建
            if not os.path.exists(target_dir):
                try:
                    os.makedirs(target_dir, exist_ok=True)
                    print(f"{Fore.GREEN}创建目录: {target_dir}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}创建目录失败: {target_dir} - {str(e)}{Style.RESET_ALL}")
            
            # 检查文件是否存在
            if os.path.exists(target_path):
                actual_size = os.path.getsize(target_path)
                if actual_size != expected_size:
                    size_mismatch_files.append((target_path, actual_size, expected_size))
                else:
                    non_missing_size += expected_size
            else:
                # 存储目录名和文件名组合
                file_key = f"{dir_name}/{expected_filename}"
                missing_files.append((file_key, expected_size))

        if total_size > 0:
            non_missing_percentage = (non_missing_size / total_size) * 100
            package_percentages[package_name] = non_missing_percentage
            package_sizes[package_name] = total_size_gb

        if case_mismatch_files:
            print(f"{Fore.RED}×{package_name}中有文件名大小写不匹配，请检查以下文件:{Style.RESET_ALL}")
            for file, expected_filename in case_mismatch_files:
                print(f"文件: {normalize_path(file)}")
                time.sleep(0.1)
                print(f"正确文件名: {expected_filename}")
                
                corrected_file_path = os.path.join(os.path.dirname(file), expected_filename)
                os.rename(file, corrected_file_path)
                print(f"{Fore.GREEN}文件名已更正为: {expected_filename}{Style.RESET_ALL}")

        if size_mismatch_files:
            print(f"{Fore.RED}×{package_name}中有文件大小不匹配，可能存在下载不完全或损坏，请检查列出的文件。{Style.RESET_ALL}")
            for file, actual_size, expected_size in size_mismatch_files:
                normalized_path = normalize_path(file)
                print(f"{normalized_path} 当前大小={actual_size}, 预期大小={expected_size}")
                time.sleep(0.1)
                
                corrupted_file_path = get_unique_filename(file)
                os.rename(file, corrupted_file_path)
                print(f"{Fore.YELLOW}文件已重命名为: {normalize_path(corrupted_file_path)}（大小不匹配）{Style.RESET_ALL}")
                
                relative_path = os.path.relpath(file, root).replace(os.sep, '/')
                download_files[relative_path] = expected_size
                if package_name not in missing_package_names:
                    missing_package_names.append(package_name)

        if missing_files:
            print(f"{Fore.RED}×{package_name}有文件缺失，请检查以下文件:{Style.RESET_ALL}")
            for file, expected_size in missing_files:
                print(normalize_path(file))
                # 确保使用正确的键值格式
                download_files[file] = expected_size
            if package_name not in missing_package_names:
                missing_package_names.append(package_name)

            if package_info["download_links"]:
                for link in package_info["download_links"]:
                    print(f"{Fore.YELLOW}{link}{Style.RESET_ALL}")
        if not missing_files and not size_mismatch_files and not case_mismatch_files:
            print(f"{Fore.GREEN}√{package_name}文件全部验证通过{Style.RESET_ALL}")


    if missing_package_names:
        print(f"{Fore.RED}△以下包体缺失文件，请检查并重新下载：{Style.RESET_ALL}")
        for package_name in missing_package_names:
            percentage = package_percentages.get(package_name, 0)
            total_size_gb = package_sizes.get(package_name, 0)

            missing_size_gb = total_size_gb * (1 - (percentage / 100))
            print(f"- {package_name} - 总大小：{total_size_gb:.2f}GB，完整度：{percentage:.2f}%，尚需下载：{missing_size_gb:.2f}GB")

    sorted_download_files = sorted(download_files.items(), key=lambda x: x[1])
    if sorted_download_files:
        with open("downloadlist.txt", "w", encoding="utf-8", newline='') as f1:
            for file, size in sorted_download_files:
                # 查找原始的下载链接
                link = None
                
                # 提取目标文件的目录和文件名
                target_basename = os.path.basename(file)
                target_dir = os.path.dirname(file) if '/' in file else ''
                
                for package_key, package_info in packages.items():
                    for dir_name, download_url, expected_size in package_info["files"]:
                        url_basename = os.path.basename(download_url)
                        
                        # 匹配文件名
                        if url_basename == target_basename and dir_name in file:
                            link = download_url
                            break
                    if link:
                        break
                
                if not link:
                    print(f"{Fore.RED}警告: 未找到文件 '{file}' 的下载链接{Style.RESET_ALL}")
                    continue
                
                # 写入目录、链接和大小，用逗号分隔
                f1.write(f"{target_dir},{link},{size}\n")
        print(f"{Fore.YELLOW}>>>缺失文件下载链接已保存到 'downloadlist.txt'。<<<<<<<<<<<<<<<<<<<<<{Style.RESET_ALL}")
    if "[1]基础模型" in missing_package_names:
        package_id = 1
        selected_package = None
        for package_name, package_info in packages.items():
            if package_info["id"] == package_id:
                selected_package = package_info
                break

        print(f"\n{Fore.CYAN}△检测到基础模型不完整，自动触发下载流程...{Style.RESET_ALL}")
        get_download_links_for_package({package_name: selected_package}, "downloadlist.txt")
        auto_download_missing_files_with_retry(max_threads=5)



def delete_log_files():
    """
    删除与脚本所在位置一致的 logs 目录下的所有 .logs 文件
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")

    if not os.path.exists(logs_dir):
        print(f"{Fore.RED}△未找到指定日志目录: {logs_dir}{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}△正在清理目录 '{logs_dir}' 中的日志文件...{Style.RESET_ALL}")

    total_size = 0
    files_found = False
    files_to_delete = []

    for root, _, files in os.walk(logs_dir):
        for file in files:
            if file.endswith(".log"):
                files_found = True
                file_path = os.path.join(root, file)
                files_to_delete.append(file_path)
                total_size += os.path.getsize(file_path)

    if files_found:
        print(f"{Fore.YELLOW}△以下日志文件将被删除：{Style.RESET_ALL}")
        for file_path in files_to_delete:
            print(f"- {file_path}")

        print(f"{Fore.CYAN}△可清理的磁盘空间: {total_size / (1024 * 1024):.2f} MB{Style.RESET_ALL}")

        confirm = input(f"{Fore.GREEN}△是否确认删除这些日志文件？(y/n): {Style.RESET_ALL}")
        if confirm.lower() == 'y':
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"{Fore.GREEN}√已删除日志文件: {file_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}△删除文件时出错: {file_path}, 错误原因: {e}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}△删除操作已取消。{Style.RESET_ALL}")
    else:
        print(f">>>未找到需要删除的日志文件<<<")
        print()

def download_file_with_resume(link, file_path, position, result_queue, max_retries=5, lock=None):
    partial_file_path = file_path + ".partial"
    retries = 0
    while retries < max_retries:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.exists(partial_file_path):
                resume_size = os.path.getsize(partial_file_path)
                headers = {'Range': f"bytes={resume_size}-"}
            else:
                resume_size = 0
                headers = {}
            response = requests.get(link, stream=True, headers=headers)
            total_size = int(response.headers.get('content-length', 0)) + resume_size
            block_size = 8192

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(partial_file_path, 'ab') as file, tqdm(
                    desc=os.path.basename(file_path),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    position=position,
                    initial=resume_size
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress_bar.update(len(data))

            final_file_path = os.path.normpath(file_path)
            partial_file_path = os.path.normpath(partial_file_path)
            os.rename(partial_file_path, final_file_path)
            print(f"{Fore.GREEN}√下载完成：{final_file_path}{Style.RESET_ALL}")

            if lock:
                with lock:
                    remove_link_from_downloadlist(link)

            result_queue.put(True)
            return

        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}△下载失败，正在重试... 错误：{e}{Style.RESET_ALL}")
            retries += 1
            time.sleep(5)
        except Exception as e:
            print(f"{Fore.RED}发生错误：{e}{Style.RESET_ALL}")
            result_queue.put(False)
            return

    print(f"△下载链接失败：{link}")
    result_queue.put(False)

def remove_link_from_downloadlist(link):
    """
    删除下载列表中已成功下载的条目
    :param link: 下载链接
    :return: None
    """
    with open("downloadlist.txt", "r") as f:
        lines = f.readlines()

    with open("downloadlist.txt", "w") as f:
        for line in lines:
            if link.strip() not in line.strip():
                f.write(line)
def trigger_manual_download():
    """手动触发指定文件下载"""
    path_mapping = load_model_paths()

    for link in MANUAL_DOWNLOAD_LIST:
        if "SimpleModels/" in link:
            path_part = link.split("SimpleModels/", 1)[1]
            path_parts = path_part.split('/')
            path_type = path_parts[0].lower()
            rel_path = '/'.join(path_parts[1:])
        else:
            continue

        sorted_base_dir = sorted(
            path_mapping.get(path_type, []),
            key=lambda x: (
                0 if "SimpleModels" in x else
                1 if any(part == "models" for part in x.split(os.sep)) else 2,
                x
            )
        )

        target_base_dir = None
        for base_dir in sorted_base_dir:
            if os.path.exists(base_dir):
                target_base_dir = base_dir
                break
        if not target_base_dir:
            continue

        file_name = os.path.basename(link)
        save_path = os.path.join(target_base_dir, rel_path)

        if os.path.exists(save_path):
            print(f"{Fore.GREEN}△文件已存在，跳过下载: {save_path}{Style.RESET_ALL}")
            continue

        print(f"{Fore.CYAN}△开始下载: {file_name}{Style.RESET_ALL}")
        result_queue = queue.Queue()
        download_file_with_resume(link, save_path, 0, result_queue)

def auto_download_missing_files_with_retry(max_threads=5):
    if not os.path.exists("downloadlist.txt"):
        print("未找到 'downloadlist.txt' 文件。")
        return

    with open("downloadlist.txt", "r") as f:
        links = f.readlines()

    if not links:
        print("没有缺失文件需要下载！")
        return

    path_mapping = load_model_paths()
    result_queue = queue.Queue()
    lock = threading.Lock()

    task_queue = queue.Queue()
    for position, line in enumerate(links):
        task_queue.put((position, line.strip()))

    def worker():
        while not task_queue.empty():
            try:
                position, line = task_queue.get_nowait()
                # 尝试按三个部分分割：目录、链接、大小
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    # 第一部分是目录，最后一部分是大小，中间的所有部分合并成链接
                    dir_path = parts[0]
                    size = parts[-1]
                    link = ','.join(parts[1:-1])
                elif len(parts) == 2:
                    # 兼容旧格式
                    link, size = parts
                    dir_path = ""
                else:
                    print(f"{Fore.RED}行格式错误: {line}{Style.RESET_ALL}")
                    task_queue.task_done()
                    continue
                
                # 使用dir_path构建完整的保存路径
                size_mb = int(size) / (1024 * 1024)
                print(f"{Fore.CYAN}▶ 正在下载: {link} ({size_mb:.1f}MB){Style.RESET_ALL}")
                
                # 从链接获取目录类型
                path_type = None
                # 检查是否是特殊的inpaint文件
                for package_name, package_info in packages.items():
                    for file_info in package_info["files"]:
                        if len(file_info) >= 2 and file_info[1] == link:
                            path_type = file_info[0]  # 获取目录名
                            break
                    if path_type:
                        break
                
                # 如果找不到对应的目录类型，使用默认值
                if not path_type:
                    # 尝试从URL路径中提取合理的目录名
                    url_path = link.split("/")
                    # 寻找常见的模型目录名
                    common_model_dirs = ["vae", "loras", "checkpoints", "clip", "diffusion_models"]
                    for dir_name in common_model_dirs:
                        if dir_name.lower() in link.lower():
                            path_type = dir_name
                            break
                    # 如果还是找不到，使用默认目录
                    if not path_type:
                        path_type = "other"
                
                script_path = os.path.abspath(__file__)
                comfyui_dir = os.path.dirname(script_path)
                models_dir = os.path.join(comfyui_dir, "models")
                target_base_dir = os.path.join(models_dir, path_type)
                
                try:
                    os.makedirs(target_base_dir, exist_ok=True)
                except Exception as e:
                    print(f"{Fore.RED}×目录创建失败[{target_base_dir}]: {str(e)}{Style.RESET_ALL}")
                    continue

                # 直接使用URL的basename作为文件名
                file_name = os.path.basename(link)
                file_path = os.path.join(target_base_dir, file_name)

                thread = threading.Thread(
                    target=download_file_with_resume,
                    args=(link, file_path, position, result_queue, 5, lock)
                )
                thread.start()
                thread.join()

                task_queue.task_done()
            except queue.Empty:
                break

    threads = []
    for _ in range(max_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    task_queue.join()

    success_count = 0
    fail_count = 0

    while not result_queue.empty():
        success = result_queue.get()
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"√下载成功：{success_count}个")
    print(f"×下载失败：{fail_count}个")

    if fail_count == 0 and success_count > 0:
        if os.path.exists("downloadlist.txt"):
            os.remove("downloadlist.txt")
            print("√下载完成，已删除 'downloadlist.txt' 文件。")
            validate_files(packages)
    else:
        print(f"△有{fail_count}个文件下载失败，请检查网络连接或手动下载文件。")

def get_download_links_for_package(packages, download_list_path):
    """
    根据 packages 中的 files 列表生成路径，并与 downloadlist.txt 中的需求进行比对，
    更新 downloadlist.txt 中需要下载的文件，只保留 files 中有的文件链接。
    """
    if not os.path.exists(download_list_path):
        print(f"{Fore.RED}>>>downloadlist.txt不存在，输入【R】重新检测<<<{Style.RESET_ALL}")
        return []

    with open(download_list_path, "r") as f:
        existing_links = [line.strip().split(",")[0] for line in f.readlines()]

    valid_files = []
    with open(download_list_path, "r") as f:
        existing_lines = [line.strip() for line in f.readlines()]

    for line in existing_lines:
        existing_link = line.split(",")[1]
        # 尝试从现有行中获取大小
        try:
            existing_size = line.split(",")[2]
        except IndexError:
            existing_size = "0"
            
        for package_name, package_info in packages.items():
            for file_info in package_info["files"]:
                # 根据您的提示，列表第一个元素为path_mapping目录
                if len(file_info) >= 2:
                    # 检查文件信息格式
                    if isinstance(file_info, (list, tuple)):
                        # 假设格式是 (dir_name, download_url, expected_size)
                        if len(file_info) == 3:
                            dir_name, generated_link, file_size = file_info
                        elif len(file_info) == 2:
                            dir_name, generated_link = file_info
                            file_size = existing_size
                        
                        if generated_link == existing_link:
                            # 保持格式一致，方便后续处理
                            valid_files.append((dir_name, generated_link, file_size))
                            break

    # 按文件大小排序
    valid_files = sorted(valid_files, key=lambda x: int(x[2]) if isinstance(x[2], str) and x[2].isdigit() else (x[2] if isinstance(x[2], int) else 0))

    with open(download_list_path, "w") as f:
        for mapping, download_path, size in valid_files:
            f.write(f"{mapping},{download_path},{size}\n")

    print(f"{Fore.YELLOW}>>>下载列表已更新，开始下载（关闭窗口可中断）。<<<{Style.RESET_ALL}")

    return valid_files

packages = {
    "base_package": {
        "id": 1,
        "name": "[1]基础模型",
        "note": "运行必须的模型",
        "files": [
            ("vae", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan2_1_VAE_bf16.safetensors", 253806278),
            ("clip", "https://www.modelscope.cn/models/city96/umt5-xxl-encoder-gguf/resolve/master/umt5-xxl-encoder-Q8_0.gguf",6043068256),
            ("controlnet","https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan21_Uni3C_controlnet_fp16.safetensors",1997314376),
            ("face_parsing","https://hf-mirror.com/jellyhe/parsing_bisenet.pth/resolve/main/parsing_bisenet.pth",53289463),
            ("loras", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors", 613561776),
            ("loras", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors", 613561776),
            ("loras", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan22_FunReward/Wan2.2-Fun-A14B-InP-HIGH-MPS_resized_dynamic_avg_rank_21_bf16.safetensors", 140945036),
            ("loras", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan22_FunReward/Wan2.2-Fun-A14B-InP-LOW-HPS2.1_resized_dynamic_avg_rank_15_bf16.safetensors", 101752852),
            ("loras", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan22-Lightning/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors", 613561776),
            ("loras", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan22-Lightning/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors", 613561776),
            ("loras", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", 630697104),
            ("loras", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 738005744),
            ("loras","https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Stand-In/Stand-In_wan2.1_T2V_14B_ver1.0_fp16.safetensors",314599728),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/added_tokens.json", 2862),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/config.json", 1995),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/configuration.json", 51),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/configuration_minicpm.py", 3367),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/generation_config.json", 268),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/image_processing_minicpmv.py", 20757),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/merges.txt", 1671853),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/model-00001-of-00002.safetensors", 4827364414),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/model-00002-of-00002.safetensors", 1699920944),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/model.safetensors.index.json", 267079),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/modeling_minicpmv.py", 17679),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/modeling_navit_siglip.py", 41835),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/preprocessor_config.json", 714),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/processing_minicpmv.py", 11026),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/resampler.py", 11374),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/special_tokens_map.json", 12103),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/tokenization_minicpmv_fast.py", 1647),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/tokenizer.json", 11437868),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/tokenizer_config.json", 25786),
            ("prompt_generator\\MiniCPM-V-4_5-int4", "https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4/resolve/master/vocab.json", 2776833),
            ("sam2", "https://www.modelscope.cn/models/Kijai/sam2-safetensors/resolve/master/sam2.1_hiera_base_plus-fp16.safetensors", 161773292),
            ("yolo", "https://hf-mirror.com/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt", 6247065)
        ],
        "download_links": []
    },
    "i2v_package": {
        "id": 2,
        "name": "[2]图生视频模型",
        "note": "图生视频模型（Q4_K_M）",
        "files": [
            ("diffusion_models", "https://www.modelscope.cn/models/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/master/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf", 9651728896),
            ("diffusion_models", "https://www.modelscope.cn/models/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/master/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf", 9651728896)
        ],
        "download_links": []
    },
    "t2v_package": {
        "id": 3,
        "name": "[3]文生视频模型",
        "note": "文生视频模型（Q4_K_M）",
        "files": [
            ("diffusion_models", "https://www.modelscope.cn/models/AI-ModelScope/Wan2.2_T2V_A14B_VACE-test/resolve/master/Wan2.2_T2V_High_Noise_14B_VACE-Q4_K_M.gguf", 11629612832),
            ("diffusion_models", "https://www.modelscope.cn/models/AI-ModelScope/Wan2.2_T2V_A14B_VACE-test/resolve/master/Wan2.2_T2V_Low_Noise_14B_VACE-Q4_K_M.gguf", 11629612832)
        ],
        "download_links": []
    },
    "t2i_package": {
        "id": 4,
        "name": "[4]文生图片补充模型",
        "note": "文生图片模型（Q4_K_M）",
        "files": [
            ("loras", "https://www.modelscope.cn/models/windecay/my_favorite_loras/resolve/master/WAN2.2-HighNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters.safetensors", 153492456),
            ("loras", "https://www.modelscope.cn/models/windecay/my_favorite_loras/resolve/master/WAN2.2-LowNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters.safetensors", 153492448),
            ("upscale_models","https://www.modelscope.cn/models/muse/4xNomos8kSCHAT-L/resolve/master/4xNomos8kSCHAT-L.pth",331564661),
            ("upscale_models","https://www.modelscope.cn/models/muse/RealESRGAN_x2plus/resolve/master/RealESRGAN_x2plus.pth",67061725)
        ],
        "download_links": []
    },
    "infinitetalk_package": {
        "id": 5,
        "name": "[5]infinitetalk对口型模型",
        "note": "infinitetalk对口型模型（fp8）",
        "files": [
            ("diffusion_models", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors", 16993877896),
            ("diffusion_models", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy_fp8_scaled/resolve/master/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors", 2713548210),
            ("clip_vision", "https://www.modelscope.cn/models/shiertier/ComfyUI-clip_vision/resolve/master/clip_vision_vit_h.safetensors", 1972298538),
            ("transformers\\TencentGameMate\\chinese-wav2vec2-base", "https://www.modelscope.cn/models/TencentGameMate/chinese-wav2vec2-base/resolve/master/config.json", 1951),
            ("transformers\\TencentGameMate\\chinese-wav2vec2-base", "https://www.modelscope.cn/models/TencentGameMate/chinese-wav2vec2-base/resolve/master/pytorch_model.bin", 380261837),
            ("transformers\\TencentGameMate\\chinese-wav2vec2-base", "https://www.modelscope.cn/models/TencentGameMate/chinese-wav2vec2-base/resolve/master/preprocessor_config.json", 160)
        ],
        "download_links": []
    },
    "qwen_edit_package": {
        "id": 6,
        "name": "[6]Qwen图像编辑",
        "note": "Qwen图像编辑（Q4_K_M），VACE动作深度参考",
        "files": [
            ("diffusion_models", "https://www.modelscope.cn/models/QuantStack/Qwen-Image-Edit-GGUF/resolve/master/Qwen_Image_Edit-Q4_K_M.gguf", 13065746976),
            ("loras", "https://www.modelscope.cn/models/lightx2v/Qwen-Image-Lightning/resolve/master/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors", 849608296),
            ("clip", "https://www.modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI/resolve/master/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9384670680),
            ("vae", "https://www.modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI/resolve/master/split_files/vae/qwen_image_vae.safetensors", 253806246),
            ("controlnet", "https://www.modelscope.cn/models/Comfy-Org/Qwen-Image-InstantX-ControlNets/resolve/master/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors", 3536027816)
        ],
        "download_links": []
    },
    "remover_package": {
        "id": 7,
        "name": "[7]MinimaxRemover视频消除",
        "note": "MinimaxRemover视频元素消除",
        "files": [
            ("diffusion_models", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/Wan2_1-MiniMaxRemover_1_3B_fp16.safetensors", 2254156824),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/config.json", 2445),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/configuration_florence2.py", 15125),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/generation_config.json", 51),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/model.safetensors", 1646020762),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/modeling_florence2.py", 127219),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/preprocessor_config.json", 631),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/processing_florence2.py", 46372),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/tokenizer.json", 2297961),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/tokenizer_config.json", 34),
            ("LLM\\CogFlorence-2.2-Large", "https://hf-mirror.com/thwri/CogFlorence-2.2-Large/resolve/main/vocab.json", 798293)
        ],
        "download_links": []
    },
    "foley_package":{
        "id": 8,
        "name": "[8]hunyuanvideo-foley视频音效",
        "note": "hunyuanvideo-foley视频音效",
        "files": [
            ("foley\\hunyuanvideo-foley-xxl", "https://www.modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo-Foley/resolve/master/config.yaml", 1313),
            ("foley\\hunyuanvideo-foley-xxl", "https://www.modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo-Foley/resolve/master/hunyuanvideo_foley.pth", 10301204679),
            ("foley\\hunyuanvideo-foley-xxl", "https://www.modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo-Foley/resolve/master/synchformer_state_dict.pth", 950058171),
            ("foley\\hunyuanvideo-foley-xxl", "https://www.modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo-Foley/resolve/master/vae_128d_48k.pth", 1486465965)
        ],
        "download_links": []
    },
    "rapid_package": {
        "id": 9,
        "name": "[9]rapid_aio_mega模型",
        "note": "rapid_aio_mega模型（fp8）",
        "files": [
            ("checkpoints", "https://www.modelscope.cn/models/Phr00t/WAN2.2-14B-Rapid-AllInOne/resolve/master/Mega-v2/wan2.2-rapid-mega-aio-v2.safetensors", 24334482407)
        ],
        "download_links": []
    },
    "animate_package": {
        "id": 10,
        "name": "[10]Wan2.2Animate视频重绘",
        "note": "Wan2.2Animate视频重绘（fp8）",
        "files": [
            ("diffusion_models", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy_fp8_scaled/resolve/master/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors", 18401760586),
            ("loras", "https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/WanAnimate_relight_lora_fp16.safetensors", 1436655192)
        ],
        "download_links": []
    }
}

# MANUAL_DOWNLOAD_LIST = []
# OBSOLETE_MODELS = []
def main():
    print()
    print_colored("★★★★★★★★★★★★★★★★★★欢迎使用模型自动下载器★★★★★★★★★★★★★★★★★★", Fore.CYAN)
    time.sleep(0.1)
    print()
    check_python_embedded()
    time.sleep(0.1)
    # print_instructions()
    # time.sleep(0.1)
    validate_files(packages)
    print()
    print_colored("★★★★★★★★★★★★★★★★★★★★★★检测已结束★★★★★★★★★★★★★★★★★★★★★★", Fore.CYAN)

if __name__ == "__main__":
    main()
    print()
    while True:
        print(f">>>按下【{Fore.YELLOW}Enter回车{Style.RESET_ALL}】----------------启动全部文件下载<<<     备注：支持断点续传，顺序从小文件开始。")
        print(f">>>输入【{Fore.YELLOW}包体编号{Style.RESET_ALL}】+【{Fore.YELLOW}回车{Style.RESET_ALL}】----------选择性下载补全<<<     备注：若速度太慢直接拿链接用P2P软件下载")
        user_input = input("请选择操作(不需要括号):")

        if user_input == "":
            print("※启动自动下载模块,支持断点续传，关闭窗口可中断。")
            auto_download_missing_files_with_retry(max_threads=5)
        elif user_input.isdigit():
            package_id = int(user_input)
            selected_package = None
            for package_name, package_info in packages.items():
                if package_info["id"] == package_id:
                    selected_package = package_info
                    break

            if selected_package:
                get_download_links_for_package({package_name: selected_package}, "downloadlist.txt")
                auto_download_missing_files_with_retry(max_threads=5)
            else:
                print(f"{Fore.RED}△包体编号{package_id} 无效，请输入正确的包体ID。{Style.RESET_ALL}")
        elif user_input.lower() == "r":
            print("重新检测文件...")
            validate_files(packages)
        # elif user_input.lower() == "h":
        #     CURRENT_DOWNLOAD_PREFIX = HF_DOWNLOAD_PREFIX
        #     current_source = "HuggingFace拥抱脸国外源"
        #     validate_files(packages)
        #     print(f"{Fore.GREEN}√下载源已切换到Huggingface：{CURRENT_DOWNLOAD_PREFIX}{Style.RESET_ALL}")
        #     print(f"{Fore.YELLOW}※提示：此切换只在本次运行有效，重启程序后将恢复默认设置。{Style.RESET_ALL}")
        # elif user_input.lower() == "m":
        #     CURRENT_DOWNLOAD_PREFIX = DEFAULT_DOWNLOAD_PREFIX
        #     current_source = "ModelScope魔搭国内源"
        #     validate_files(packages)
        #     print(f"{Fore.GREEN}√下载源已切换到ModelScope：{CURRENT_DOWNLOAD_PREFIX}{Style.RESET_ALL}")
        #     print(f"{Fore.YELLOW}※提示：此切换只在本次运行有效，重启程序后将恢复默认设置。{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}△无效的输入，请输入回车或有效的包体编号（不需要括号）。{Style.RESET_ALL}")