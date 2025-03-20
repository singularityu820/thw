import multiprocessing
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def process_single_pair(args):
    """
    处理单个文件对的核⼼函数
    返回：(EXR文件名, PNG文件名, 状态码, 错误信息)
    状态码：0=成功 1=数据不一致 2=读取错误 3=其他错误
    """
    exr_file, png_file, root, out_root = args
    exr_path = os.path.join(root, exr_file)
    png_path = os.path.join(root, png_file)
    npz_path = os.path.join(out_root, f"{exr_file[:-4]}.npz")

    try:
        # 读取并转换EXR数据
        input_exr = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
        if input_exr is None:
            return (exr_file, png_file, 2, "EXR读取失败")
        input_exr = input_exr[:, :, ::-1].copy()  # BGR转RGB

        # 读取并转换PNG数据
        png = cv2.imread(png_path)
        if png is None:
            return (exr_file, png_file, 2, "PNG读取失败")
        png = png[:, :, ::-1].copy()  # BGR转RGB

        # 处理PNG数据
        n, d, r, s = np.split(png, 4, axis=1)
        svbrdf = np.concatenate([n, d, r[:, :, :1], s], axis=2)

        # 保存NPZ文件
        np.savez(npz_path, input=input_exr, svbrdf=svbrdf)

        # 验证数据一致性（复用内存数据）
        data = np.load(npz_path)
        input_match = np.allclose(input_exr, data["input"], atol=1e-6)
        svbrdf_match = np.array_equal(svbrdf, data["svbrdf"])

        if not (input_match and svbrdf_match):
            return (exr_file, png_file, 1, f"数据不一致 EXR:{input_match} SVG:{svbrdf_match}")

        return (exr_file, png_file, 0, "成功")

    except Exception as e:
        return (exr_file, png_file, 3, f"处理异常: {str(e)}")


def batch_processor(root, out_root, max_workers=None):
    """批量处理器"""
    # 创建输出目录
    os.makedirs(out_root, exist_ok=True)

    # 获取并配对文件
    exr_files = sorted([f for f in os.listdir(root) if f.endswith(".exr")])
    png_files = sorted([f for f in os.listdir(root) if f.endswith(".png")])

    # 验证文件配对
    assert len(exr_files) == len(png_files), "EXR和PNG文件数量不匹配"
    print(f"找到{len(exr_files)}对需要处理的数据")

    # 准备任务参数
    task_args = [(exr, png, root, out_root) for exr, png in zip(exr_files, png_files)]

    # 创建日志文件
    log_path = os.path.join(out_root, "processing_errors.csv")
    with open(log_path, "w") as f:
        f.write("EXR文件,PNG文件,状态,错误信息\n")

    # 创建进程池
    num_workers = max_workers or (multiprocessing.cpu_count() // 2)
    print(f"使用{num_workers}个进程进行并行处理")

    # 进度跟踪
    start_time = time.time()
    processed = 0
    error_count = 0

    with multiprocessing.Pool(num_workers) as pool:
        # 使用imap_unordered获取最快完成的结果
        results = pool.imap_unordered(process_single_pair, task_args, chunksize=100)

        # 使用tqdm显示进度条
        with tqdm(open(log_path, "a"), total=len(task_args)) as log_file:
            for result in results:
                exr, png, code, msg = result
                processed += 1

                # 记录错误信息
                if code != 0:
                    error_count += 1
                    log_file.write(f"{exr},{png},{code},{msg}\n")
                    log_file.flush()  # 确保及时写入

                # 更新进度
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = processed / elapsed
                    tqdm.write(
                        f"已处理 {processed}/{len(task_args)} | "
                        f"错误率 {error_count / processed:.2%} | "
                        f"速度 {speed:.1f} 文件/秒"
                    )

    # 生成汇总报告
    total_time = time.time() - start_time
    print(f"\n处理完成！总计耗时: {total_time:.1f}秒")
    print(f"总处理文件对: {len(task_args)}")
    print(f"成功数量: {len(task_args) - error_count}")
    print(f"错误数量: {error_count}")
    print(f"平均速度: {len(task_args) / total_time:.1f} 文件/秒")


if __name__ == "__main__":
    # 配置路径
    input_root = "/home/zjj/dataset/MatSynth/train_HDR_Linear"
    output_root = "/home/zjj/dataset/MatSynth/train_HDR_Linear_npz"

    # # 启动批量处理（设置max_workers=None自动使用CPU核心数的一半）
    batch_processor(input_root, output_root, max_workers=None)
