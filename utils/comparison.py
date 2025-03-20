import os

import cv2
import numpy as np
from tqdm import tqdm


def read(path):
    img = cv2.imread(path)
    img1 = img[:, : 256 * 5]
    img2 = img[:, 256 * 8 : 256 * 9]
    img3 = img[:, -256:]
    img = np.hstack([img1, img2, img3])
    return img


if __name__ == "__main__":
    exps = ["004", "004_1", "010", "007", "007_1", "007_2"]
    exps = [f"/home/zjj/exps/hdr/results/{exp}/visualization" for exp in exps]
    img_names = os.listdir(exps[0])

    # 输出路径配置
    output_dir = "/home/zjj/exps/hdr/comparison/004_004_1_010_007_007_1_007_2"
    os.makedirs(output_dir, exist_ok=True)

    # 样式参数配置
    label_width = 200  # 标签栏宽度
    row_sep_height = 2  # 行间分隔线高度
    row_sep_color = (200, 200, 200)  # 浅灰色分隔线
    vertical_sep_width = 1  # 标签与图片间垂直线宽度
    vertical_sep_color = (150, 150, 150)  # 灰色垂直线

    for img_name in tqdm(img_names):
        rows = []
        for exp_path in exps:
            # 读取图片
            img_path = os.path.join(exp_path, img_name)
            img = read(img_path)

            # 获取实验名称
            exp_name = os.path.basename(os.path.dirname(exp_path))

            # 创建白色标签栏
            img_height, img_width = img.shape[:2]
            label = np.full((img_height, label_width, 3), 255, dtype=np.uint8)

            # 添加黑色文字（自动调整字体大小）
            max_font_scale = label_width / 100  # 基于标签宽度动态调整
            font = cv2.FONT_HERSHEY_SIMPLEX
            for font_scale in np.linspace(max_font_scale, 0.5, 10):
                text_size = cv2.getTextSize(exp_name, font, font_scale, 2)[0]
                if text_size[0] < label_width * 0.9 and text_size[1] < img_height * 0.3:
                    break

            text_x = (label_width - text_size[0]) // 2
            text_y = (img_height + text_size[1]) // 2
            cv2.putText(
                label,
                exp_name,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),  # 黑色文字
                2,
                cv2.LINE_AA,
            )

            # 添加标签与图片间的垂直分割线
            vertical_sep = np.full(
                (img_height, vertical_sep_width, 3), vertical_sep_color, dtype=np.uint8
            )
            combined_row = np.hstack([label, vertical_sep, img])
            rows.append(combined_row)

        if rows:
            # 添加行间分隔线
            separator = np.full(
                (row_sep_height, rows[0].shape[1], 3), row_sep_color, dtype=np.uint8
            )

            # 构建最终图像（行间插入分隔线）
            final_image = rows[0]
            for row in rows[1:]:
                final_image = np.vstack([final_image, separator, row])

            # 保存结果
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, final_image)
