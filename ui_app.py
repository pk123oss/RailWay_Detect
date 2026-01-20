"""
统一可视化界面

功能：
- 在界面中上传一张图片
- 使用两个模型对同一张图片进行推理：
  1）铁轨像素级掩码分割模型（YOLO segment）
  2）入侵物体检测模型（YOLO detect）
- 在一张图上同时叠加：
  - 铁轨的颜色掩码区域
  - 入侵物体的检测框和类别标签
- 警报功能：检测物体是否入侵轨道
- 日志输出：保存检测信息
- 界面中同时展示原图和标注后的图像
"""

from pathlib import Path
from functools import lru_cache
from typing import Tuple, List, Dict
from datetime import datetime
import json

import cv2
import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / "detection_logs"
LOG_DIR.mkdir(exist_ok=True)


@lru_cache(maxsize=2)
def load_model(model_path: str) -> YOLO:
    """加载 YOLO 模型（带缓存）"""
    p = Path(model_path)
    if not p.is_file():
        raise FileNotFoundError(f"模型文件不存在：{p}")
    return YOLO(str(p))


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL"""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def check_intrusion(
    box_xyxy: np.ndarray, track_mask: np.ndarray, overlap_threshold: float = 0.1
) -> Tuple[bool, Dict]:
    """
    检测物体框是否入侵轨道掩码
    
    Args:
        box_xyxy: 检测框坐标 [x1, y1, x2, y2]
        track_mask: 轨道掩码 (h, w) uint8，255为轨道区域
        overlap_threshold: 重叠阈值，框内轨道像素占比超过此值认为入侵
        
    Returns:
        (是否入侵, 详细信息字典)
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    h, w = track_mask.shape
    
    # 确保坐标在图像范围内
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    # 提取检测框区域
    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
    
    # 计算重叠区域
    overlap_mask = cv2.bitwise_and(box_mask, track_mask)
    overlap_pixels = np.sum(overlap_mask > 0)
    box_pixels = (x2 - x1) * (y2 - y1)
    
    # 计算重叠比例
    overlap_ratio = overlap_pixels / box_pixels if box_pixels > 0 else 0.0
    
    # 检测边界接触（检测框边界是否与掩码有接触）
    # 创建检测框边界
    border_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(border_mask, (x1, y1), (x2, y2), 255, 1)  # 只绘制边界
    border_contact = np.sum(cv2.bitwise_and(border_mask, track_mask) > 0) > 0
    
    # 判断是否入侵
    is_intrusion = overlap_ratio >= overlap_threshold or border_contact
    
    info = {
        "overlap_ratio": float(overlap_ratio),
        "overlap_pixels": int(overlap_pixels),
        "box_pixels": int(box_pixels),
        "border_contact": bool(border_contact),
        "box_area": [int(x1), int(y1), int(x2), int(y2)],
    }
    
    return is_intrusion, info


def save_detection_log(
    image_path: str,
    track_mask_info: Dict,
    detections: List[Dict],
    intrusions: List[Dict],
    timestamp: str = None,
) -> str:
    """
    保存检测日志到文件
    
    Args:
        image_path: 图像路径
        track_mask_info: 轨道掩码信息
        detections: 所有检测结果列表
        intrusions: 入侵检测结果列表
        timestamp: 时间戳（可选）
        
    Returns:
        日志文件路径
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_data = {
        "timestamp": timestamp,
        "image_path": str(image_path),
        "track_mask": track_mask_info,
        "detections": detections,
        "intrusions": intrusions,
        "has_intrusion": len(intrusions) > 0,
    }
    
    # 保存JSON格式
    log_file_json = LOG_DIR / f"detection_{timestamp}.json"
    with open(log_file_json, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    # 保存文本格式（人类可读）
    log_file_txt = LOG_DIR / f"detection_{timestamp}.txt"
    with open(log_file_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("铁路场景检测日志\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"图像路径: {image_path}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("轨道掩码信息\n")
        f.write("-" * 60 + "\n")
        f.write(f"掩码区域像素数: {track_mask_info.get('mask_pixels', 0)}\n")
        f.write(f"图像总像素数: {track_mask_info.get('image_pixels', 0)}\n")
        f.write(f"掩码覆盖率: {track_mask_info.get('coverage', 0):.2f}%\n\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"物体检测结果 (共 {len(detections)} 个)\n")
        f.write("-" * 60 + "\n")
        for i, det in enumerate(detections, 1):
            f.write(f"\n检测对象 {i}:\n")
            f.write(f"  类别: {det['class_name']}\n")
            f.write(f"  置信度: {det['confidence']:.4f}\n")
            f.write(f"  检测框: [{det['box'][0]}, {det['box'][1]}, {det['box'][2]}, {det['box'][3]}]\n")
            f.write(f"  检测框面积: {det['box_area']} 像素\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("入侵检测结果\n")
        f.write("=" * 60 + "\n")
        if intrusions:
            f.write(f"⚠️  检测到 {len(intrusions)} 个入侵物体！\n\n")
            for i, intr in enumerate(intrusions, 1):
                f.write(f"入侵对象 {i}:\n")
                f.write(f"  类别: {intr['class_name']}\n")
                f.write(f"  置信度: {intr['confidence']:.4f}\n")
                f.write(f"  检测框: {intr['box']}\n")
                f.write(f"  重叠比例: {intr['overlap_ratio']:.4f} ({intr['overlap_ratio']*100:.2f}%)\n")
                f.write(f"  重叠像素数: {intr['overlap_pixels']}\n")
                f.write(f"  边界接触: {'是' if intr['border_contact'] else '否'}\n")
                f.write(f"  入侵原因: {'重叠区域超过阈值' if intr['overlap_ratio'] >= 0.1 else '边界接触'}\n\n")
        else:
            f.write("✓ 未检测到入侵物体\n")
    
    return str(log_file_txt)


def run_inference_with_details(
    image: Image.Image,
    track_model_path: str,
    intrude_model_path: str,
    track_conf: float = 0.25,
    intrude_conf: float = 0.25,
    overlap_threshold: float = 0.1,
) -> Tuple[Image.Image, Image.Image, str, Dict]:
    """
    与 run_inference 同款，但返回结构化 details，方便外部评测/统计。

    Returns:
        (orig_pil, vis_pil, alert_text, details)
        details: {
          track_mask_info, detections, intrusions, log_path
        }
    """
    # 直接调用 run_inference 的核心逻辑（复制关键部分），避免解析文本
    if image is None:
        raise gr.Error("请先上传一张图片。")

    img_bgr = pil_to_bgr(image)
    h, w = img_bgr.shape[:2]

    vis_bgr = img_bgr.copy()
    track_mask = np.zeros((h, w), dtype=np.uint8)
    track_mask_info = {"mask_pixels": 0, "image_pixels": h * w, "coverage": 0.0}
    detections: List[Dict] = []
    intrusions: List[Dict] = []

    # track
    if track_model_path:
        track_model = load_model(track_model_path)
        results = track_model.predict(source=img_bgr, conf=track_conf, verbose=False)
        result = results[0]
        if getattr(result, "masks", None) is not None and hasattr(result.masks, "data"):
            for mask_tensor in result.masks.data:
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                track_mask[mask_resized > 0.5] = 255

            mask_pixels = np.sum(track_mask > 0)
            track_mask_info = {
                "mask_pixels": int(mask_pixels),
                "image_pixels": int(h * w),
                "coverage": float(mask_pixels / (h * w) * 100),
            }

            mask_color = np.zeros_like(vis_bgr)
            mask_color[track_mask == 255] = (0, 255, 0)
            vis_bgr = cv2.addWeighted(vis_bgr, 0.7, mask_color, 0.3, 0)

    # detect
    if intrude_model_path:
        intrude_model = load_model(intrude_model_path)
        det_results = intrude_model.predict(source=img_bgr, conf=intrude_conf, verbose=False)
        det = det_results[0]
        if getattr(det, "boxes", None) is not None:
            names = intrude_model.names
            for box in det.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else (
                    names[cls_id]
                    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names)
                    else str(cls_id)
                )

                box_area = (x2 - x1) * (y2 - y1)
                det_info = {
                    "class_name": cls_name,
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "box_area": int(box_area),
                }
                detections.append(det_info)

                is_intrusion = False
                intrusion_info = {}
                if np.sum(track_mask > 0) > 0:
                    is_intrusion, intrusion_info = check_intrusion(
                        xyxy, track_mask, overlap_threshold=overlap_threshold
                    )

                if is_intrusion:
                    cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    intrusion_record = {**det_info, **intrusion_info}
                    intrusions.append(intrusion_record)
                else:
                    cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 保存日志（与 run_inference 相同目录）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = save_detection_log("dataset_image", track_mask_info, detections, intrusions, timestamp)

    # 生成警报文本（简版）
    alert_text = (
        f"轨道覆盖率: {track_mask_info['coverage']:.2f}% | "
        f"det={len(detections)} | intr={len(intrusions)} | log={log_path}"
    )
    if intrusions:
        alert_text = "⚠️ 入侵警报！" + "\n" + alert_text

    details = {
        "track_mask_info": track_mask_info,
        "detections": detections,
        "intrusions": intrusions,
        "log_path": log_path,
    }

    return image, bgr_to_pil(vis_bgr), alert_text, details


def run_inference(
    image: Image.Image,
    track_model_path: str,
    intrude_model_path: str,
    track_conf: float = 0.25,
    intrude_conf: float = 0.25,
) -> Tuple[Image.Image, Image.Image, str]:
    """
    对单张图像运行两个模型并可视化结果

    Args:
        image: 输入图像（PIL）
        track_model_path: 铁轨分割模型路径（segment）
        intrude_model_path: 入侵物体检测模型路径（detect）
        track_conf: 铁轨分割置信度阈值
        intrude_conf: 入侵物体检测置信度阈值
    Returns:
        (原图, 叠加了掩码和检测框的图像, 警报信息文本)
    """
    if image is None:
        raise gr.Error("请先上传一张图片。")

    # 转为 OpenCV BGR
    img_bgr = pil_to_bgr(image)
    h, w = img_bgr.shape[:2]

    vis_bgr = img_bgr.copy()
    track_mask = np.zeros((h, w), dtype=np.uint8)
    track_mask_info = {"mask_pixels": 0, "image_pixels": h * w, "coverage": 0.0}
    detections = []
    intrusions = []

    # 1. 铁轨像素级掩码（segment 模型）
    if track_model_path:
        try:
            track_model = load_model(track_model_path)
        except Exception as e:
            raise gr.Error(f"加载铁轨分割模型失败：{e}")

        # 运行分割推理
        results = track_model.predict(
            source=img_bgr,
            conf=track_conf,
            verbose=False,
        )
        result = results[0]

        if getattr(result, "masks", None) is not None and hasattr(
            result.masks, "data"
        ):
            # 遍历所有 mask，将其合并为一个整体轨道区域
            for mask_tensor in result.masks.data:
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np, (w, h), interpolation=cv2.INTER_NEAREST
                )
                track_mask[mask_resized > 0.5] = 255

            # 计算掩码信息
            mask_pixels = np.sum(track_mask > 0)
            track_mask_info = {
                "mask_pixels": int(mask_pixels),
                "image_pixels": int(h * w),
                "coverage": float(mask_pixels / (h * w) * 100),
            }

            # 将掩码渲染为绿色半透明区域
            mask_color = np.zeros_like(vis_bgr)
            mask_color[track_mask == 255] = (0, 255, 0)  # BGR 绿色
            vis_bgr = cv2.addWeighted(vis_bgr, 0.7, mask_color, 0.3, 0)

        else:
            # 如果模型不是分割模型，则给出提示信息但不中断流程
            print(
                f"[提示] 铁轨模型未输出 masks（task={getattr(track_model, 'task', None)}），"
                "可能是检测模型或未检测到铁轨区域。"
            )

    # 2. 入侵物体检测（detect 模型）
    if intrude_model_path:
        try:
            intrude_model = load_model(intrude_model_path)
        except Exception as e:
            raise gr.Error(f"加载入侵物体检测模型失败：{e}")

        det_results = intrude_model.predict(
            source=img_bgr,
            conf=intrude_conf,
            verbose=False,
        )
        det = det_results[0]

        if getattr(det, "boxes", None) is not None:
            names = intrude_model.names
            for box in det.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(
                    names, dict
                ) else (
                    names[cls_id]
                    if isinstance(names, (list, tuple))
                    and 0 <= cls_id < len(names)
                    else str(cls_id)
                )

                # 记录检测信息
                box_area = (x2 - x1) * (y2 - y1)
                det_info = {
                    "class_name": cls_name,
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "box_area": int(box_area),
                }
                detections.append(det_info)

                # 检测是否入侵轨道
                if np.sum(track_mask > 0) > 0:  # 如果有轨道掩码
                    is_intrusion, intrusion_info = check_intrusion(
                        xyxy, track_mask, overlap_threshold=0.1
                    )

                    if is_intrusion:
                        # 入侵物体：使用更粗的红色框和警告标记
                        cv2.rectangle(
                            vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 4
                        )  # 更粗的红色框

                        # 添加警告文字
                        warning_label = f"⚠ INTRUSION: {cls_name} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(
                            warning_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            vis_bgr,
                            (x1, y1 - th - 8),
                            (x1 + tw + 4, y1),
                            (0, 0, 255),
                            -1,
                        )
                        cv2.putText(
                            vis_bgr,
                            warning_label,
                            (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                            lineType=cv2.LINE_AA,
                        )

                        # 记录入侵信息
                        intrusion_record = {
                            **det_info,
                            **intrusion_info,
                        }
                        intrusions.append(intrusion_record)
                    else:
                        # 非入侵物体：正常红色框
                        cv2.rectangle(
                            vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2
                        )

                        label = f"{cls_name} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            vis_bgr,
                            (x1, y1 - th - 4),
                            (x1 + tw + 2, y1),
                            (0, 0, 255),
                            -1,
                        )
                        cv2.putText(
                            vis_bgr,
                            label,
                            (x1 + 1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            lineType=cv2.LINE_AA,
                        )
                else:
                    # 没有轨道掩码，正常绘制
                    cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{cls_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        vis_bgr, (x1, y1 - th - 4), (x1 + tw + 2, y1), (0, 0, 255), -1
                    )
                    cv2.putText(
                        vis_bgr,
                        label,
                        (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        lineType=cv2.LINE_AA,
                    )

    # 保存日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = save_detection_log(
        "uploaded_image", track_mask_info, detections, intrusions, timestamp
    )

    # 生成警报信息文本
    alert_text = f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    alert_text += f"轨道掩码覆盖率: {track_mask_info['coverage']:.2f}%\n"
    alert_text += f"检测到物体数量: {len(detections)}\n"
    alert_text += f"入侵物体数量: {len(intrusions)}\n\n"

    if intrusions:
        alert_text += "⚠️ 警报：检测到入侵物体！\n\n"
        for i, intr in enumerate(intrusions, 1):
            alert_text += f"入侵对象 {i}:\n"
            alert_text += f"  类别: {intr['class_name']}\n"
            alert_text += f"  置信度: {intr['confidence']:.4f}\n"
            alert_text += f"  重叠比例: {intr['overlap_ratio']*100:.2f}%\n"
            alert_text += f"  边界接触: {'是' if intr['border_contact'] else '否'}\n\n"
    else:
        alert_text += "✓ 未检测到入侵物体\n"

    alert_text += f"\n日志已保存: {log_path}"

    # 转回 PIL 以便 Gradio 显示
    orig_pil = image
    vis_pil = bgr_to_pil(vis_bgr)
    return orig_pil, vis_pil, alert_text


def build_interface() -> gr.Blocks:
    """构建 Gradio 界面"""
    default_track_model = str(
        (PROJECT_ROOT / "railway_track_detection" / "best.pt").resolve()
    )
    default_intrude_model = str(
        (PROJECT_ROOT / "runs" / "train" / "weights" / "best.pt").resolve()
    )

    with gr.Blocks(title="铁路场景综合检测可视化") as demo:
        gr.Markdown(
            """
### 铁路场景综合检测可视化与入侵警报系统

- **输入**：一张铁路场景图片
- **输出**：
  - 左侧：原始图像
  - 右侧：叠加了
    - 铁轨像素级掩码区域（绿色半透明）
    - 入侵物体检测框及类别标签（红色，入侵物体用粗框+警告标记）
  - 检测结果：显示检测信息和入侵警报
- **警报功能**：自动检测物体是否入侵轨道（重叠或边界接触）
- **日志保存**：自动保存检测日志到 `detection_logs/` 目录
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="上传图片（铁路场景）", type="pil"
                )

                track_model_box = gr.Textbox(
                    label="铁轨分割模型路径（YOLO segment）",
                    value=default_track_model,
                )
                intrude_model_box = gr.Textbox(
                    label="入侵物体检测模型路径（YOLO detect）",
                    value=default_intrude_model,
                )

                with gr.Row():
                    track_conf_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="铁轨分割置信度阈值",
                    )
                    intrude_conf_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="入侵物体检测置信度阈值",
                    )

                run_btn = gr.Button("运行检测与识别", variant="primary")

            with gr.Column(scale=1):
                orig_image = gr.Image(
                    label="原始图像", interactive=False
                )
                vis_image = gr.Image(
                    label="标注后图像（铁轨掩码 + 入侵物体检测）",
                    interactive=False,
                )
                alert_output = gr.Textbox(
                    label="检测结果与警报信息",
                    lines=15,
                    interactive=False,
                    value="等待检测...",
                )

        run_btn.click(
            fn=run_inference,
            inputs=[
                image_input,
                track_model_box,
                intrude_model_box,
                track_conf_slider,
                intrude_conf_slider,
            ],
            outputs=[orig_image, vis_image, alert_output],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    # 本地运行：默认 http://127.0.0.1:7860
    app.launch()



