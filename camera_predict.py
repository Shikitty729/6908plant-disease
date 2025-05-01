import torch
import cv2
import io
import time
from PIL import Image
import IPython
from torchvision import transforms

# ----------------- 全局常量定义 -----------------
FONT = cv2.FONT_HERSHEY_SIMPLEX  # 文本字体
FONT_SCALE = 0.7                 # 文本缩放比例
COLOR = (0, 255, 0)              # 文本颜色 (BGR)：绿色
THICKNESS = 2                    # 文本线宽
PREDICTION_INTERVAL = 0.5        # 两次推理之间的最小时间间隔（秒）

# ----------------- 工具函数 -----------------
def show_array(a, fmt='jpeg'):
    """
    在 Jupyter Notebook 中显示 RGB 图像数组。

    参数:
      a   (np.ndarray): 要显示的 RGB 图像数组。
      fmt (str)      : 图像编码格式，'jpeg' 或 'png'。
    """
    f = io.BytesIO()                     # 创建字节流缓存
    Image.fromarray(a).save(f, fmt)      # 将数组保存到缓存
    display(IPython.display.Image(data=f.getvalue()))  # 在 notebook 中渲染

def initialize_model(model, weights_path, device):
    """
    加载模型权重并设置推理模式。

    参数:
      model        (torch.nn.Module): 未加载权重的模型实例。
      weights_path (str)            : 权重文件路径 (.pth / .pt)。
      device       (torch.device)   : 推理设备 (如 torch.device('cuda:0'))。

    返回:
      torch.nn.Module: 已加载权重并切换到 eval 模式的模型。
    """
    state_dict = torch.load(weights_path, map_location=device)  # 读入权重
    model.load_state_dict(state_dict)                          # 装载到模型
    model.to(device)                                            # 搬到指定设备
    model.eval()                                                # 切换到推理模式
    return model

# ----------------- 主函数 -----------------
def live_plant_disease_detection(model,
                                 weights_path,
                                 device,
                                 disease_names,
                                 cam_id=0):
    """
    在 Jupyter Notebook 中实时捕获摄像头画面，并做植物病害 Top-3 分类显示。

    参数:
      model         (torch.nn.Module): 未加载权重的模型实例。
      weights_path  (str)            : 模型权重文件路径。
      device        (torch.device)   : torch.device('cuda:0') 或 'cpu'。
      disease_names (List[str])      : 类别名称列表，索引对齐模型输出。
      cam_id        (int)            : 摄像头设备 ID，默认 0。
    """
    # 1. 加载权重并准备模型
    model = initialize_model(model, weights_path, device)

    # 2. 定义图像预处理 pipeline：BGR array → PIL → Resize → Tensor → Normalize
    preprocess = transforms.Compose([
        transforms.ToPILImage(),                  # 转为 PIL Image
        transforms.Resize((224, 224)),            # 缩放到模型输入大小
        transforms.ToTensor(),                    # 转为 [C,H,W] tensor, 值域 [0,1]
        transforms.Normalize(                     # ImageNet 风格归一化
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    # 3. 打开摄像头
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("开始实时植物病害检测，按 Ctrl+C 停止")
    last_time = time.time()  # 上次推理时间
    preds = []               # 存储当前 Top-3 预测结果

    try:
        while True:
            # 读取一帧 BGR 图
            ret, frame = cap.read()
            if not ret:
                print("帧读取失败，退出循环")
                break

            # 控制推理频率，避免每帧都跑模型
            now = time.time()
            if now - last_time >= PREDICTION_INTERVAL:
                # 4. 预处理并推理
                input_tensor = preprocess(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(input_tensor)               # 模型原始输出
                    probs  = torch.softmax(logits, dim=1)[0]   # 转为概率
                    top_p, top_i = torch.topk(probs, 3)        # Top-3
                    # 组合成 [(idx, name, prob), …]
                    preds = [
                        (idx.item(), disease_names[idx.item()], top_p[j].item())
                        for j, idx in enumerate(top_i)
                    ]
                last_time = now

            # 5. 将 BGR 转为 RGB，用于显示
            disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 6. 在图上叠加 Top-3 文本
            for i, (_, name, prob) in enumerate(preds):
                # 名称里可能有 "plant___disease" 格式，先拆分
                plant, cond = name.split('___') if '___' in name else (name, '')
                plant = plant.replace('_', ' ')
                cond  = cond.replace('_', ' ')
                text = f"{plant}: {cond} ({prob:.2f})"
                # 绘制文本
                cv2.putText(disp, text, (10, 30 + i*30),
                            FONT, FONT_SCALE, COLOR, THICKNESS)

            # 7. 显示到 Notebook
            show_array(disp)
            IPython.display.clear_output(wait=True)

    except KeyboardInterrupt:
        # 捕捉 Ctrl+C 打断
        print("实时检测已停止")
    finally:
        # 8. 释放摄像头资源
        cap.release()
