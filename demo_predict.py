from ultralytics import YOLO

# 加载模型（这里先用官方预训练的yolov8n.pt做测试，后面可以换成你自己训练的模型）
model = YOLO("yolov8n.pt")

# 对图片进行预测
results = model("./ultralytics/assets/2.png")  # 用项目自带的示例图片测试

# 显示结果
results[0].show()