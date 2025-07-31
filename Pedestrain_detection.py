import os
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

transform = T.Compose([
    T.ToTensor()
])

dataset_dir = "PennFudanPed/PennFudanPed/PNGImages"

image_paths = [os.path.join(dataset_dir, fname) for fname in os.listdir(dataset_dir) if fname.endswith('.png')]

def visualize(img_tensor, boxes, scores, threshold=0.8):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(10, 7))
    ax.imshow(img)

    accurate_detections = 0
    
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            accurate_detections += 1
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'{scores[i]:.2f}', color='yellow', fontsize=10, backgroundcolor='black')
    
    total_detections = len(scores)
    accuracy = (accurate_detections / total_detections) * 100 if total_detections > 0 else 0

    plt.title(f"Detections above threshold: {accurate_detections}/{total_detections} ({accuracy:.2f}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

for img_path in image_paths[:5]:
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    print(f"Processing: {os.path.basename(img_path)}")
    print(f"Total Detections: {len(prediction['boxes'])}")
    visualize(img_tensor, prediction['boxes'], prediction['scores'])
