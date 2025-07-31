Pedestrian Detection using PyTorch Faster R-CNN:

  This project detects pedestrians in images using PyTorch’s pre-trained Faster R-CNN model with a ResNet-50 FPN backbone. It loads images from the Penn-Fudan Pedestrian Dataset, performs object detection, and visualizes results with bounding boxes and confidence scores.

Features

- Uses pretrained `fasterrcnn_resnet50_fpn` from `torchvision`
- Detects pedestrians in images with high accuracy
- Filters predictions using a confidence threshold (default: 0.8)
- Visualizes bounding boxes with matplotlib

Dataset

- Name: Penn-Fudan Pedestrian Dataset  
- Source: [https://www.cis.upenn.edu/~jshi/ped_html/](https://www.cis.upenn.edu/~jshi/ped_html/)  

