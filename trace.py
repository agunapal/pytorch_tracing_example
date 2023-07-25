ifrom torchvision.models import ResNet50_Weights, resnet50, vgg16, VGG16_Weights
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()
model.to(device)

input_data = torch.randn((1,3,224, 224), device=device)

output = model(input_data)
traced_model = torch.jit.trace(model, input_data)
output = traced_model(input_data)
print(output.dtype)
print(traced_model.graph)

#Change to half precision
model.half()
input_data = torch.randn((1,3,224, 224), dtype=torch.float16, device=device)
traced_model = torch.jit.trace(model, input_data)

output = traced_model(input_data)
print(output.dtype)
print(traced_model.graph)

