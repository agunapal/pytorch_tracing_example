import argparse
import torch
from torchvision.models import ResNet50_Weights, resnet50, vgg16, VGG16_Weights
import torch._dynamo.config
import logging

torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.output_code = True


def get_dtype(precision):
    if precision=="float32":
        dtype = torch.float32
    elif precision=="float16":
        dtype = torch.float16
    elif precision=="bfloat16":
        dtype = torch.bfloat16
    return dtype
    

def convert_model(method, device, precision):

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    model.to(device)

    dtype = get_dtype(precision)

    input_data = torch.randn((1,3,224, 224), dtype=dtype, device=device)

    if precision == "float16":
        model.half()
    elif precision == "bfloat16":
        model.bfloat16()

    if method == "trace":
        converted_model = torch.jit.trace(model, input_data)
    elif method == "script":
        converted_model = torch.jit.script(model)
    elif method == "compile":
        converted_model = torch.compile(model)

    output = converted_model(input_data)
    print(f"Output dtype is {output.dtype}")
    if method != "compile":
        print(converted_model.graph)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        help="Choose one of the methods: trace, script or compile (Default : trace)",
        default= "trace"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device name: Ex: cuda, cpu(Default : cuda:0)",
        default= "cuda:0"
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="Choose precision: float32, float16, bfloat16 (Default : float32)",
        default= "float32"
    )
    args = parser.parse_args()

    convert_model(args.method, args.device, args.precision)




