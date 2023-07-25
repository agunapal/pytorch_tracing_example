# pytorch_tracing_example

```
python trace.py
```

Logs

```
torch.float32
graph(%self.1 : __torch__.torchvision.models.resnet.ResNet,
      %x.1 : Float(1, 3, 224, 224, strides=[150528, 50176, 224, 1], requires_grad=0, device=cuda:0)):
  %fc : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="fc"](%self.1)
  %avgpool : __torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d = prim::GetAttr[name="avgpool"](%self.1)
  %layer4 : __torch__.torch.nn.modules.container.___torch_mangle_141.Sequential = prim::GetAttr[name="layer4"](%self.1)
  %layer3 : __torch__.torch.nn.modules.container.___torch_mangle_113.Sequential = prim::GetAttr[name="layer3"](%self.1)
  %layer2 : __torch__.torch.nn.modules.container.___torch_mangle_61.Sequential = prim::GetAttr[name="layer2"](%self.1)
  %layer1 : __torch__.torch.nn.modules.container.___torch_mangle_25.Sequential = prim::GetAttr[name="layer1"](%self.1)
  %maxpool : __torch__.torch.nn.modules.pooling.MaxPool2d = prim::GetAttr[name="maxpool"](%self.1)
  %relu.1 : __torch__.torch.nn.modules.activation.ReLU = prim::GetAttr[name="relu"](%self.1)
  %bn1.1 : __torch__.torch.nn.modules.batchnorm.BatchNorm2d = prim::GetAttr[name="bn1"](%self.1)
  %conv1.1 : __torch__.torch.nn.modules.conv.Conv2d = prim::GetAttr[name="conv1"](%self.1)
  %2995 : Tensor = prim::CallMethod[name="forward"](%conv1.1, %x.1)
  %2996 : Tensor = prim::CallMethod[name="forward"](%bn1.1, %2995)
  %2997 : Tensor = prim::CallMethod[name="forward"](%relu.1, %2996)
  %2998 : Tensor = prim::CallMethod[name="forward"](%maxpool, %2997)
  %2999 : Tensor = prim::CallMethod[name="forward"](%layer1, %2998)
  %3000 : Tensor = prim::CallMethod[name="forward"](%layer2, %2999)
  %3001 : Tensor = prim::CallMethod[name="forward"](%layer3, %3000)
  %3002 : Tensor = prim::CallMethod[name="forward"](%layer4, %3001)
  %3003 : Tensor = prim::CallMethod[name="forward"](%avgpool, %3002)
  %2210 : int = prim::Constant[value=1]() # /home/ubuntu/anaconda3/envs/torchserve/lib/python3.10/site-packages/torchvision/models/resnet.py:279:0
  %2211 : int = prim::Constant[value=-1]() # /home/ubuntu/anaconda3/envs/torchserve/lib/python3.10/site-packages/torchvision/models/resnet.py:279:0
  %input : Float(1, 2048, strides=[2048, 1], requires_grad=1, device=cuda:0) = aten::flatten(%3003, %2210, %2211) # /home/ubuntu/anaconda3/envs/torchserve/lib/python3.10/site-packages/torchvision/models/resnet.py:279:0
  %3004 : Tensor = prim::CallMethod[name="forward"](%fc, %input)
  return (%3004)

torch.float16
graph(%self.1 : __torch__.torchvision.models.resnet.___torch_mangle_443.ResNet,
      %x.1 : Half(1, 3, 224, 224, strides=[150528, 50176, 224, 1], requires_grad=0, device=cuda:0)):
  %fc : __torch__.torch.nn.modules.linear.___torch_mangle_442.Linear = prim::GetAttr[name="fc"](%self.1)
  %avgpool : __torch__.torch.nn.modules.pooling.___torch_mangle_441.AdaptiveAvgPool2d = prim::GetAttr[name="avgpool"](%self.1)
  %layer4 : __torch__.torch.nn.modules.container.___torch_mangle_440.Sequential = prim::GetAttr[name="layer4"](%self.1)
  %layer3 : __torch__.torch.nn.modules.container.___torch_mangle_412.Sequential = prim::GetAttr[name="layer3"](%self.1)
  %layer2 : __torch__.torch.nn.modules.container.___torch_mangle_360.Sequential = prim::GetAttr[name="layer2"](%self.1)
  %layer1 : __torch__.torch.nn.modules.container.___torch_mangle_324.Sequential = prim::GetAttr[name="layer1"](%self.1)
  %maxpool : __torch__.torch.nn.modules.pooling.___torch_mangle_296.MaxPool2d = prim::GetAttr[name="maxpool"](%self.1)
  %relu.1 : __torch__.torch.nn.modules.activation.___torch_mangle_295.ReLU = prim::GetAttr[name="relu"](%self.1)
  %bn1.1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_294.BatchNorm2d = prim::GetAttr[name="bn1"](%self.1)
  %conv1.1 : __torch__.torch.nn.modules.conv.___torch_mangle_293.Conv2d = prim::GetAttr[name="conv1"](%self.1)
  %2995 : Tensor = prim::CallMethod[name="forward"](%conv1.1, %x.1)
  %2996 : Tensor = prim::CallMethod[name="forward"](%bn1.1, %2995)
  %2997 : Tensor = prim::CallMethod[name="forward"](%relu.1, %2996)
  %2998 : Tensor = prim::CallMethod[name="forward"](%maxpool, %2997)
  %2999 : Tensor = prim::CallMethod[name="forward"](%layer1, %2998)
  %3000 : Tensor = prim::CallMethod[name="forward"](%layer2, %2999)
  %3001 : Tensor = prim::CallMethod[name="forward"](%layer3, %3000)
  %3002 : Tensor = prim::CallMethod[name="forward"](%layer4, %3001)
  %3003 : Tensor = prim::CallMethod[name="forward"](%avgpool, %3002)
  %2210 : int = prim::Constant[value=1]() # /home/ubuntu/anaconda3/envs/torchserve/lib/python3.10/site-packages/torchvision/models/resnet.py:279:0
  %2211 : int = prim::Constant[value=-1]() # /home/ubuntu/anaconda3/envs/torchserve/lib/python3.10/site-packages/torchvision/models/resnet.py:279:0
  %input : Half(1, 2048, strides=[2048, 1], requires_grad=1, device=cuda:0) = aten::flatten(%3003, %2210, %2211) # /home/ubuntu/anaconda3/envs/torchserve/lib/python3.10/site-packages/torchvision/models/resnet.py:279:0
  %3004 : Tensor = prim::CallMethod[name="forward"](%fc, %input)
  return (%3004)


```
