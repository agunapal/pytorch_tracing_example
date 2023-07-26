# pytorch_tracing_example

## Tracing
```
python convert_model.py --precision bfloat16 --method trace --device cpu
```

Logs

```
Output dtype is torch.bfloat16
graph(%self.1 : __torch__.torchvision.models.resnet.ResNet,
      %x.1 : BFloat16(1, 3, 224, 224, strides=[150528, 50176, 224, 1], requires_grad=0, device=cpu)):
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
  %input : BFloat16(1, 2048, strides=[2048, 1], requires_grad=1, device=cpu) = aten::flatten(%3003, %2210, %2211) # /home/ubuntu/anaconda3/envs/torchserve/lib/python3.10/site-packages/torchvision/models/resnet.py:279:0
  %3004 : Tensor = prim::CallMethod[name="forward"](%fc, %input)
  return (%3004)

```

## Scripting

```
python convert_model.py --precision float16 --method script --device cuda:0
```

Logs

```
Output dtype is torch.float16
graph(%self : __torch__.torchvision.models.resnet.ResNet,
      %x.1 : Tensor):
  %3 : Tensor = prim::CallMethod[name="_forward_impl"](%self, %x.1) # /home/ubuntu/anaconda3/envs/torchserve/lib/python3.10/site-packages/torchvision/models/resnet.py:285:15
  return (%3)
```

## Compile
```
python convert_model.py --precision bfloat16 --method compile --device cpu
```

Logs

```
[2023-07-26 17:47:10,325] torch._dynamo.output_graph: [INFO] TRACED GRAPH
 __compiled_fn_0 <eval_with_key>.5 opcode         name                        target                                                      args                                             kwargs
-------------  --------------------------  ----------------------------------------------------------  -----------------------------------------------  --------
placeholder    x                           x                                                           ()                                               {}
call_module    self_conv1                  self_conv1                                                  (x,)                                             {}
call_module    self_bn1                    self_bn1                                                    (self_conv1,)                                    {}
call_module    self_relu                   self_relu                                                   (self_bn1,)                                      {}
call_module    self_maxpool                self_maxpool                                                (self_relu,)                                     {}
call_module    self_layer1_0_conv1         self_layer1_0_conv1                                         (self_maxpool,)                                  {}
call_module    self_layer1_0_bn1           self_layer1_0_bn1                                           (self_layer1_0_conv1,)                           {}
call_module    self_layer1_0_relu          self_layer1_0_relu                                          (self_layer1_0_bn1,)                             {}
call_module    self_layer1_0_conv2         self_layer1_0_conv2                                         (self_layer1_0_relu,)                            {}
call_module    self_layer1_0_bn2           self_layer1_0_bn2                                           (self_layer1_0_conv2,)                           {}
call_module    self_layer1_0_relu_1        self_layer1_0_relu                                          (self_layer1_0_bn2,)                             {}
call_module    self_layer1_0_conv3         self_layer1_0_conv3                                         (self_layer1_0_relu_1,)                          {}
call_module    self_layer1_0_bn3           self_layer1_0_bn3                                           (self_layer1_0_conv3,)                           {}
call_module    self_layer1_0_downsample_0  self_layer1_0_downsample_0                                  (self_maxpool,)                                  {}
call_module    self_layer1_0_downsample_1  self_layer1_0_downsample_1                                  (self_layer1_0_downsample_0,)                    {}
call_function  iadd                        <built-in function iadd>                                    (self_layer1_0_bn3, self_layer1_0_downsample_1)  {}
call_module    self_layer1_0_relu_2        self_layer1_0_relu                                          (iadd,)                                          {}
call_module    self_layer1_1_conv1         self_layer1_1_conv1                                         (self_layer1_0_relu_2,)                          {}
call_module    self_layer1_1_bn1           self_layer1_1_bn1                                           (self_layer1_1_conv1,)                           {}
call_module    self_layer1_1_relu          self_layer1_1_relu                                          (self_layer1_1_bn1,)                             {}
call_module    self_layer1_1_conv2         self_layer1_1_conv2                                         (self_layer1_1_relu,)                            {}
call_module    self_layer1_1_bn2           self_layer1_1_bn2                                           (self_layer1_1_conv2,)                           {}
call_module    self_layer1_1_relu_1        self_layer1_1_relu                                          (self_layer1_1_bn2,)                             {}
call_module    self_layer1_1_conv3         self_layer1_1_conv3                                         (self_layer1_1_relu_1,)                          {}
call_module    self_layer1_1_bn3           self_layer1_1_bn3                                           (self_layer1_1_conv3,)                           {}
call_function  iadd_1                      <built-in function iadd>                                    (self_layer1_1_bn3, self_layer1_0_relu_2)        {}
call_module    self_layer1_1_relu_2        self_layer1_1_relu                                          (iadd_1,)                                        {}
call_module    self_layer1_2_conv1         self_layer1_2_conv1                                         (self_layer1_1_relu_2,)                          {}
call_module    self_layer1_2_bn1           self_layer1_2_bn1                                           (self_layer1_2_conv1,)                           {}
call_module    self_layer1_2_relu          self_layer1_2_relu                                          (self_layer1_2_bn1,)                             {}
call_module    self_layer1_2_conv2         self_layer1_2_conv2                                         (self_layer1_2_relu,)                            {}
call_module    self_layer1_2_bn2           self_layer1_2_bn2                                           (self_layer1_2_conv2,)                           {}
call_module    self_layer1_2_relu_1        self_layer1_2_relu                                          (self_layer1_2_bn2,)                             {}
call_module    self_layer1_2_conv3         self_layer1_2_conv3                                         (self_layer1_2_relu_1,)                          {}
call_module    self_layer1_2_bn3           self_layer1_2_bn3                                           (self_layer1_2_conv3,)                           {}
call_function  iadd_2                      <built-in function iadd>                                    (self_layer1_2_bn3, self_layer1_1_relu_2)        {}
call_module    self_layer1_2_relu_2        self_layer1_2_relu                                          (iadd_2,)                                        {}
call_module    self_layer2_0_conv1         self_layer2_0_conv1                                         (self_layer1_2_relu_2,)                          {}
call_module    self_layer2_0_bn1           self_layer2_0_bn1                                           (self_layer2_0_conv1,)                           {}
call_module    self_layer2_0_relu          self_layer2_0_relu                                          (self_layer2_0_bn1,)                             {}
call_module    self_layer2_0_conv2         self_layer2_0_conv2                                         (self_layer2_0_relu,)                            {}
call_module    self_layer2_0_bn2           self_layer2_0_bn2                                           (self_layer2_0_conv2,)                           {}
call_module    self_layer2_0_relu_1        self_layer2_0_relu                                          (self_layer2_0_bn2,)                             {}
call_module    self_layer2_0_conv3         self_layer2_0_conv3                                         (self_layer2_0_relu_1,)                          {}
call_module    self_layer2_0_bn3           self_layer2_0_bn3                                           (self_layer2_0_conv3,)                           {}
call_module    self_layer2_0_downsample_0  self_layer2_0_downsample_0                                  (self_layer1_2_relu_2,)                          {}
call_module    self_layer2_0_downsample_1  self_layer2_0_downsample_1                                  (self_layer2_0_downsample_0,)                    {}
call_function  iadd_3                      <built-in function iadd>                                    (self_layer2_0_bn3, self_layer2_0_downsample_1)  {}
call_module    self_layer2_0_relu_2        self_layer2_0_relu                                          (iadd_3,)                                        {}
call_module    self_layer2_1_conv1         self_layer2_1_conv1                                         (self_layer2_0_relu_2,)                          {}
call_module    self_layer2_1_bn1           self_layer2_1_bn1                                           (self_layer2_1_conv1,)                           {}
call_module    self_layer2_1_relu          self_layer2_1_relu                                          (self_layer2_1_bn1,)                             {}
call_module    self_layer2_1_conv2         self_layer2_1_conv2                                         (self_layer2_1_relu,)                            {}
call_module    self_layer2_1_bn2           self_layer2_1_bn2                                           (self_layer2_1_conv2,)                           {}
call_module    self_layer2_1_relu_1        self_layer2_1_relu                                          (self_layer2_1_bn2,)                             {}
call_module    self_layer2_1_conv3         self_layer2_1_conv3                                         (self_layer2_1_relu_1,)                          {}
call_module    self_layer2_1_bn3           self_layer2_1_bn3                                           (self_layer2_1_conv3,)                           {}
call_function  iadd_4                      <built-in function iadd>                                    (self_layer2_1_bn3, self_layer2_0_relu_2)        {}
call_module    self_layer2_1_relu_2        self_layer2_1_relu                                          (iadd_4,)                                        {}
call_module    self_layer2_2_conv1         self_layer2_2_conv1                                         (self_layer2_1_relu_2,)                          {}
call_module    self_layer2_2_bn1           self_layer2_2_bn1                                           (self_layer2_2_conv1,)                           {}
call_module    self_layer2_2_relu          self_layer2_2_relu                                          (self_layer2_2_bn1,)                             {}
call_module    self_layer2_2_conv2         self_layer2_2_conv2                                         (self_layer2_2_relu,)                            {}
call_module    self_layer2_2_bn2           self_layer2_2_bn2                                           (self_layer2_2_conv2,)                           {}
call_module    self_layer2_2_relu_1        self_layer2_2_relu                                          (self_layer2_2_bn2,)                             {}
call_module    self_layer2_2_conv3         self_layer2_2_conv3                                         (self_layer2_2_relu_1,)                          {}
call_module    self_layer2_2_bn3           self_layer2_2_bn3                                           (self_layer2_2_conv3,)                           {}
call_function  iadd_5                      <built-in function iadd>                                    (self_layer2_2_bn3, self_layer2_1_relu_2)        {}
call_module    self_layer2_2_relu_2        self_layer2_2_relu                                          (iadd_5,)                                        {}
call_module    self_layer2_3_conv1         self_layer2_3_conv1                                         (self_layer2_2_relu_2,)                          {}
call_module    self_layer2_3_bn1           self_layer2_3_bn1                                           (self_layer2_3_conv1,)                           {}
call_module    self_layer2_3_relu          self_layer2_3_relu                                          (self_layer2_3_bn1,)                             {}
call_module    self_layer2_3_conv2         self_layer2_3_conv2                                         (self_layer2_3_relu,)                            {}
call_module    self_layer2_3_bn2           self_layer2_3_bn2                                           (self_layer2_3_conv2,)                           {}
call_module    self_layer2_3_relu_1        self_layer2_3_relu                                          (self_layer2_3_bn2,)                             {}
call_module    self_layer2_3_conv3         self_layer2_3_conv3                                         (self_layer2_3_relu_1,)                          {}
call_module    self_layer2_3_bn3           self_layer2_3_bn3                                           (self_layer2_3_conv3,)                           {}
call_function  iadd_6                      <built-in function iadd>                                    (self_layer2_3_bn3, self_layer2_2_relu_2)        {}
call_module    self_layer2_3_relu_2        self_layer2_3_relu                                          (iadd_6,)                                        {}
call_module    self_layer3_0_conv1         self_layer3_0_conv1                                         (self_layer2_3_relu_2,)                          {}
call_module    self_layer3_0_bn1           self_layer3_0_bn1                                           (self_layer3_0_conv1,)                           {}
call_module    self_layer3_0_relu          self_layer3_0_relu                                          (self_layer3_0_bn1,)                             {}
call_module    self_layer3_0_conv2         self_layer3_0_conv2                                         (self_layer3_0_relu,)                            {}
call_module    self_layer3_0_bn2           self_layer3_0_bn2                                           (self_layer3_0_conv2,)                           {}
call_module    self_layer3_0_relu_1        self_layer3_0_relu                                          (self_layer3_0_bn2,)                             {}
call_module    self_layer3_0_conv3         self_layer3_0_conv3                                         (self_layer3_0_relu_1,)                          {}
call_module    self_layer3_0_bn3           self_layer3_0_bn3                                           (self_layer3_0_conv3,)                           {}
call_module    self_layer3_0_downsample_0  self_layer3_0_downsample_0                                  (self_layer2_3_relu_2,)                          {}
call_module    self_layer3_0_downsample_1  self_layer3_0_downsample_1                                  (self_layer3_0_downsample_0,)                    {}
call_function  iadd_7                      <built-in function iadd>                                    (self_layer3_0_bn3, self_layer3_0_downsample_1)  {}
call_module    self_layer3_0_relu_2        self_layer3_0_relu                                          (iadd_7,)                                        {}
call_module    self_layer3_1_conv1         self_layer3_1_conv1                                         (self_layer3_0_relu_2,)                          {}
call_module    self_layer3_1_bn1           self_layer3_1_bn1                                           (self_layer3_1_conv1,)                           {}
call_module    self_layer3_1_relu          self_layer3_1_relu                                          (self_layer3_1_bn1,)                             {}
call_module    self_layer3_1_conv2         self_layer3_1_conv2                                         (self_layer3_1_relu,)                            {}
call_module    self_layer3_1_bn2           self_layer3_1_bn2                                           (self_layer3_1_conv2,)                           {}
call_module    self_layer3_1_relu_1        self_layer3_1_relu                                          (self_layer3_1_bn2,)                             {}
call_module    self_layer3_1_conv3         self_layer3_1_conv3                                         (self_layer3_1_relu_1,)                          {}
call_module    self_layer3_1_bn3           self_layer3_1_bn3                                           (self_layer3_1_conv3,)                           {}
call_function  iadd_8                      <built-in function iadd>                                    (self_layer3_1_bn3, self_layer3_0_relu_2)        {}
call_module    self_layer3_1_relu_2        self_layer3_1_relu                                          (iadd_8,)                                        {}
call_module    self_layer3_2_conv1         self_layer3_2_conv1                                         (self_layer3_1_relu_2,)                          {}
call_module    self_layer3_2_bn1           self_layer3_2_bn1                                           (self_layer3_2_conv1,)                           {}
call_module    self_layer3_2_relu          self_layer3_2_relu                                          (self_layer3_2_bn1,)                             {}
call_module    self_layer3_2_conv2         self_layer3_2_conv2                                         (self_layer3_2_relu,)                            {}
call_module    self_layer3_2_bn2           self_layer3_2_bn2                                           (self_layer3_2_conv2,)                           {}
call_module    self_layer3_2_relu_1        self_layer3_2_relu                                          (self_layer3_2_bn2,)                             {}
call_module    self_layer3_2_conv3         self_layer3_2_conv3                                         (self_layer3_2_relu_1,)                          {}
call_module    self_layer3_2_bn3           self_layer3_2_bn3                                           (self_layer3_2_conv3,)                           {}
call_function  iadd_9                      <built-in function iadd>                                    (self_layer3_2_bn3, self_layer3_1_relu_2)        {}
call_module    self_layer3_2_relu_2        self_layer3_2_relu                                          (iadd_9,)                                        {}
call_module    self_layer3_3_conv1         self_layer3_3_conv1                                         (self_layer3_2_relu_2,)                          {}
call_module    self_layer3_3_bn1           self_layer3_3_bn1                                           (self_layer3_3_conv1,)                           {}
call_module    self_layer3_3_relu          self_layer3_3_relu                                          (self_layer3_3_bn1,)                             {}
call_module    self_layer3_3_conv2         self_layer3_3_conv2                                         (self_layer3_3_relu,)                            {}
call_module    self_layer3_3_bn2           self_layer3_3_bn2                                           (self_layer3_3_conv2,)                           {}
call_module    self_layer3_3_relu_1        self_layer3_3_relu                                          (self_layer3_3_bn2,)                             {}
call_module    self_layer3_3_conv3         self_layer3_3_conv3                                         (self_layer3_3_relu_1,)                          {}
call_module    self_layer3_3_bn3           self_layer3_3_bn3                                           (self_layer3_3_conv3,)                           {}
call_function  iadd_10                     <built-in function iadd>                                    (self_layer3_3_bn3, self_layer3_2_relu_2)        {}
call_module    self_layer3_3_relu_2        self_layer3_3_relu                                          (iadd_10,)                                       {}
call_module    self_layer3_4_conv1         self_layer3_4_conv1                                         (self_layer3_3_relu_2,)                          {}
call_module    self_layer3_4_bn1           self_layer3_4_bn1                                           (self_layer3_4_conv1,)                           {}
call_module    self_layer3_4_relu          self_layer3_4_relu                                          (self_layer3_4_bn1,)                             {}
call_module    self_layer3_4_conv2         self_layer3_4_conv2                                         (self_layer3_4_relu,)                            {}
call_module    self_layer3_4_bn2           self_layer3_4_bn2                                           (self_layer3_4_conv2,)                           {}
call_module    self_layer3_4_relu_1        self_layer3_4_relu                                          (self_layer3_4_bn2,)                             {}
call_module    self_layer3_4_conv3         self_layer3_4_conv3                                         (self_layer3_4_relu_1,)                          {}
call_module    self_layer3_4_bn3           self_layer3_4_bn3                                           (self_layer3_4_conv3,)                           {}
call_function  iadd_11                     <built-in function iadd>                                    (self_layer3_4_bn3, self_layer3_3_relu_2)        {}
call_module    self_layer3_4_relu_2        self_layer3_4_relu                                          (iadd_11,)                                       {}
call_module    self_layer3_5_conv1         self_layer3_5_conv1                                         (self_layer3_4_relu_2,)                          {}
call_module    self_layer3_5_bn1           self_layer3_5_bn1                                           (self_layer3_5_conv1,)                           {}
call_module    self_layer3_5_relu          self_layer3_5_relu                                          (self_layer3_5_bn1,)                             {}
call_module    self_layer3_5_conv2         self_layer3_5_conv2                                         (self_layer3_5_relu,)                            {}
call_module    self_layer3_5_bn2           self_layer3_5_bn2                                           (self_layer3_5_conv2,)                           {}
call_module    self_layer3_5_relu_1        self_layer3_5_relu                                          (self_layer3_5_bn2,)                             {}
call_module    self_layer3_5_conv3         self_layer3_5_conv3                                         (self_layer3_5_relu_1,)                          {}
call_module    self_layer3_5_bn3           self_layer3_5_bn3                                           (self_layer3_5_conv3,)                           {}
call_function  iadd_12                     <built-in function iadd>                                    (self_layer3_5_bn3, self_layer3_4_relu_2)        {}
call_module    self_layer3_5_relu_2        self_layer3_5_relu                                          (iadd_12,)                                       {}
call_module    self_layer4_0_conv1         self_layer4_0_conv1                                         (self_layer3_5_relu_2,)                          {}
call_module    self_layer4_0_bn1           self_layer4_0_bn1                                           (self_layer4_0_conv1,)                           {}
call_module    self_layer4_0_relu          self_layer4_0_relu                                          (self_layer4_0_bn1,)                             {}
call_module    self_layer4_0_conv2         self_layer4_0_conv2                                         (self_layer4_0_relu,)                            {}
call_module    self_layer4_0_bn2           self_layer4_0_bn2                                           (self_layer4_0_conv2,)                           {}
call_module    self_layer4_0_relu_1        self_layer4_0_relu                                          (self_layer4_0_bn2,)                             {}
call_module    self_layer4_0_conv3         self_layer4_0_conv3                                         (self_layer4_0_relu_1,)                          {}
call_module    self_layer4_0_bn3           self_layer4_0_bn3                                           (self_layer4_0_conv3,)                           {}
call_module    self_layer4_0_downsample_0  self_layer4_0_downsample_0                                  (self_layer3_5_relu_2,)                          {}
call_module    self_layer4_0_downsample_1  self_layer4_0_downsample_1                                  (self_layer4_0_downsample_0,)                    {}
call_function  iadd_13                     <built-in function iadd>                                    (self_layer4_0_bn3, self_layer4_0_downsample_1)  {}
call_module    self_layer4_0_relu_2        self_layer4_0_relu                                          (iadd_13,)                                       {}
call_module    self_layer4_1_conv1         self_layer4_1_conv1                                         (self_layer4_0_relu_2,)                          {}
call_module    self_layer4_1_bn1           self_layer4_1_bn1                                           (self_layer4_1_conv1,)                           {}
call_module    self_layer4_1_relu          self_layer4_1_relu                                          (self_layer4_1_bn1,)                             {}
call_module    self_layer4_1_conv2         self_layer4_1_conv2                                         (self_layer4_1_relu,)                            {}
call_module    self_layer4_1_bn2           self_layer4_1_bn2                                           (self_layer4_1_conv2,)                           {}
call_module    self_layer4_1_relu_1        self_layer4_1_relu                                          (self_layer4_1_bn2,)                             {}
call_module    self_layer4_1_conv3         self_layer4_1_conv3                                         (self_layer4_1_relu_1,)                          {}
call_module    self_layer4_1_bn3           self_layer4_1_bn3                                           (self_layer4_1_conv3,)                           {}
call_function  iadd_14                     <built-in function iadd>                                    (self_layer4_1_bn3, self_layer4_0_relu_2)        {}
call_module    self_layer4_1_relu_2        self_layer4_1_relu                                          (iadd_14,)                                       {}
call_module    self_layer4_2_conv1         self_layer4_2_conv1                                         (self_layer4_1_relu_2,)                          {}
call_module    self_layer4_2_bn1           self_layer4_2_bn1                                           (self_layer4_2_conv1,)                           {}
call_module    self_layer4_2_relu          self_layer4_2_relu                                          (self_layer4_2_bn1,)                             {}
call_module    self_layer4_2_conv2         self_layer4_2_conv2                                         (self_layer4_2_relu,)                            {}
call_module    self_layer4_2_bn2           self_layer4_2_bn2                                           (self_layer4_2_conv2,)                           {}
call_module    self_layer4_2_relu_1        self_layer4_2_relu                                          (self_layer4_2_bn2,)                             {}
call_module    self_layer4_2_conv3         self_layer4_2_conv3                                         (self_layer4_2_relu_1,)                          {}
call_module    self_layer4_2_bn3           self_layer4_2_bn3                                           (self_layer4_2_conv3,)                           {}
call_function  iadd_15                     <built-in function iadd>                                    (self_layer4_2_bn3, self_layer4_1_relu_2)        {}
call_module    self_layer4_2_relu_2        self_layer4_2_relu                                          (iadd_15,)                                       {}
call_module    self_avgpool                self_avgpool                                                (self_layer4_2_relu_2,)                          {}
call_function  flatten                     <built-in method flatten of type object at 0x7fe53ec2f540>  (self_avgpool, 1)                                {}
call_module    self_fc                     self_fc                                                     (flatten,)                                       {}
output         output                      output                                                      ((self_fc,),)                                    {}

...

Output dtype is torch.bfloat16
```

