import torch
try:
  arg_0 = 16
  arg_1 = 16
  arg_2_0 = 1
  arg_2_1 = 3
  arg_2 = [arg_2_0,arg_2_1,]
  stride = 1
  padding_0 = 0
  padding_1 = 1
  padding = [padding_0,padding_1,]
  bias = True
  dilation_0 = 1
  dilation_1 = 1
  dilation = [dilation_0,dilation_1,]
  res = torch.nn.Conv2d(arg_0,arg_1,arg_2,stride=stride,padding=padding,bias=bias,dilation=dilation,)
  ins_0 = torch.rand(torch.Size([8, 3, 256, 512]), dtype=torch.float32)
  ins = [ins_0,]
  res(*ins)
except Exception:
  pass
