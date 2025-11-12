import torch

aten = torch.ops.aten

compute_intensive_ops = [  
   aten.mm.default,
   aten.convolution.default,
   aten.convolution_backward.default,
   aten.bmm.default,
   aten.addmm.default,
   aten._scaled_dot_product_flash_attention.default,
   aten._scaled_dot_product_efficient_attention.default,
   aten._flash_attention_forward.default,
   aten._efficient_attention_forward.default,
   aten.upsample_bilinear2d.default,
   aten._scaled_mm.default
]
