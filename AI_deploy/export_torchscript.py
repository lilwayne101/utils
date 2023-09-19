import torch
import torchvision.models as models


# An instance of your model.
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()
# An example input you would normally provide to your model's forward() method.
example = torch.randn(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, (example,))

traced_script_module.save("model.torchscript")
