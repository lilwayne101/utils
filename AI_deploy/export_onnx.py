import torch
import onnx
import torch.nn as nn
from torchvision import models
import onnxruntime
import numpy as np
import onnxsim

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.out_layer = nn.Linear(1000, 2)

    def forward(self, x, x1):
        out1 = self.model(x)
        out2 = self.model(x1)
        return out1, self.out_layer(out2)


def export_onnx(onnx_model_name, model, input_dict: dict, output_name: list, dynamic=True, simplify=True):
    """
    :param onnx_model_name:
    :param model:
    :param input_dict: {"input1":(1, 3, 224, 224), "input2":(1, 3, 224, 224),...}
    :param output_name: {"output1","output2",...}
    :param dynamic:
    :param simplify:
    :return:
    """
    inp = (torch.randn(i) for i in input_dict.values())
    model.eval()
    # generate ONNX model
    input_name = [i for i in input_dict]
    all_name = input_name + output_name
    dynamic_axes = {}
    for i in all_name:
        dynamic_axes[i] = {0: 'batch_size'}

    # Export the model
    torch.onnx.export(model=model,    # model being run
                      args=tuple(inp),   # model input (or a tuple for multiple inputs)
                      f=onnx_model_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,   # store the trained parameter wights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding
                      input_names=[i for i in input_name],   # the model's input names
                      output_names=[i for i in output_name],  # the model's output names
                      dynamic_axes=dynamic_axes if dynamic else None
                      )

    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)

    # 验证精度
    ort_session = onnxruntime.InferenceSession(onnx_model_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {i.name: to_numpy(torch.randn(input_dict[i.name.split(".")[0]])) for i in ort_session.get_inputs()} if '.' in ort_session.get_inputs()[0].name else {i.name: to_numpy(torch.randn(input_dict[i.name])) for i in ort_session.get_inputs()}
    ort_outs = ort_session.run(None, ort_inputs)
    net_input = [torch.tensor(i) for i in ort_inputs.values()]
    torch_out = model(*net_input)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and result looks good!")

    # Simplify
    if simplify:
        try:
            model_onnx, check = onnxsim.simplify(onnx_model_name)
            assert check, "assert check failed"
            onnx.save(model_onnx, onnx_model_name)
        except Exception as e:
            print(f"simplifier failure: {e}")


if __name__ == '__main__':
    net = Model()
    input_dict = {
        "input1": (1, 3, 224, 224),
        "input2": (1, 3, 224, 224)
    }
    export_onnx(onnx_model_name="model.onnx", model=net, input_dict=input_dict, output_name=['input1', 'input2'],
                dynamic=True, simplify=True)
