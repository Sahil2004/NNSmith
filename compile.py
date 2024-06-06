import os
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort


"""
Custom model class definition
If you want to test on different model declare model class of that model here

"""

class Model(nn.Module):
    def __init__(self, input_size=4, output_size=3):
        super(Model, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def load_model(model_path):
    """
    Load a PyTorch model from a file.
    """
    model = torch.load(model_path)
    model.eval() 
    return model

def parse_model(model):
    """
    Parse the PyTorch model architecture.
    """
    layers_info = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.Softmax)):
            layers_info.append({'name': name, 'type': type(layer).__name__})
    print("Model architecture:")
    for layer_info in layers_info:
        print(f"Layer: {layer_info['name']}, Type: {layer_info['type']}")

def compile_and_optimize_model(model, output_path):
    """
    Compile and optimize the PyTorch model using ONNX Runtime compiler and save it.
    """
    example_input = torch.randn(1, 4)  # Adjust input size to match the model input
    onnx_path = output_path.replace(".pth", ".onnx")
    torch.onnx.export(model, example_input, onnx_path, export_params=True, opset_version=9,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    onnx.save_model(onnx_model, output_path)

    print(f"ONNX model saved to {onnx_path}")

    sess_options = ort.SessionOptions()
    if ort.get_device() == "CUDA":
        device = 'gpu'
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.log_severity_level = 2  # Verbose logging for debugging
    else:
        device = 'cpu'
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    compiled_model = ort.InferenceSession(onnx_path, sess_options)

    print(f"Compiled and optimized ONNX model saved to {output_path} (Device: {device})")

def main():
    model_path = 'model/model.pth'
    output_path = 'compile/optimized_model.onnx'

    os.makedirs('compile', exist_ok=True)

    model = load_model(model_path)
    parse_model(model)

    compile_and_optimize_model(model, output_path)

    print("Compilation and optimization of the model are complete.")

if __name__ == "__main__":
    main()
