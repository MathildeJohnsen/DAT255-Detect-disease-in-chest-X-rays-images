import cv2
import torch
import numpy as np
import torch.nn.functional as F


def get_last_conv_layer(model):
    for layer in reversed(model.features):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise ValueError("Fant ikke Conv2d-lag i model.features")


class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        weights = gradients.mean(dim=(1, 2))  # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(self.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_heatmap(original_image, cam, alpha=0.4):
    """
    original_image: numpy array i RGB-format
    cam: 2D heatmap i området [0,1]
    """
    h, w, _ = original_image.shape

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap, alpha, 0)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)