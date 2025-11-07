import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

class GradientAscent:
    """
    Clase para aplicar Gradient Ascent sobre un modelo de visión.
    Usa estimación de gradientes mediante perturbaciones aleatorias.
    """

    def __init__(self, model: nn.Module, lr: float = 0.1, iterations: int = 200, device=None, image=None):
        self.model = model.eval()
        self.lr = lr
        self.iterations = iterations
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.image = image

    def explain(self, target_class: int, input_shape=(3, 28, 28), init_random: bool = True, 
                init_image: bool = False, sigma: float = 0.1, mask=None):
        """
        Aplica gradient ascent usando estimación de gradientes con perturbaciones.
        
        Args:
            target_class: clase objetivo a maximizar
            input_shape: forma del input (C, H, W)
            init_random: si True, empieza desde ruido; si False, desde ruido pequeño
            init_image: si True, usa self.image como punto de partida
            sigma: desviación estándar para las perturbaciones
            mask: tensor booleano o binario (H, W) indicando dónde aplicar perturbaciones
                  True/1 = píxel editable, False/0 = píxel fijo
        
        Returns:
            Tensor de la imagen optimizada
        """
        
        # Inicializa la imagen
        if init_image and self.image is not None:
            image = deepcopy(self.image).to(self.device)
        elif init_random:
            image = torch.rand(input_shape, device=self.device) * 0.5 + 0.25
        else:
            image = torch.randn(input_shape, device=self.device) * 0.1 + 0.5
        
        image = torch.clamp(image, 0, 1)
        
        # Guarda la imagen inicial para preservar píxeles fijos
        initial_image = image.clone()
        
        # Procesa la máscara
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.to(self.device)
            # Convierte a booleano si es necesario
            if mask.dtype != torch.bool:
                mask = mask.bool()
            # Expande la máscara para todos los canales: (H, W) -> (C, H, W)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(input_shape[0], -1, -1)
        else:
            # Si no hay máscara, todos los píxeles son editables
            mask = torch.ones(input_shape, dtype=torch.bool, device=self.device)
        
        best_prob = 0
        best_img = None
        new_img = None

        with torch.no_grad():
            for epoch in tqdm(range(self.iterations)):
                # Genera una perturbación aleatoria
                eps = torch.randn_like(image).to(self.device)
                
                # Aplica la máscara a la perturbación (solo perturba píxeles editables)
                eps = eps * mask
                
                # Crea imagen perturbada
                noisy_image = (image + sigma * eps).clamp(0, 1)
                
                # Evalúa el modelo en la imagen perturbada
                f = self.model(noisy_image.unsqueeze(0))[:, target_class]
                
                # Estima el gradiente
                grad_estimate = eps * f / sigma
                
                # Actualiza SOLO los píxeles dentro de la máscara
                image = image + self.lr * grad_estimate * mask
                image = image.clamp(0, 1)
                
                # Restaura los píxeles fuera de la máscara a sus valores originales
                image = torch.where(mask, image, initial_image)
                
                # Evalúa el resultado
                output = self.model(image.unsqueeze(0))
                probs = F.softmax(output, dim=1)
                loss = -output[0, target_class]
                pred = output.argmax(dim=1).item()
                current_prob = probs[0, target_class].item()
                
                # Guarda la mejor imagen
                if current_prob > best_prob:
                    best_prob = current_prob
                    best_img = image.clone()
                
                # Logging
                if epoch % 10 == 0 or epoch == self.iterations - 1:
                    sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
                    print(f"Epoch {epoch + 1}: Pred: {pred}, Target: {target_class}, "
                          f"Loss: {loss.item():.4f}, Prob: {current_prob:.4f}, "
                          f"Top3: {sorted_indices[:3].tolist()}")
                
                # Detección de convergencia
                if pred == target_class and current_prob > 0.9:
                    new_img = image.clone()
                    print(f"Converged at epoch {epoch + 1}")
                    break

        # Retorna la mejor imagen encontrada
        result_img = new_img if new_img is not None else (best_img if best_img is not None else image)
        return result_img.detach().cpu()