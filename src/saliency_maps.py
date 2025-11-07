# ai libraries
import torch
import torch.nn.functional as F

# other libraries
import copy

# from typing import Callable


class SaliencyMap:
    """
    This is the class for computing saliency maps.

    Attr:
        model: model used to classify.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        This function is the constructor

        Args:
            model: model used to classify.

        Returns:
            None.
        """

        self.model = model

    def explain(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the explanation.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels,
                height, width].

        Raises:
            RuntimeError: No gradients were computed during the
                backward pass.

        Returns:
            saliency maps tensor. Dimensions: [batch, height, width].
        """

        # TODO
        # El saliency map es coger el máximo de los gradientes
        # a través de los canales. Los gradientes son el de los
        # outputs respecto a los inputs

        # Cogemos la clase que va a predecir
        # Lo hago con un requieres_grad = True
        # Para coger los gradientes después
        inputs.requires_grad = True
        predictions: torch.Tensor = self.model(inputs)
        predicted_class: torch.Tensor = torch.argmax(predictions, dim=1)

        # One-hot encoding
        one_hot_output: torch.Tensor = torch.zeros_like(predictions)
        one_hot_output[range(inputs.size(0)), predicted_class] = 1

        # Calculamos los gradientes
        gradients: torch.Tensor = torch.autograd.grad(
            outputs=predictions,
            inputs=inputs,
            grad_outputs=one_hot_output,
            create_graph=True,
        )[0]

        # Lanzar un error si no se calculan gradientes
        if gradients is None:
            raise RuntimeError("No gradients were computed during the backward pass.")

        # Cogemos el máximo a través de los canales
        saliency_map: torch.Tensor = torch.max(torch.abs(gradients), dim=1)[0]

        return saliency_map


class SmoothGradSaliencyMap(torch.nn.Module):
    """
    This is the class for computing smoothgrad saliency maps.

    Attr:
        model: model used to classify.
    """

    def __init__(self) -> None:
        """
        Thi function is the constructor for SmoothGradSaliencyMap.

        Args:
            model: model used to classify.

        Returns:
            None.
        """

        # call super class constructor
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        model: torch.nn.Module,
        noise_level: float,
        sample_size: int,
    ) -> torch.Tensor:
        return self.explain(inputs, model, noise_level, sample_size)


    def explain(
        self,
        inputs: torch.Tensor,
        model: torch.nn.Module,
        noise_level: float,
        sample_size: int,
    ) -> torch.Tensor:
        """
        This method computes the explanation.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels,
                height, width].

        Raises:
            RuntimeError: No gradients were computed during the
                backward pass.

        Returns:
            saliency maps tensor. Dimensions: [batch, height, width].
        """

        # TODO
        B: int
        H: int
        W: int
        B, _, H, W = inputs.size()
        saliency_map: torch.Tensor = torch.zeros(B, H, W).to(inputs.device)

        for _ in range(sample_size):
            noisy_input: torch.Tensor = torch.randn_like(inputs) * noise_level + inputs

            with torch.enable_grad():
                noisy_input = noisy_input.requires_grad_(True)
                predictions: torch.Tensor = model(noisy_input)
                predicted_class: torch.Tensor = torch.argmax(predictions, dim=1)

                # One-hot encoding
                one_hot_output: torch.Tensor = torch.zeros_like(predictions)
                one_hot_output[range(inputs.size(0)), predicted_class] = 1

                # Calculamos los gradientes
                gradients: torch.Tensor = torch.autograd.grad(
                    outputs=predictions,
                    inputs=noisy_input,
                    grad_outputs=one_hot_output,
                    create_graph=True,
                )[0]

            # Lanzar un error si no se calculan gradientes
            if gradients is None:
                raise RuntimeError(
                    "No gradients were computed during the backward pass."
                )

            # Cogemos el máximo a través de los canales
            saliency_map += torch.max(gradients, dim=1)[0]

        # Promediar el resultado
        saliency_map /= sample_size
        return saliency_map


class DeConvNet:
    @torch.no_grad()
    def __init__(self, model: torch.nn.Module) -> None:
        """
        This is the constructor for DeConvNet.

        Args:
            model: model used to classify.
        """

        # set attributes
        self.model = copy.deepcopy(model)
        self.register_hooks()

    def register_hooks(self) -> None:
        """
        This function registers the hooks needed for deconvnet.

        Returns:
            None.
        """

        # TODO
        def relu_hook(module, grad_input, grad_output):

            return (F.relu(grad_output[0]),)

        for layer in self.model.modules():

            if isinstance(layer, torch.nn.ReLU):
                layer.register_backward_hook(relu_hook)

    def explain(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the explanation.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels,
                height, width].

        Raises:
            RuntimeError: No gradients were computed during the
                backward pass.

        Returns:
            saliency maps tensor. Dimensions: [batch, height, width].
        """

        # TODO
        # El saliency map es coger el máximo de los gradientes
        # a través de los canales. Los gradientes son el de los
        # outputs respecto a los inputs

        # Cogemos la clase que va a predecir
        # Lo hago con un requieres_grad = True
        # Para coger los gradientes después
        inputs.requires_grad = True
        predictions: torch.Tensor = self.model(inputs)
        predicted_class: torch.Tensor = torch.argmax(predictions, dim=1)

        # One-hot encoding
        one_hot_output: torch.Tensor = torch.zeros_like(predictions)
        one_hot_output[range(inputs.size(0)), predicted_class] = 1

        # Calculamos los gradientes
        gradients: torch.Tensor = torch.autograd.grad(
            outputs=predictions,
            inputs=inputs,
            grad_outputs=one_hot_output,
            create_graph=True,
        )[0]

        # Lanzar un error si no se calculan gradientes
        if gradients is None:
            raise RuntimeError("No gradients were computed during the backward pass.")

        # Cogemos el máximo a través de los canales
        saliency_map: torch.Tensor = torch.max(gradients, dim=1)[0]

        return saliency_map
