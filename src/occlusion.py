import torch 
import torch.nn.functional as F

class Occlusion(torch.nn.Module):
    """
    This is the Occlusion class.

    Args:
        model: model to classify images.
        mask_size: size of the mask to occlude.
    """

    def __init__(self, model: torch.nn.Module, mask_size: int = 16) -> None:
        """
        Constructor of Occlusion class.

        Args:
            model: neural net used for classifying images.
        mask_size: mask used for hiding part of the image in the
            creation of the maps. Defaults to 50.
        """

        # call super class constructor
        super().__init__()

        # set attribute
        self.model = model
        self.mask_size = mask_size

    def occlude(self, inputs: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """
        This method mask a part of the inputs and returns it.

        Even if is returning the value, it should make this operation
        treating the inputs as if they were passed by reference (no
        copies should be made).

        Args:
            inputs: input tensor. Dimensions: [batch, channels, width,
                height].
            i: width position where the masking starts.
            j: height position where the masking starts.

        Returns:
            masked tensor. Dimensions: [batch, channels, width, height].
        """

        # TODO
        inputs[:, :, i:i+self.mask_size, j:j+self.mask_size] = 0
        return inputs

    def forward_occlude(
        self, inputs: torch.Tensor, class_index: torch.Tensor
    ) -> torch.Tensor:
        """
        This method computes the forward of each occluded batch. The
        result must show the probability for the class that was
        originally predicted. Since we want to show the zones where the
        probability drop is higher brighter, you must compute
        1 - probability.

        Args:
            inputs: input tensor. Dimensions: [batch, channels, width,
                height].
            class_index: indexes of the class that was originally
                predicted for each class. [batch, 1].

        Returns:
            outputs tensor. Dimensions: [batch, 1, 1].
        """

        # TODO

        outputs = self.model(inputs)

        probs = F.softmax(outputs)
        probs = torch.gather(probs, 1, class_index)


        return 1- probs.unsqueeze(1)

    def update_probs(
        self,
        inputs: torch.Tensor,
        i: int,
        j: int,
        probabilities: torch.Tensor,
        average_tensor: torch.Tensor,
        class_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method updates the probabilities and average tensor for
        one pass (one batch prediction). You will have to call occlude
        and forward methods here.

        This method should treat the inputs as passed by value and the
        probabilities and average passed by reference (no copies
        allowed).

        Args:
            inputs: input tensor. Dimensions: [batch, channels, width,
                height].
            i: width position where the masking starts.
            j: height position where the masking starts.
            probabilities: tensor to store the inverse probabilities
                (1 - p) of the inputs. Dimensions: [batch, width,
                height].
            average_tensor: tensor to store how many times
                probabilities were computed for each pixel. Dimensions:
                [batch, width, height].
            class_index: indexes of the class that was originally
                predicted for each class. [batch, 1].

        Returns:
            probabilities: indexes of the class that was originally
                predicted for each class. [batch, 1].
            average_tensor: tensor to store how many times
                probabilities were computed for each pixel. Dimensions:
                [batch, width, height].
            
        """

        # TODO
        B, C, W, H = inputs.size()
        inputs = inputs.clone()
        occluded = self.occlude(inputs, i, j)
        new_prob = self.forward_occlude(occluded, class_index).to(inputs.device)


        average_tensor[:, i:i+self.mask_size, j:j+self.mask_size] += 1
        probabilities[:, i:i+self.mask_size, j:j+self.mask_size] += new_prob

        return probabilities, average_tensor

    def explain(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This class computes the whole occlusion explainability method.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels, width,
                height].

        Returns:
            occlusion maps. Dimensions: [batch, width, height].
        """

        # TODO
        class_index = torch.argmax(self.model(inputs), dim=1).unsqueeze(1)
        B, C, H, W = inputs.size()
        device = inputs.device
        # Está dando error porque, por algún motivo, me está devolviendo al revés el new_probs
        occlusion_maps = torch.zeros((B, H, W), device=device)
        average_tensor = torch.zeros((B, H, W), device=device)
        probabilities = torch.zeros((B, H, W), device=device)

        for h in range(H):
            for w in range(W):
                probabilities, average_tensor = self.update_probs(inputs=inputs, i=h, j=w, probabilities=probabilities, average_tensor=average_tensor, class_index=class_index)
        
        occlusion_maps = probabilities/average_tensor
        occlusion_maps = (occlusion_maps-occlusion_maps.min())/(occlusion_maps.max()-occlusion_maps.min())
        return occlusion_maps