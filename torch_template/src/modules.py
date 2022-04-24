from torch import nn, Tensor

from torch import DeviceObjType as TorchDevice

class TemplateNeuralNetwork(nn.Module):
    def __init__(self,
        in_features: int=0,
        out_features: int=0,
        device: TorchDevice = "cpu"
        ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features)
        )

    def init_weights(self) -> None:
        return None

    def forward(self, input_data) -> Tensor:
        prediction = Tensor
        return prediction

    def from_config(config: dict, device):

        in_features = config.get("in_features", 0)
        out_features = config.get("out_features", 0)
        
        return TemplateNeuralNetwork(in_features, out_features, device)