import abc

import torch
import torch.nn as nn


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_layers: int = 2, n_head: int = 4):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens

        self.embedding = nn.Embedding(n_tokens, d_latent)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_latent, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        # self.transformer = nn.TransformerEncoderLayer(d_model=d_latent, nhead=n_head, batch_first=True)

        self.output_layer = nn.Linear(d_latent, n_tokens)

        self.pos_embedding = nn.Parameter(torch.randn(1, 600, d_latent))
        self.start_token = nn.Parameter(torch.randn(1, 1, d_latent))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, H, W = x.shape
        seq_len = H * W
        flattened = x.view(B, seq_len)
        embedded = self.embedding(flattened)

        pos_embedded = embedded + self.pos_embedding[:, :seq_len, :]

        # start_token = torch.zeros(B, 1, self.d_latent, device=x.device)
        # x_shifted = torch.cat([start_token, embedded[:, :-1, :]], dim=1)
        start_token = self.start_token.expand(B, -1, -1)
        x_shifted = torch.cat([start_token, pos_embedded[:, :-1, :]], dim=1)

        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        x_trans = self.transformer(x_shifted, mask=mask)
        # x_trans = self.transformer(x_shifted, src_mask=mask)

        output = self.output_layer(x_trans)

        return output.view(B, H, W, self.n_tokens), {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        x = torch.zeros((B, h * w), dtype=torch.long, device=device)

        for i in range(h * w):
            logits, _ = self.forward(x.view(B, h, w))
            logits_flattened = logits.view(B, h * w, self.n_tokens)
            probabilities = torch.softmax(logits_flattened[:, i, :], dim=-1)
            x[:, i] = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    
        return x.view(B, h, w)
