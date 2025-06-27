from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import HubertModel


class TrustworthinessClassifier(pl.LightningModule):
    def __init__(self, hubert_model_name: str):
        super().__init__()
        self.save_hyperparameters()

        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        self.linear = nn.Sequential(nn.Linear(self.hubert.config.hidden_size, 1))

    def forward(self, wav: Tensor, mask: Tensor) -> Tensor:
        """
        The model's forward pass

        Parameters
        ----------
        input_values : Tensor
            A Tensor of shape (batch_size, sequence_length). Contains floating-point mono 16 kHz audio waveform data
        attention_mask : Tensor
            A Tensor of shape (batch_size, sequence_length). Contains Boolean values indiating whether the corresponding element in input_values is not masked (True) or masked (False)

        Returns
        -------
        Tensor
            A Tensor of shape (batch_size, 1) for binary classification.
        """
        outputs = self.hubert(input_values=wav, attention_mask=mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.linear(pooled_output)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        wav, mask, trustworthy = batch
        batch_size = wav.shape[0]

        y_pred = self(wav=wav, mask=mask)
        loss = F.binary_cross_entropy_with_logits(input=y_pred, target=trustworthy)

        self.log(
            "training_loss",
            loss.detach(),
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        wav, mask, trustworthy = batch
        batch_size = wav.shape[0]

        y_pred = self(wav=wav, mask=mask)
        loss = F.binary_cross_entropy_with_logits(input=y_pred, target=trustworthy)

        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        wav, mask, trustworthy = batch
        batch_size = wav.shape[0]

        y_pred = self(wav=wav, mask=mask)
        loss = F.binary_cross_entropy_with_logits(input=y_pred, target=trustworthy)

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss
