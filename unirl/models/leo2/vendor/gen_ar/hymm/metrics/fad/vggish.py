import torch
import torch.nn as nn
import math

from . import vggish_params

_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _frame(data, window_length, hop_length):
    num_samples = data.shape[0]
    num_frames = 1 + int(math.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.stride()[0] * hop_length,) + data.stride()
    return torch.as_strided(data, shape, strides)


def _hertz_to_mel(frequencies_hertz):
    return _MEL_HIGH_FREQUENCY_Q * torch.log(1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def _spectrogram_to_mel_matrix(
    num_mel_bins=20,
    num_spectrogram_bins=129,
    audio_sample_rate=8000,
    lower_edge_hertz=125.0,
    upper_edge_hertz=3800.0,
):
    nyquist_hertz = audio_sample_rate / 2.0
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" % (lower_edge_hertz, upper_edge_hertz))

    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" % (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = torch.linspace(0.0, nyquist_hertz, num_spectrogram_bins)

    spectrogram_bins_mel = _hertz_to_mel(spectrogram_bins_hertz)
    # The i'th mel band (starting from i=1) has center frequency
    # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
    # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
    # the band_edges_mel arrays.
    band_edges_mel = torch.linspace(
        _hertz_to_mel(torch.tensor(lower_edge_hertz)),
        _hertz_to_mel(torch.tensor(upper_edge_hertz)),
        num_mel_bins + 2,
    )
    # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
    # of spectrogram values.
    mel_weights_matrix = torch.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i : i + 3]
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the *mel* domain, not hertz.
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)

        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = torch.maximum(torch.tensor(0.0), torch.minimum(lower_slope, upper_slope))

    # HTK excludes the spectrogram DC bin; make sure it always gets a zero
    # coefficient.
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def _stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    frames = _frame(signal, window_length, hop_length)
    window = torch.hann_window(window_length, periodic=True).to(signal.device)
    windowed_frames = frames * window
    return torch.abs(torch.fft.rfft(windowed_frames, int(fft_length)))


def _log_mel_spectrogram(
    data,
    audio_sample_rate=8000,
    log_offset=0.0,
    window_length_secs=0.025,
    hop_length_secs=0.010,
    **kwargs,
):
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(math.ceil(math.log(window_length_samples) / math.log(2.0)))

    spectrogram = _stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples,
    )
    mel_spectrogram = torch.matmul(
        spectrogram,
        _spectrogram_to_mel_matrix(
            num_spectrogram_bins=spectrogram.shape[1],
            audio_sample_rate=audio_sample_rate,
            **kwargs,
        ).to(spectrogram),
    )
    return torch.log(mel_spectrogram + log_offset)


def _waveform_to_examples(data):
    # Compute log mel spectrogram features, with shape (n_frame, n_mel)
    log_mel = _log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_BANDS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ,
    )

    # Frame features into examples, with shape (n_example, n_frame, n_mel)
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))

    example_hop_length = int(round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = _frame(log_mel, window_length=example_window_length, hop_length=example_hop_length)

    # (n_example, 1, n_frame, n_mel)
    return log_mel_examples.unsqueeze(1)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = _waveform_to_examples(x)
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


class VGGish(VGG):
    def __init__(self, vgg_ckpt, pca_ckpt, pretrained=True):
        super().__init__(make_layers())
        if pretrained:
            state_dict = torch.load(vgg_ckpt, "cpu")
            super().load_state_dict(state_dict)
        
        # According to fadtk, embedding does not use last relu
        self.embeddings = self.embeddings[:-1]
        # According to fadtk, pca set as false
        # self.pproc = Postprocessor()
        # if pretrained:
        #     state_dict = torch.load(pca_ckpt, "cpu")
        #     self.pproc.load_state_dict(state_dict)

    def forward(self, x, fs=None):
        x = VGG.forward(self, x)
        # x = self._postprocess(x)
        return x

    def _postprocess(self, x):
        return self.pproc(x)
