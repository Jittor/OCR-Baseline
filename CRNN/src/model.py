from jittor import nn


class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class, rnn_hidden=256):
        super().__init__()

        self.cnn = self._cnn_backbone(img_channel, img_height, img_width)

        self.rnn_hidden = rnn_hidden

        self.blstm1 = nn.LSTM(512 * (img_height // 16 - 1), rnn_hidden, bidirectional=True)
        self.linear1_1 = nn.Linear(rnn_hidden, rnn_hidden)
        self.linear1_2 = nn.Linear(rnn_hidden, rnn_hidden)

        self.blstm2 = nn.LSTM(rnn_hidden, rnn_hidden, bidirectional=True)
        self.linear2_1 = nn.Linear(rnn_hidden, num_class)
        self.linear2_2 = nn.Linear(rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i + 1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv(input_channel, output_channel, kernel_sizes[i], stride=strides[i], padding=paddings[i]))

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm(output_channel))

            relu = nn.ReLU()
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module('pooling2', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module('pooling3', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        # (512, img_height // 16, img_width // 4)

        conv_relu(6)
        # (512, img_height // 16 - 1, img_width // 4 - 1)

        # output_channel = channels[-1]
        # output_height = img_height // 16 - 1
        # output_width = img_width // 4 - 1

        return cnn

    def execute(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.shape
        conv = conv.view(batch, channel * height, width)
        seq = conv.permute(2, 0, 1)  # (width, batch, feature)

        rec1, _ = self.blstm1(seq)
        out1 = self.linear1_1(rec1[:, :, :self.rnn_hidden]) + \
            self.linear1_2(rec1[:, :, self.rnn_hidden:])

        rec2, _ = self.blstm2(out1)
        logits = self.linear2_1(
            rec2[:, :, :self.rnn_hidden]) + self.linear2_2(rec2[:, :, self.rnn_hidden:])
        # shape: (seq_len, batch, num_class)

        log_probs = nn.log_softmax(logits, dim=2)
        return log_probs
