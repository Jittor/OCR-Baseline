from PIL import Image
import sys

sys.path.extend(['src'])
from src.model import CRNN
from src.ctc_decoder import ctc_decode
from src.utils import process_img
from src.datasets import LABEL2CHAR
import src.config


class Step3:
    def __init__(self, ckp_path) -> None:
        num_class = len(LABEL2CHAR) + 1
        self.model = CRNN(src.config.img_channel, src.config.img_height, src.config.img_width, num_class,
                          rnn_hidden=src.config.rnn_hidden)
        self.model.load(ckp_path)
        self.model.eval()

    def infer(self, image):
        image = image.convert('L')
        image = process_img(image, img_width=src.config.img_width, img_height=src.config.img_height,
                            img_channel=src.config.img_channel)
        image = image.unsqueeze(0)
        log_probs = self.model(image)
        preds = ctc_decode(log_probs.numpy(), method='greedy', label2char=LABEL2CHAR)
        prediction = "".join(preds[0])
        return prediction


if __name__ == "__main__":
    step3 = Step3('/mnt/disk/llt/model/CRNN.pkl')
    img = Image.open(
        '/mnt/disk/llt/data/task3_demo/010003_1508413314371766_1.jpg')
    results = step3.infer(img)
    print(results)
