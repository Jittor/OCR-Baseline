from PIL import Image
import sys
import os

curr_path = os.path.dirname(__file__)
sys.path.extend([
    os.path.join(curr_path, '../'),
    os.path.join(curr_path, './'),
    os.path.join(curr_path, './src'),
    ])
from CRNN.src.model import CRNN
from CRNN.src.ctc_decoder import ctc_decode
from CRNN.src.utils import process_img
from CRNN.src.datasets import LABEL2CHAR
from CRNN.src import config


class Step3:
    def __init__(self, ckp_path) -> None:
        num_class = len(LABEL2CHAR) + 1
        self.model = CRNN(config.img_channel, config.img_height, config.img_width, num_class,
                          rnn_hidden=config.rnn_hidden)
        self.model.load(ckp_path)
        self.model.eval()

    def infer(self, image):
        image = image.convert('L')
        image = process_img(image, img_width=config.img_width, img_height=config.img_height,
                            img_channel=config.img_channel)
        image = image.unsqueeze(0)
        log_probs = self.model(image)
        preds = ctc_decode(log_probs.numpy(), method='greedy', label2char=LABEL2CHAR)
        prediction = "".join(preds[0])
        return prediction


if __name__ == "__main__":
    step3 = Step3('/project/train/models/CRNN.pkl')
    img = Image.open('/home/data/1306/016201_1508424072625001_1.jpg')
    results = step3.infer(img)
    print(results)
