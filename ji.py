from model import Pipeline
import cv2
import json


def init():  # 模型初始化
    model = Pipeline()
    return model


def process_image(net, input_image, args=None):
    dets = net(input_image)
    '''
        此示例 dets 数据为
        dets = [[294, 98, 335, 91, 339, 115, 298, 122, '店面招牌'], [237, 111, 277, 106, 280, 127, 240, 133, '文字识别']]
    '''
    result = {"model_data": {"objects": []}}
    for det in dets:
        points = det[:8]
        name = det[-1]
        confidence = None
        single_obj = {
            'points': points,
            'name': name,
            'confidence': confidence
        }
        result['model_data']['objects'].append(single_obj)
    return json.dumps(result)


'''
ev_sdk输出json样例
{
"model_data": {
    "objects": [
        {
            "points": [
                294,
                98,
                335,
                91,
                339,
                115,
                298,
                122
            ],
            "name": "店面招牌",
            "confidence": 0.9977550506591797
        },
        {
            "points": [
                237,
                111,
                277,
                106,
                280,
                127,
                240,
                133
            ],
            "name": "文字识别",
            "confidence": 0.9727065563201904
        }
    ]
    }
}
'''

if __name__ == "__main__":
    net = init()
    input_image = cv2.imread('/home/data/1413/011284_1508290961970980.jpg')
    output = process_image(net, input_image)
    print(json.loads(output))


# python /project/ev_sdk/src/ji.py
