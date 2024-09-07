import os
import cv2
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
from mltu.configs import BaseModelConfigs


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


def load_config(config_path):
    print(f"Attempting to load config from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    return BaseModelConfigs.load(config_path)


def load_model(configs):
    model_path = configs.model_path
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return ImageToWordModel(model_path=model_path, char_list=configs.vocab)


def load_validation_data(val_csv_path):
    print(f"Loading validation data from: {val_csv_path}")
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"Validation CSV file not found at {val_csv_path}")
    return pd.read_csv(val_csv_path).values.tolist()


def main():
    print("Current working directory:", os.getcwd())
    print("Script directory:", os.path.dirname(os.path.abspath(__file__)))

    # Adjust these paths as necessary
    config_path = os.path.join(os.path.dirname(__file__), "Models", "03_handwriting_recognition", "202409080211",
                               "configs.yaml")
    val_csv_path = os.path.join(os.path.dirname(__file__), "Models", "03_handwriting_recognition", "202409080211",
                                "val.csv")

    try:
        configs = load_config(config_path)
        model = load_model(configs)
        df = load_validation_data(val_csv_path)

        accum_cer = []
        for image_path, label in tqdm(df):
            image_path = image_path.replace("\\", "/")
            print(f"Processing image: {image_path}")

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            prediction_text = model.predict(image)
            cer = get_cer(prediction_text, label)
            print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
            accum_cer.append(cer)

            # Resize and display image
            image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"Average CER: {np.average(accum_cer)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
