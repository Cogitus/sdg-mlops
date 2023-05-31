import os

import gradio as gr
import mlflow
import numpy as np
from dotenv import load_dotenv

# getting environment variables from .env (the AWS secrets)
load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def predict(model_input):
    model = mlflow.tensorflow.load_model(model_uri="models:/testing/1")

    model_input = [model_input]
    output = model.predict(np.array(model_input))

    labels = (output > 0.5) + 0
    labels = labels.flatten().tolist()
    labels = [str(label) for label in labels]
    labels = [(str(label), f"SDG{i}") for label, i in zip(labels, range(16))]

    # Eexample: ('1', 'SDG3')
    return " ".join(label[1] for label in labels if label[0] == "1")


demo = gr.Interface(
    fn=predict, inputs=gr.Textbox(lines=2, placeholder="Name Here..."), outputs="text"
)

demo.launch()
