import os

import gradio as gr
import mlflow
import numpy as np

aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

# this only serves for executing the code outside the docker container
if aws_access_key_id is None or aws_secret_access_key is None:
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
    fn=predict,
    inputs=gr.Textbox(
        lines=2, placeholder="Type the academic text for classification here"
    ),
    outputs="text",
)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
