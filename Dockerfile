FROM pytorch/pytorch
WORKDIR /app

RUN sudo apt install gcc

COPY ./requirements.txt /app
RUN pip install -r requirements.txt

COPY ./custom_nodes/comfyui-reactor-node/requirements.txt /app
RUN pip install -r requirements.txt

COPY ./custom_nodes/comfyui_controlnet_aux/requirements.txt /app
RUN pip install -r requirements.txt

COPY ./custom_nodes/ComfyUI-Impact-Pack/requirements.txt /app
RUN pip install -r requirements.txt

#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get update && apt-get install libgl1 -y
COPY . .
ENTRYPOINT ["python", "main.py"]
