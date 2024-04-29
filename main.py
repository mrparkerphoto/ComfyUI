import uuid

import comfy.options

comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time
import json
import pika
from google.cloud import storage
from google.oauth2 import service_account


def execute_prestartup_script():
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            print(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        print("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()

execute_prestartup_script()

# Main code
import asyncio
import itertools
import shutil
import gc

from comfy.cli_args import args
import logging

if os.name == "nt":
    logging.getLogger("xformers").addFilter(
        lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc

import comfy.utils
import yaml

import execution
import server
from server import BinaryEventTypes
from nodes import init_custom_nodes
import comfy.model_management


def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning(
                "\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


def prompt_worker(q, server):
    e = execution.PromptExecutor(server)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    with open(args.gcp_service_account_key_file) as key_file:
        api_key_string = json.loads(key_file.read())
    storage_credentials = service_account.Credentials.from_service_account_info(api_key_string)

    storage_client = storage.Client(
        args.gcp_project_id, credentials=storage_credentials
    )

    amqp_connection = pika.BlockingConnection(pika.ConnectionParameters(args.rabbitmq_host, heartbeat=6000))
    amqp_receiving_channel = amqp_connection.channel()
    amqp_receiving_channel.queue_declare(queue='comfy_requests', durable=True)
    amqp_receiving_channel.basic_qos(prefetch_count=1)

    amqp_response_channel = amqp_connection.channel()
    amqp_response_channel.queue_declare(queue='comfy_responses', durable=True)

    def download_input_image(image_link, img_name):
        bucket = storage_client.get_bucket(args.gcp_bucket)
        blob = bucket.blob(image_link)
        output_dir = folder_paths.get_directory_by_type("input")
        img_file = os.path.join(output_dir, img_name)
        blob.download_to_filename(img_file)

    def download_lora(lora_link):
        bucket = storage_client.get_bucket(args.gcp_bucket)
        blob = bucket.blob(lora_link)
        filename = lora_link.split("/")[-1]
        output_dir = folder_paths.get_directory_by_type("loras")
        img_file = os.path.join(output_dir, filename)
        blob.download_to_filename(img_file)

    def upload_image(base_link, img_id, img_path):
        bucket = storage_client.get_bucket(args.gcp_bucket)
        upload_link = base_link + img_id + ".png"
        blob = bucket.blob(upload_link)
        blob.upload_from_filename(img_path)
        return upload_link

    def consume_amqp_prompt(ch, method, properties, body):
        json_body = json.loads(body)
        data_setup = json_body['extra_data']['data_setup']

        if data_setup['request_type'] == "lookbooks":
            image_link = data_setup['image_link']
            image_name = data_setup['image_name']
            download_input_image(image_link, image_name)
            lora_link = data_setup['lora_link']
            download_lora(lora_link)

        item = server.run_prompt(json_body)
        prompt_id = item[1]
        execution_start_time = time.perf_counter()

        e.execute(item[2], prompt_id, item[3], item[4])
        nonlocal need_gc
        need_gc = True

        results = e.outputs_ui
        current_time = time.perf_counter()

        images = []
        for node in results:
            if "images" in results[node]:
                for image in results[node]["images"]:
                    if image["type"] == "output":
                        images.append(image)

        img_links = []
        for image in images:
            filename = os.path.basename(image['filename'])
            output_dir = folder_paths.get_directory_by_type("output")
            img_file = os.path.join(output_dir, filename)
            img_id = str(uuid.uuid4())
            img_link = upload_image(data_setup["base_link"], img_id, img_file)
            img_links.append(img_link)

        execution_time = current_time - execution_start_time
        logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        nonlocal last_gc_collect

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                comfy.model_management.cleanup_models()
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False

        data = {"response_type": data_setup["request_type"],
                "image_links": img_links}
        payload = json.dumps(data)
        amqp_response_channel.basic_publish(exchange='',
                                            routing_key='comfy_responses',
                                            body=payload)

        ch.basic_ack(delivery_tag=method.delivery_tag)

    amqp_receiving_channel.basic_consume(queue='comfy_requests', auto_ack=False,
                                         on_message_callback=consume_amqp_prompt)
    amqp_receiving_channel.start_consuming()


def hijack_progress(server):
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()
        progress = {"value": value, "max": total, "prompt_id": server.last_prompt_id, "node": server.last_node_id}

        server.send_sync("progress", progress, server.client_id)
        if preview_image is not None:
            server.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server.client_id)

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                logging.info("Adding extra search path {} {}".format(x, full_path))
                folder_paths.add_model_folder_path(x, full_path)


if __name__ == "__main__":
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater

            new_updater.update_windows_updater()
        except:
            pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = server.PromptServer(loop)
    q = execution.PromptQueue(server)

    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    init_custom_nodes()

    cuda_malloc_warning()

    prompt_worker(q, server)

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.quick_test_for_ci:
        exit(0)

    call_on_start = None
    cleanup_temp()
