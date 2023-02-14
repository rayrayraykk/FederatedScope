import sys
import gradio as gr

from celery import Celery
from datetime import datetime

import federatedscope.organizer.cfg_client as cfg_client
from federatedscope.organizer.module.room import RoomManager
from federatedscope.organizer.module.ecs import ECSManager
from federatedscope.organizer.module.logging import FileLogger, GRLogger

IS_OPEN_ACCORDIN = False
LOG_NAME = str(datetime.now().strftime('organizer_%Y%m%d%H%M%S')) + '.log'


def read_logs():
    # Used for cache `sys.out`
    sys.stdout.flush()
    with open(LOG_NAME, "r") as f:
        return f.read()


# Initialization
organizer = Celery()
organizer.config_from_object(cfg_client)

logger = GRLogger()
room_manager = RoomManager(cfg_client.USER, organizer, logger)
ecs_manager = ECSManager(cfg_client.USER, logger)

sys.stdout = FileLogger(LOG_NAME)

# TODO: delete this line
gr.close_all()

with gr.Blocks() as demo:
    gr.Markdown("Welcome to FederatedScope Cloud Demo!")

    with gr.Tab("Lobby"):
        with gr.Accordion("Display Rooms", open=True):
            lobby_disp_condition = gr.Textbox(label='condition (optinal)',
                                              ines=1)
            lobby_disp_btn = gr.Button("Display Rooms")
            lobby_disp_btn.click(room_manager.display,
                                 inputs=lobby_disp_condition,
                                 outputs=None)

        with gr.Accordion("Get Authorization", open=IS_OPEN_ACCORDIN):
            gr.Markdown("...")

        with gr.Accordion("Create Room", open=IS_OPEN_ACCORDIN):
            gr.Markdown("...")

        with gr.Accordion("Matching", open=IS_OPEN_ACCORDIN):
            gr.Markdown("...")

        with gr.Accordion("Shutdown", open=IS_OPEN_ACCORDIN):
            gr.Markdown("...")

    with gr.Tab("ECS"):
        with gr.Accordion("Display ECS", open=True):
            gr.Markdown("...")

        with gr.Accordion("Add ECS", open=IS_OPEN_ACCORDIN):
            gr.Markdown("...")

        with gr.Accordion("Shutdown ECS", open=IS_OPEN_ACCORDIN):
            gr.Markdown("...")

        with gr.Accordion("Join Room", open=IS_OPEN_ACCORDIN):
            gr.Markdown("...")

    # log block for gradio
    logs = gr.Textbox(label='Log')
    demo.load(read_logs, None, logs, every=1)

# demo.queue().launch()

demo.queue().launch(share=False,
                    server_name="0.0.0.0",
                    debug=True,
                    server_port=7860)
