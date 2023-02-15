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
ecs_manager = ECSManager(cfg_client.USER, logger, room_manager)

sys.stdout = FileLogger(LOG_NAME)

# TODO: delete this line
gr.close_all()

with gr.Blocks() as demo:
    gr.Markdown("Welcome to FederatedScope Cloud Demo!")

    out_block = None  # gr.Textbox(label='Output', lines=10, interactive=False)

    with gr.Tab("Lobby"):
        with gr.Accordion("Display Rooms", open=True):
            lobby_disp_input = gr.Textbox(label='Condition (optional)',
                                          placeholder='dict/{}')
            lobby_disp_btn = gr.Button("Display Rooms")
            lobby_disp_btn.click(room_manager.display,
                                 inputs=lobby_disp_input,
                                 outputs=out_block)

        with gr.Accordion("Get Authorization", open=IS_OPEN_ACCORDIN):
            with gr.Row():
                lobby_auth_input_idx = gr.Number(label='Room ID', value=1)
                lobby_auth_input_password = gr.Textbox(label='Password',
                                                       type='password')
            lobby_disp_btn = gr.Button("Get Authorization")
            lobby_disp_btn.click(
                room_manager.authorize,
                inputs=[lobby_auth_input_idx, lobby_auth_input_password],
                outputs=out_block)

        with gr.Accordion("Create Room", open=IS_OPEN_ACCORDIN):
            with gr.Row():
                lobby_add_input_yaml = gr.File(label="Upload Yaml",
                                               file_count="single",
                                               file_types=[".yaml"])
                with gr.Column():
                    lobby_add_input_opts = gr.Textbox(label='Opts (optional)',
                                                      placeholder='list/[]')
                    lobby_add_input_password = gr.Textbox(label='Password',
                                                          type='password')
            lobby_add_btn = gr.Button("Create Room")
            lobby_add_btn.click(room_manager.add,
                                inputs=[
                                    lobby_add_input_yaml, lobby_add_input_opts,
                                    lobby_add_input_password
                                ],
                                outputs=out_block)

        with gr.Accordion("Matching", open=IS_OPEN_ACCORDIN):
            # TODO: fix!
            with gr.Box():  # "Basic settings"
                gr.Markdown("Basic settings")
                with gr.Row():
                    with gr.Box():
                        gr.Markdown("Data")
                        with gr.Tab("Choose"):
                            lobby_matching_basic_data_choose = gr.Dropdown(
                                ['adult', 'credit', 'abalone', 'blog'],
                                value=['adult'],
                                label='vFL data')
                        with gr.Tab("Upload"):
                            lobby_matching_basic_data_upload = gr.File(
                                label="Upload Data",
                                file_count="single",
                                file_types=["text"])
                    lobby_matching_input_opts = gr.Textbox(label='Opts')
            with gr.Box():  # "Tuning setting"
                gr.Markdown("Tuning setting")
                with gr.Row():
                    lobby_matching_tune_optimizer = gr.Dropdown(
                        ['rs', 'bo_rf', 'bo_gp'],
                        value=['bo_rf'],
                        label='Optimizer')
                    lobby_matching_tune_model = gr.Dropdown(
                        ['lr', 'xgb', 'gbdt'],
                        value=['lr', 'xgb'],
                        multiselect=True,
                        label='Model Selection')
                    lobby_matching_tune_feat = gr.Dropdown(
                        [
                            '', 'min_max_norm', 'instance_norm',
                            'standardization', 'log_transform',
                            'uniform_binning', 'variance_filter', 'iv_filter'
                        ],
                        value=[
                            '', 'min_max_norm', 'instance_norm',
                            'standardization', 'log_transform',
                            'uniform_binning', 'variance_filter', 'iv_filter'
                        ],
                        multiselect=True,
                        label='Feature Engineer')
                with gr.Box():
                    gr.Markdown("Learning Rate")
                    with gr.Row():
                        lobby_matching_tune_min_lr = gr.Slider(0,
                                                               1,
                                                               value=0.1,
                                                               label='Minimum')
                        lobby_matching_tune_max_lr = gr.Slider(0,
                                                               1,
                                                               value=0.8,
                                                               label='Maximum')
                lobby_matching_tune_yaml = gr.File(label="Upload Yaml",
                                                   file_count="single",
                                                   file_types=[".yaml"])
            lobby_matching_btn = gr.Button("Room Matching")
            lobby_matching_btn.click(
                room_manager.matching,
                # TODO: make list of input to dict
                inputs=[
                    lobby_matching_basic_data_choose,
                    lobby_matching_basic_data_upload,
                    lobby_matching_tune_optimizer, lobby_matching_tune_model,
                    lobby_matching_tune_feat, lobby_matching_tune_min_lr,
                    lobby_matching_tune_max_lr, lobby_matching_tune_yaml
                ],
                outputs=out_block)

        with gr.Accordion("Shutdown", open=IS_OPEN_ACCORDIN):
            lobby_shut_input = gr.Number(label='Room ID (optional)', value=0)
            lobby_shut_btn = gr.Button("Shutdown Room(s)")
            lobby_shut_btn.click(room_manager.shutdown,
                                 inputs=lobby_shut_input,
                                 outputs=out_block)

    with gr.Tab("ECS"):
        with gr.Accordion("Join Room", open=True):
            with gr.Row():
                ecs_add_input_idx = gr.Number(label='Room ID', value='1')
                ecs_add_input_ip = gr.Textbox(label='IP Address',
                                              placeholder='127.0.0.1')
                ecs_add_input_opts = gr.Textbox(label='opts (optional)',
                                                placeholder='list/[]')
            ecs_join_btn = gr.Button("Join Room")
            ecs_join_btn.click(ecs_manager.join,
                               inputs=[
                                   ecs_add_input_idx, ecs_add_input_ip,
                                   ecs_add_input_opts
                               ],
                               outputs=out_block)

        with gr.Accordion("Display ECS", open=IS_OPEN_ACCORDIN):
            ecs_disp_input = gr.Textbox(label='Condition (optional)',
                                        placeholder='dict/{}')
            ecs_disp_btn = gr.Button("Display ECS")
            ecs_disp_btn.click(ecs_manager.display,
                               inputs=lobby_disp_input,
                               outputs=out_block)

        with gr.Accordion("Add ECS", open=IS_OPEN_ACCORDIN):
            with gr.Row():
                ecs_add_input_ip = gr.Textbox(label='IP Address',
                                              placeholder='127.0.0.1')
                ecs_add_input_user = gr.Textbox(label='User Name',
                                                placeholder='list/[]')
                ecs_add_input_password = gr.Textbox(label='Password',
                                                    type='password')
            ecs_add_btn = gr.Button("Add ECS")
            ecs_add_btn.click(ecs_manager.add,
                              inputs=[
                                  ecs_add_input_ip, ecs_add_input_user,
                                  ecs_add_input_password
                              ],
                              outputs=out_block)

        with gr.Accordion("Shutdown ECS", open=IS_OPEN_ACCORDIN):
            ecs_shut_input_ip = gr.Textbox(label='IP Address',
                                           placeholder='127.0.0.1')
            ecs_shut_btn = gr.Button("Shutdown ECS")
            ecs_shut_btn.click(ecs_manager.shutdown,
                               inputs=ecs_shut_input_ip,
                               outputs=out_block)

    # log block for gradio
    logs = gr.Textbox(label='Log')
    demo.load(read_logs, None, logs, every=1)

# demo.queue().launch()

demo.queue().launch(share=False,
                    server_name="0.0.0.0",
                    debug=True,
                    server_port=7860)
