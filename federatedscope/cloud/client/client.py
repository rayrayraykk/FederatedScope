import sys
import gradio as gr

from celery import Celery
from datetime import datetime

import federatedscope.cloud.client.config as cfg_client
from federatedscope.cloud.client.module.room import RoomManager
from federatedscope.cloud.client.module.ecs import ECSManager
from federatedscope.cloud.client.module.logger import FileLogger, GRLogger

IS_OPEN_ACCORDION = False
IS_LOG_MODE = True
LOG_NAME = str(datetime.now().strftime('organizer_%Y%m%d%H%M%S')) + '.log'
BANNER = 'https://img.alicdn.com/imgextra/i4/O1CN01BGnHzH2A2vAGLsws0_' \
         '!!6000000008146-2-tps-2320-277.png'


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

# Defining Components
room_df = gr.Dataframe(headers=[
    'idx', 'abstract', 'cfg', 'auth', 'log_file', 'port', 'pid', 'cur_client',
    'max_client'
],
                       col_count=9,
                       interactive=False)

discover_df = gr.Dataframe(
    headers=['idx', 'domain', 'abstract', 'rate', 'status'],
    col_count=5,
    interactive=False)

request_df = gr.Dataframe(
    headers=['idx', 'room_idx', 'domain', 'abstract', 'rate', 'status'],
    col_count=6,
    interactive=False)

process_df = gr.Dataframe(headers=[
    'idx', 'abstract', 'cfg', 'auth', 'log_file', 'port', 'pid', 'cur_client',
    'max_client'
],
                          col_count=9,
                          interactive=False)

with gr.Blocks() as demo:
    gr.Image(type="pil", value=BANNER, interactive=False, shape=(1, 2))

    out_block = None  # gr.Textbox(label='Output', lines=10, interactive=False)

    with gr.Accordion("Instrument panel", open=True):
        with gr.Tab("Room List"):
            room_df.render()
            lobby_disp_input = gr.Textbox(label='Display Condition (optional)',
                                          placeholder='dict/{}')
            lobby_disp_btn = gr.Button("Display Room")
            lobby_disp_btn.click(room_manager.display,
                                 inputs=lobby_disp_input,
                                 outputs=room_df)
        with gr.Tab("Discover"):
            discover_df.render()
            with gr.Row():
                lobby_disc_input_idx = gr.Number(label='Target Room ID',
                                                 value=1)
                lobby_disc_input_domain = gr.Dropdown(
                    ['tabular', 'image', 'text', 'graph'],
                    value=['tabular'],
                    label='Domain')
                lobby_disc_input_key = gr.Dropdown(['User ID', 'Item ID'],
                                                   value=['tabular'],
                                                   label='Primary Key')
            with gr.Box():
                gr.Markdown('Extract Statistics')
                with gr.Row():
                    lobby_disc_input_column = gr.Number(label='Target Column',
                                                        value=1)
                    lobby_disc_input_extract = gr.Dropdown(['DP', 'SS', 'HE'],
                                                           value=['DP'],
                                                           label='Encryption')
            lobby_disc_btn = gr.Button("Discover")
            lobby_disc_btn.click(room_manager.discover,
                                 inputs=[
                                     lobby_disc_input_idx,
                                     lobby_disc_input_domain,
                                     lobby_disc_input_key,
                                     lobby_disc_input_column,
                                     lobby_disc_input_extract
                                 ],
                                 outputs=discover_df)
            lobby_cand_idx = gr.Number(label='Invite')
            lobby_send_req_btn = gr.Button("Send Request")
            lobby_send_req_btn.click(room_manager.send_request,
                                     inputs=lobby_cand_idx,
                                     outputs=discover_df)
        with gr.Tab("Request"):
            request_df.render()
            lobby_disp_req_btn = gr.Button("Display Request")
            lobby_disp_req_btn.click(room_manager.display_request,
                                     inputs=None,
                                     outputs=request_df)
            lobby_req_idx = gr.Number(label='Agree idx of request')
            lobby_resb_req_btn = gr.Button("Respond to request")
            lobby_resb_req_btn.click(room_manager.respond_request,
                                     inputs=lobby_req_idx,
                                     outputs=request_df)
        # with gr.Tab("Process"):
        #     process_df.render()
        #     lobby_disp_proc_btn = gr.Button("Display Process")
        #     lobby_disp_proc_btn.click(room_manager.display_process,
        #                               inputs=None,
        #                               outputs=process_df)
    with gr.Accordion("Console", open=True):
        with gr.Tab("Lobby"):
            with gr.Accordion("Authorization", open=IS_OPEN_ACCORDION):
                with gr.Row():
                    lobby_auth_input_idx = gr.Number(label='Room ID', value=1)
                    lobby_auth_input_password = gr.Textbox(label='Password',
                                                           type='password')
                lobby_auth_btn = gr.Button("Get Authorization")
                lobby_auth_btn.click(
                    room_manager.authorize,
                    inputs=[lobby_auth_input_idx, lobby_auth_input_password],
                    outputs=out_block)
                with gr.Row():
                    lobby_auth_app_idx = gr.Number(label='Applicant ID',
                                                   value=1)
                    lobby_auth_app_action = gr.Dropdown(['agree', 'reject'],
                                                        value=['agree'],
                                                        label='Action')
                lobby_auth_btn = gr.Button("Send Authorization")
                lobby_auth_btn.click(
                    room_manager.send_authorize,
                    inputs=[lobby_auth_app_idx, lobby_auth_app_action],
                    outputs=out_block)

            with gr.Accordion("Create Room", open=IS_OPEN_ACCORDION):
                with gr.Accordion("Basic settings", open=True):
                    with gr.Row():
                        with gr.Tab("Choose from FS"):
                            lobby_add_data_choose = gr.Dropdown(
                                [
                                    '--------tabular--------',
                                    'adult',
                                    'credit',
                                    'abalone',
                                    'blog',
                                    '--------image--------',
                                    'FEMNIST',
                                    'CIFAR10',
                                    '--------text--------',
                                    'Twitter',
                                    '--------graph--------',
                                    'Cora',
                                    'CiteSeer',
                                    'PubMed',
                                ],
                                multiselect=False,
                                label='Choose Data')
                        with gr.Tab("Upload Data"):
                            lobby_add_data_upload = gr.File(
                                label="Upload Data", file_count="single")
                        with gr.Column():
                            lobby_add_scenario = gr.Dropdown(['vFL', 'hFL'],
                                                             value=['vFL'],
                                                             label='Scenario')
                            lobby_add_domain = gr.Dropdown(
                                ['tabular', 'image', 'text', 'graph'],
                                value=['tabular'],
                                label='Domain')
                    with gr.Row():
                        lobby_add_input_yaml = gr.File(label="Upload Yaml",
                                                       file_count="single",
                                                       file_types=[".yaml"])
                        with gr.Column():
                            lobby_add_input_private = gr.Checkbox(
                                label="Private")
                            lobby_add_input_opts = gr.Textbox(
                                label='Opts (optional)', placeholder='list/[]')
                            lobby_add_input_password = gr.Textbox(
                                label='Password', type='password')

                add_basic_input = [
                    lobby_add_data_upload, lobby_add_data_choose,
                    lobby_add_scenario, lobby_add_domain, lobby_add_input_yaml,
                    lobby_add_input_private, lobby_add_input_opts,
                    lobby_add_input_password
                ]

                with gr.Accordion("Tuning Center", open=IS_OPEN_ACCORDION):
                    with gr.Row():
                        lobby_add_tune_optimizer = gr.Dropdown(
                            [
                                'rs', 'bo_rf', 'bo_gp', 'bo_kde', 'fedex',
                                'pfedhpo', 'fts'
                            ],
                            value=['bo_rf'],
                            label='Optimizer')
                        lobby_add_tune_model = gr.Dropdown(
                            [
                                '--------tabular--------',
                                'lr',
                                'xgb',
                                'gbdt',
                                'rf',
                                '--------image--------',
                                'convnet2',
                                'convnet5',
                                'vgg11',
                                'resnet18',
                                '--------text--------',
                                'lstm',
                                'transformer',
                                '--------graph--------',
                                'gcn',
                                'sage',
                                'gpr',
                                'gat',
                                'gin',
                                'mpnn',
                            ],
                            multiselect=True,
                            label='Model Selection')
                        lobby_add_tune_feat = gr.Dropdown(
                            [
                                '', 'min_max_norm', 'instance_norm',
                                'standardization', 'log_transform',
                                'uniform_binning', 'variance_filter',
                                'iv_filter'
                            ],
                            value=[''],
                            multiselect=True,
                            label='Feature Engineer')
                        lobby_add_tune_pruning = gr.Dropdown(
                            ['none', 'median', 'patience', 'threshold'],
                            value=['none'],
                            multiselect=False,
                            label='Pruning')
                    with gr.Box():
                        gr.Markdown("Learning Rate")
                        with gr.Row():
                            lobby_add_tune_min_lr = gr.Slider(0,
                                                              1,
                                                              value=0.0,
                                                              label='Minimum')
                            lobby_add_tune_max_lr = gr.Slider(0,
                                                              1,
                                                              value=0.0,
                                                              label='Maximum')
                    with gr.Box():
                        gr.Markdown("Weight Decay")
                        with gr.Row():
                            lobby_add_tune_min_wd = gr.Slider(0,
                                                              1,
                                                              value=0.0,
                                                              label='Minimum')
                            lobby_add_tune_max_wd = gr.Slider(0,
                                                              1,
                                                              value=0.0,
                                                              label='Maximum')

                add_tune_input = [
                    lobby_add_tune_optimizer, lobby_add_tune_model,
                    lobby_add_tune_feat, lobby_add_tune_pruning,
                    lobby_add_tune_min_lr, lobby_add_tune_max_lr,
                    lobby_add_tune_min_wd, lobby_add_tune_max_wd
                ]
                lobby_add_btn = gr.Button("Create Room")
                lobby_add_btn.click(room_manager.create,
                                    inputs=add_basic_input + add_tune_input,
                                    outputs=out_block)

            with gr.Accordion("Shutdown", open=IS_OPEN_ACCORDION):
                lobby_shut_input = gr.Number(label='Room ID (optional)',
                                             value=0)
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

            with gr.Accordion("Display ECS", open=IS_OPEN_ACCORDION):
                ecs_disp_input = gr.Textbox(label='Condition (optional)',
                                            placeholder='dict/{}')
                ecs_disp_btn = gr.Button("Display ECS")
                ecs_disp_btn.click(ecs_manager.display,
                                   inputs=lobby_disp_input,
                                   outputs=out_block)

            with gr.Accordion("Add ECS", open=IS_OPEN_ACCORDION):
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

            with gr.Accordion("Shutdown ECS", open=IS_OPEN_ACCORDION):
                ecs_shut_input_ip = gr.Textbox(label='IP Address',
                                               placeholder='127.0.0.1')
                ecs_shut_btn = gr.Button("Shutdown ECS")
                ecs_shut_btn.click(ecs_manager.shutdown,
                                   inputs=ecs_shut_input_ip,
                                   outputs=out_block)
        with gr.Tab('Monitor'):
            # TODO: task manager to show the details
            url = 'http://39.103.132.84:8080/vfl_demo/vfl_demo?workspace' \
                  '=user-vfl_demo'
            iframe = f'<iframe src={url} ' \
                     f'style="border:none;height:1024px;width:100%">'
            gr.HTML(iframe)

    # log block for gradio
    if IS_LOG_MODE:
        logs = gr.Textbox(label='Log', max_lines=5)
        demo.load(read_logs, None, logs, every=1)

# demo.queue().launch()
if IS_LOG_MODE:
    demo.queue().launch(share=False,
                        server_name="0.0.0.0",
                        debug=True,
                        server_port=7860)
else:
    demo.launch(share=False,
                server_name="0.0.0.0",
                debug=True,
                server_port=7860)
