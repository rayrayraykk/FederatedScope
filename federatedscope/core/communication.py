import grpc
from concurrent import futures

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc
from federatedscope.core.gRPC_server import gRPCComServeFunc
from federatedscope.core.message import Message


class StandaloneCommManager(object):
    """
    The communicator used for standalone mode
    """
    def __init__(self, comm_queue, monitor=None):
        """
        Initialize the internal state of the NeighborsManager.

        Args:
            self: write your description
            comm_queue: write your description
            monitor: write your description
        """
        self.comm_queue = comm_queue
        self.neighbors = dict()
        self.monitor = monitor  # used to track the communication related
        # metrics

    def receive(self):
        """
        Receive a message from the socket.

        Args:
            self: write your description
        """
        # we don't need receive() in standalone
        pass

    def add_neighbors(self, neighbor_id, address=None):
        """
        Add a neighbor to the map.

        Args:
            self: write your description
            neighbor_id: write your description
            address: write your description
        """
        self.neighbors[neighbor_id] = address

    def get_neighbors(self, neighbor_id=None):
        """
        Get neighbors for a given neighbor id.

        Args:
            self: write your description
            neighbor_id: write your description
        """
        address = dict()
        if neighbor_id:
            if isinstance(neighbor_id, list):
                for each_neighbor in neighbor_id:
                    address[each_neighbor] = self.get_neighbors(each_neighbor)
                return address
            else:
                return self.neighbors[neighbor_id]
        else:
            # Get all neighbors
            return self.neighbors

    def send(self, message):
        """
        Send a message to the server.

        Args:
            self: write your description
            message: write your description
        """
        self.comm_queue.append(message)
        download_bytes, upload_bytes = message.count_bytes()
        self.monitor.track_upload_bytes(upload_bytes)


class gRPCCommManager(object):
    """
        The implementation of gRPCCommManager is referred to the tutorial on
        https://grpc.io/docs/languages/python/
    """
    def __init__(self, host='0.0.0.0', port='50050', client_num=2):
        """
        Initializes the grpc server.

        Args:
            self: write your description
            host: write your description
            port: write your description
            client_num: write your description
        """
        self.host = host
        self.port = port
        options = [
            ("grpc.max_send_message_length",
             global_cfg.distribute.grpc_max_send_message_length),
            ("grpc.max_receive_message_length",
             global_cfg.distribute.grpc_max_receive_message_length),
            ("grpc.enable_http_proxy",
             global_cfg.distribute.grpc_enable_http_proxy),
        ]
        self.server_funcs = gRPCComServeFunc()
        self.grpc_server = self.serve(max_workers=client_num,
                                      host=host,
                                      port=port,
                                      options=options)
        self.neighbors = dict()
        self.monitor = None  # used to track the communication related metrics

    def serve(self, max_workers, host, port, options):
        """
        This function is referred to
        https://grpc.io/docs/languages/python/basics/#starting-the-server
        """
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=options)
        gRPC_comm_manager_pb2_grpc.add_gRPCComServeFuncServicer_to_server(
            self.server_funcs, server)
        server.add_insecure_port("{}:{}".format(host, port))
        server.start()

        return server

    def add_neighbors(self, neighbor_id, address):
        """
        Add a neighbor to the graph.

        Args:
            self: write your description
            neighbor_id: write your description
            address: write your description
        """
        if isinstance(address, dict):
            self.neighbors[neighbor_id] = '{}:{}'.format(
                address['host'], address['port'])
        elif isinstance(address, str):
            self.neighbors[neighbor_id] = address
        else:
            raise TypeError(f"The type of address ({type(address)}) is not "
                            "supported yet")

    def get_neighbors(self, neighbor_id=None):
        """
        Get neighbors for a given neighbor id.

        Args:
            self: write your description
            neighbor_id: write your description
        """
        address = dict()
        if neighbor_id:
            if isinstance(neighbor_id, list):
                for each_neighbor in neighbor_id:
                    address[each_neighbor] = self.get_neighbors(each_neighbor)
                return address
            else:
                return self.neighbors[neighbor_id]
        else:
            # Get all neighbors
            return self.neighbors

    def _send(self, receiver_address, message):
        """
        Send a message to a remote gRPC server.

        Args:
            self: write your description
            receiver_address: write your description
            message: write your description
        """
        def _create_stub(receiver_address):
            """
            This part is referred to
            https://grpc.io/docs/languages/python/basics/#creating-a-stub
            """
            channel = grpc.insecure_channel(receiver_address,
                                            options=(('grpc.enable_http_proxy',
                                                      0), ))
            stub = gRPC_comm_manager_pb2_grpc.gRPCComServeFuncStub(channel)
            return stub, channel

        stub, channel = _create_stub(receiver_address)
        request = message.transform(to_list=True)
        try:
            stub.sendMessage(request)
        except grpc._channel._InactiveRpcError:
            pass
        channel.close()

    def send(self, message):
        """
        Sends a message to all connected receivers.

        Args:
            self: write your description
            message: write your description
        """
        receiver = message.receiver
        if receiver is not None:
            if not isinstance(receiver, list):
                receiver = [receiver]
            for each_receiver in receiver:
                if each_receiver in self.neighbors:
                    receiver_address = self.neighbors[each_receiver]
                    self._send(receiver_address, message)
        else:
            for each_receiver in self.neighbors:
                receiver_address = self.neighbors[each_receiver]
                self._send(receiver_address, message)

    def receive(self):
        """
        Receive a message from the server.

        Args:
            self: write your description
        """
        received_msg = self.server_funcs.receive()
        message = Message()
        message.parse(received_msg.msg)
        return message
