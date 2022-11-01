import queue
from collections import deque

from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc


class gRPCComServeFunc(gRPC_comm_manager_pb2_grpc.gRPCComServeFuncServicer):
    def __init__(self):
        """
        Initialize the internal message queue.

        Args:
            self: write your description
        """
        self.msg_queue = deque()

    def sendMessage(self, request, context):
        """
        Adds a message to the message queue and returns a response.

        Args:
            self: write your description
            request: write your description
            context: write your description
        """
        self.msg_queue.append(request)

        return gRPC_comm_manager_pb2.MessageResponse(msg='ACK')

    def receive(self):
        """
        Receive a message from the queue.

        Args:
            self: write your description
        """
        while len(self.msg_queue) == 0:
            continue
        msg = self.msg_queue.popleft()
        return msg
