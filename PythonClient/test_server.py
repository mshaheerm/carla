from concurrent import futures
import time

import grpc

import test_grpc_pb2
import test_grpc_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ProcessGroundTruth(test_grpc_pb2_grpc.GroundtruthdataServicer):

    def ProcessGroundTruth(self, request, context):

        print("client data received: " + str(request))

        return test_grpc_pb2.Empty(result='Client Data Sent')

    # def SayHello(self, request, context):
    #     return test_grpc_pb2.HelloReply(message='Hello, %s!' % request.name)

    def ProcessGThost(self, request, context):
        # print("Ego Vehicle data received: " + str(request))
        return test_grpc_pb2.Empty(result='Client Data Sent')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    test_grpc_pb2_grpc.add_GroundtruthdataServicer_to_server(ProcessGroundTruth(), server)
    server.add_insecure_port('[::]:63558')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
