import os, time, numpy

os.environ["SERVICE_TYPE"] = ""
os.environ["SERVICE_NAME"] = ""
os.environ['INPUT_SIZE'] = str(1)
os.environ['LAYERS'] = str(1)
os.environ['UNITS'] = str(20)
os.environ['LEAKY'] = str(0.8)
os.environ['RHO'] = str(0.9)
os.environ['CONNECTIVITY'] = str(1.0)
os.environ['N_CLASSES'] = str(1)
os.environ['MODEL_PATH'] = "./app/storage/model_float16.tflite"
os.environ['USE_FLOAT16'] = str(1)
from base.communication.packet import DataPacket
from modules.or_module_tf import ORModule

if __name__ == "__main__":

    # this should be run disabling the @TEACHINGNode(produce=True, consume=True)
    # decorator

    or_obj = ORModule()
    arr = numpy.random.rand(224,224,3)

    msg = DataPacket(
            topic='topic',
            body={"img": str(arr.flatten().tolist())}
        )

    def fake_generator():
        while True:
            time.sleep(0.2)
            yield msg

    for res in or_obj(fake_generator()):
        if isinstance(res, DataPacket):
            print("All good here!")
            break
        else:
            raise ValueError