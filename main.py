import os
from pydoc import locate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    service_class = locate(os.environ['SERVICE_TYPE'])
    service = service_class()
    service()