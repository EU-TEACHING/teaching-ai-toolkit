import os
from pydoc import locate


if __name__ == '__main__':
    service_class = locate(os.environ['SERVICE_TYPE'])
    service = service_class()
    service()