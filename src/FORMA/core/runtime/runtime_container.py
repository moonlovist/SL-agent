import logging

from dotenv import load_dotenv

from FORMA.core.llm import ThisIsModel


class RuntimeContainer:

    def __init__(self, configs):
        self.configs = configs
        self._models = {}

    def get_model(self, type):
        if type not in self._models:
            config = self.configs.model.__dict__[type]
            self._models[type] = ThisIsModel(config).create_client()
        return self._models[type]
