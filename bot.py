import os
from typing import Optional, List

from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ann_automl.core.nn_auto import create_classification_model
from ann_automl.utils.time import *

import logging


class ParamsQualityChecker(object):
    def __init__(self):
        self.default_params = {
            'target_accuracy': 0.9,
            'time_limit': None,
            'output_dir': 'model.zip',
            'classes': ['dog', 'cat'],
        }
        self.parameters_mismatch = []

    def type_check(
        self,
        target_accuracy: Optional[float] = None,
        time_limit: Optional[int] = None,
        output_dir: str = 'model.zip',
        classes: Optional[List[str]] = None,
    ):
        if not isinstance(target_accuracy, (type(None), float)):
            target_accuracy = self._mismatch('target_accuracy')
        if not isinstance(time_limit, (type(None), int, float)):
            time_limit = self._mismatch('time_limit')
        if not isinstance(output_dir, (type(None), str)):
            output_dir = self._mismatch('output_dir')
        if not (isinstance(classes, list) and all(isinstance(elem, str) for elem in classes)):
            classes = self._mismatch('classes')

        self.parameters = locals()
        del self.parameters['self']

    def _mismatch(self, param_name: str):
        self.parameters_mismatch.append(param_name)
        return self.default_params[param_name]


class LLMBot(object):
    def __init__(
        self,
        executions_limit=5,
        prompt_data_path="data/lm/init_v0.txt",
        model_path="codellama-7b-instruct.Q4_K_M.gguf",
        temperature=1,
        max_tokens=75,
        n_ctx=2000,
        top_p=1,
        verbose=False,
    ):
        self.executions_limit = executions_limit
        self.prompt_data_path = prompt_data_path
        self.llm = self._llm_init(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            top_p=top_p,
            verbose=verbose,
        )
        self.params_checker = None

    def _llm_init(
        self,
        model_path,
        temperature,
        max_tokens,
        n_ctx,
        top_p,
        verbose,
    ):
        logging.debug('LlamaCpp init ..')
        self.state = 'bot llm init'
        llm = LlamaCpp(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            top_p=top_p,
            verbose=verbose,
        )
        return llm

    def request(self, text_request):
        if text_request[0] != '/':
            self.state = 'request processing'
            yield self.state
            self._text_to_executable_query(text_request)
            
            yield f"""
                I understood the request with following parameters: {self.params_checker.parameters}.\n
                If you want to start model training - type command /ok.\n
                If you want to change some parameters - type command in following way: /parameter=value.
            """

        elif text_request.strip('/') == 'debug':
            logging.getLogger().setLevel(level=logging.DEBUG)
            logging.debug('debug mode activated')
            yield 'debug mode activated'

        elif text_request.strip('/') == 'help':
            yield """
                Formulate your request in the following ways:\n
                "Please, create a model for pet classification and save the model in tflite format. You can optimize the model until tomorrow 12 pm";\n
                "I want to create plant classifier to recognize plants and animals from iPhone camera with high accuracy. I want to get result until the end of the next week";\n
                "Planes classification model in 10 hours";\n
                "We need a classifier for wild animals. You can spend 1 week for it. Save as cl_animals.zip";\n
            """

        elif text_request.strip('/') != 'ok':
            param_change = text_request.strip('/').split('=')
            
            if param_change[0]=='classes':
                _value = param_change[1].split(', ')
            elif param_change[0]=='output_dir':
                _value = param_change[1]
            else:
                _value = float(param_change[1])
            self.params_checker.parameters[param_change[0]] = _value
            
            yield f"""
                I understood the request with following parameters: {self.params_checker.parameters}.\n
                If you want to start model training - type command /ok.\n
                If you want to change some parameters - type command in following way: /parameter=value.
            """
        
        elif text_request.strip('/') == 'ok':
            self.state = 'automl model training'
            yield self.state
            logging.getLogger().setLevel(level=logging.INFO)
            output_model_path = create_classification_model(**self.params_checker.parameters)
            yield os.path.abspath(output_model_path)

    def _text_to_executable_query(
        self,
        text_request: str,
    ):
        executable = False
        executions_count = 0
        while (executable==False) and (executions_count <= self.executions_limit):
            try:
                params_checker = ParamsQualityChecker()

                logging.debug(f'execution try [{executions_count}]: llm model running ...')
                query = self._llm_inference(text_request)
                logging.debug(f'execution try [{executions_count}]: query: {query}')

                query = self._query_parsing(query)
                logging.debug(f'execution try [{executions_count}]: query after processing: {query}')
                
                exec(query)
                self.params_checker = params_checker
                executable = True
            except:
                executions_count += 1
                logging.debug(f'execution try [{executions_count}]: retry')
                pass

    def _llm_inference(self, request):
        prompt = self._prompt_parsing(self.prompt_data_path)
        result = self.llm(prompt + "\nRequest: {" + request + "}")
        return result

    def _query_parsing(self, query):
        return query.split('Code:')[1].split('{')[1].split('}')[0]

    def _prompt_parsing(self, txt_prompt_path):
        with open(txt_prompt_path, "r", encoding='utf8') as f:
            prompt = f.read()
        return prompt
