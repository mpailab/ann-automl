import enum
import os
import threading
from sys import argv
from time import sleep
from typing import Union

from origamibot import OrigamiBot as Bot
from origamibot.listener import Listener

from matplotlib import pyplot as plt

from ann_automl.core.nn_auto import create_classification_model
from ann_automl.core.nnfuncs import StopFlag, ExperimentLog, multithreading_mode
from ann_automl.lm.lm_funcs import LMWorker, get_params_from_request
from ann_automl.utils.process import process
from ann_automl.utils.thread_wrapper import ObjectWrapper
from ann_automl.utils.time import time_as_str


plt_lock = threading.RLock()


class ChatThread:
    def __init__(self, bot: Bot, chat_id: int, lm_worker: Union[LMWorker, ObjectWrapper]):
        self.bot = bot
        self.chat_id = chat_id
        self.training = False
        self.lm_worker = lm_worker
        self.stop_flag = StopFlag()
        self.result_file_path = None
        self.train_process = None
        self.best_accuracy = 0
        self.exp_log = None
        # create directorty for logging information about experiments in this chat
        self.exp_log_dir = f'data/bot/{chat_id}'
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self._emulation = False
        self._debug = False
        self._processing = False
        self.requests = f'{self.exp_log_dir}/requests.txt'

    def __del__(self):
        self.lm_worker.join_thread()

    def emulation(self, message, on_off):
        if str(on_off).lower() in ['on', '1', 'true']:
            self._emulation = True
            self.bot.send_message(self.chat_id, 'Emulation mode is on')
        elif str(on_off).lower() in ['off', '0', 'false']:
            self._emulation = False
            self.bot.send_message(self.chat_id, 'Emulation mode is off')
        else:
            self.bot.send_message(self.chat_id, 'Unknown parameter. Use /emulation on or /emulation off')

    def debug(self, message, on_off):
        if str(on_off).lower() in ['on', '1', 'true']:
            self._debug = True
            self.bot.send_message(self.chat_id, 'Debug mode is on')
        elif str(on_off).lower() in ['off', '0', 'false']:
            self._debug = False
            self.bot.send_message(self.chat_id, 'Debug mode is off')
        else:
            self.bot.send_message(self.chat_id, 'Unknown parameter. Use /debug on or /debug off')

    def stop(self, message, return_result):
        if self.training:
            if not return_result:
                self.result_file_path = None
            self.stop_flag()
            msg = self.bot.send_message(self.chat_id, 'Wait for training to stop...')
            self.train_process.wait()
            self.bot.edit_message_text(self.chat_id, 'Training stopped', msg.message_id)
            self.training = False
        else:
            if self._emulation:
                self.bot.send_message(message.chat.id, 'Stop command recieved (bot in emulation mode)')
            else:
                self.bot.send_message(self.chat_id, 'Nothing to stop, no processes were started')

    def accuracy(self, message):
        if self.training:
            if self.exp_log.best_val_acc > 0:
                msg = f'Currently, best achieved accuracy is:\n' \
                      f'  {self.exp_log.best_val_acc:.4f} on validation set'
                if self.exp_log.best_acc > 0:
                    msg += f'\n   {self.exp_log.best_acc:.4f} on test set'
                self.bot.send_message(self.chat_id, msg)
            else:
                self.bot.send_message(self.chat_id, 'Currently, waiting for start training process')
        elif self.best_accuracy > 0:
            self.bot.send_message(self.chat_id, f'Accuracy achieved during last training process was {self.best_accuracy}')
        else:
            self.bot.send_message(self.chat_id, 'No training process was started after last bot server restart')

    def _on_train_finished(self, *args, **kwargs):
        self.training = False
        self.best_accuracy = self.exp_log.best_acc
        self.result_file_path = self.train_process.value
        try:
            if self.result_file_path is not None and os.path.exists(self.result_file_path):
                with open(self.result_file_path, 'rb') as f:
                    self.bot.send_document(self.chat_id, f, caption=f'Your model is ready. Best accuracy achieved: {self.best_accuracy}')
            else:
                self.bot.send_message(self.chat_id, 'Training finished, but no result was produced')
        except Exception as e:
            self.bot.send_message(self.chat_id, f'Training finished, but error occured while sending result: {e}')

    def debug_msg(self, message):
        if self._debug:
            self.bot.send_message(self.chat_id, "Debug message:\n"+message)

    def _start_train(self, params):
        self.result_file_path = os.path.join(self.exp_log_dir, f'{params["output_dir"]}')
        params['output_dir'] = self.result_file_path
        self.exp_log = ExperimentLog()
        self.train_process = process(create_classification_model)(**params, start=False, stop_flag=self.stop_flag,
                                                                  exp_log=self.exp_log)
        self.train_process.set_handler('message', lambda msg: self.bot.send_message(self.chat_id, msg))
        self.train_process.set_handler('print', lambda msg: self.debug_msg(msg))
        self.train_process.on_finish = self._on_train_finished
        self.train_process.start()
        self.training = True

    def plot(self, message):
        if self._emulation:
            self._emulate_plot(message)
        with plt_lock:
            if self.exp_log is not None:
                curr = self.exp_log.current_run
                if curr is not None:
                    acc, val_acc = curr.acc, curr.val_acc
                    loss, val_loss = curr.loss, curr.val_loss
                    if len(acc) > 0:
                        x = list(range(1, len(acc)+1))
                        # plot accuracy (left axis) and loss (right axis)
                        fig, (axacc, axloss) = plt.subplots(1, 2, figsize=(12, 6))
                        axacc.plot(x, acc, label='train')
                        axacc.plot(x, val_acc, label='val')
                        axacc.set_xlabel('epoch')
                        axacc.set_ylabel('accuracy')
                        axacc.legend()
                        axloss.plot(x, loss, label='train')
                        axloss.plot(x, val_loss, label='val')
                        axloss.set_xlabel('epoch')
                        axloss.set_ylabel('loss')
                        axloss.legend()
                        plt.tight_layout()
                        fig_path = os.path.join(self.exp_log_dir, 'plot.png')
                        plt.savefig(fig_path)
                        with open(fig_path, 'rb') as f:
                            self.bot.send_photo(self.chat_id, f, caption='Current training process')
                            return
                if self.training:
                    self.bot.send_message(self.chat_id, 'No information available to plot. Maybe first epoch is not finished yet?')
                else:
                    self.bot.send_message(self.chat_id, 'No active training processes now')
            else:
                self.bot.send_message(self.chat_id, 'No training processes were started after last bot server restart')

    def _emulate_plot(self, message):
        with plt_lock:
            # draw plot with matplotlib and send it to user
            epochs = list(range(1, 11))
            train_loss = [0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.1, 0.08, 0.07, 0.07]
            val_loss = [0.6, 0.5, 0.4, 0.3, 0.2, 0.21, 0.22, 0.22, 0.24, 0.23]
            train_acc = [0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
            val_acc = [0.4, 0.5, 0.6, 0.7, 0.8, 0.79, 0.78, 0.79, 0.76, 0.77]
            # draw plot in 2 columns (left - loss, right - accuracy)
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor('xkcd:white')
            ax[0].plot(epochs, train_loss, label='train')
            ax[0].plot(epochs, val_loss, label='val')
            ax[0].set_title('Loss')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].legend()
            ax[1].plot(epochs, train_acc, label='train')
            ax[1].plot(epochs, val_acc, label='val')
            ax[1].set_title('Accuracy')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend()
            # save plot to file
            plt.savefig('plot.png')
            # send plot to user
            with open('plot.png', 'rb') as f:
                self.bot.send_photo(message.chat.id, f, caption='Training plot (bot in emulation mode)')

    def process_message(self, message):
        if self._processing:
            self.bot.send_message(message.chat.id, 'Previous request is now processed. Wait and then try again')
            return
        try:
            with open(self.requests, 'a') as f:
                f.write("---\nRequest: {" + message.text + "}\n")
            self._processing = True
            if message.text[0] == '/':
                return
            if self.training:
                self.bot.send_message(self.chat_id, 'Previous training process is still running.\n'
                                                    'Wait for it to finish or stop it (/stop), then ask again.')
                return

            self.bot.send_message(self.chat_id, 'Processing request ...')
            # parse message
            params = self.lm_worker.run(get_params_from_request, message.text)
            with open(self.requests, 'a') as f:
                f.write("Params: {" + str(params) + "}\n")
            if isinstance(params, str):
                self.bot.send_message(self.chat_id, params)
                return
            text = "Parameters:\n"
            text += f"classes = [{', '.join(params['classes'])}]\n"
            text += f"time_limit = {time_as_str(params['time_limit'])}\n"
            text += f"for_mobile = {params['for_mobile']}"
            if params['optimize_over_target']:
                text += f"\nstop criterion: by timeout"
            else:
                text += f"\nstop criterion: by timeout or when accuracy >= {params['target_accuracy']}"
            self.bot.send_message(self.chat_id, text)
            if self._emulation:
                self.bot.send_message(self.chat_id, 'Emulation mode is on. Training process will not be started.')
            else:
                self._start_train(params)
        except Exception as e:
            self._processing = False
            self.bot.send_message(self.chat_id, f'Error: {e}')
        finally:
            self._processing = False


class BotsCommands:
    def __init__(self, bot: Bot, lm_worker: Union[LMWorker, ObjectWrapper], chats):
        self.bot = bot
        self.lm_worker = lm_worker
        self.chats = chats

    def _chat(self, chat_id):
        if chat_id not in self.chats:
            self.chats[chat_id] = ChatThread(self.bot, chat_id, self.lm_worker)
        return self.chats[chat_id]

    def start(self, message):  # /start command
        self.bot.send_message(message.chat.id,
                              'Hello!\n'
                              'This is bot for testing natural language interface of ANN-AutoML.\n'
                              'You can ask me to create neural network for some image classification task.\n'
                              'For example, you can ask me to create neural network for classification of animals.\n'
                              'I will try to understand your request and create neural network for you.')

    def echo(self, message, value: str):  # /echo [value: str] command
        self.bot.send_message(message.chat.id, value)

    def stop(self, message):
        chat = self._chat(message.chat.id)
        chat.stop(message, return_result=True)

    def cancel(self, message):
        chat = self._chat(message.chat.id)
        chat.stop(message, return_result=False)

    def plot(self, message):
        chat = self._chat(message.chat.id)
        chat.plot(message)

    def accuracy(self, message):
        chat = self._chat(message.chat.id)
        chat.accuracy(message)

    def emulation(self, message, value: str):
        chat = self._chat(message.chat.id)
        chat.emulation(message, value)

    def debug(self, message, value: str):
        chat = self._chat(message.chat.id)
        chat.debug(message, value)


class MessageListener(Listener):  # Event listener must inherit Listener
    def __init__(self, bot, lm_worker, chats):
        self.bot = bot
        self.lm_worker = lm_worker
        self.chats = chats
        self.m_count = 0

    def _chat(self, chat_id):
        if chat_id not in self.chats:
            self.chats[chat_id] = ChatThread(self.bot, chat_id, self.lm_worker)
        return self.chats[chat_id]

    def on_message(self, message):   # called on every message
        self.m_count += 1
        print(f'Recieved message from {message.chat.id} ({self.m_count} total)')
        if message.text[0] == '/':
            cmd = message.text.split()[0]
            if cmd not in ['/start', '/stop', '/cancel', '/plot', '/accuracy', '/emulation', '/debug', '/echo']:
                self.bot.send_message(message.chat.id, 'Unknown command ' + cmd)
                return
        self._chat(message.chat.id).process_message(message)
        # self.bot.send_message(message.chat.id, f'Current message count (after bot restart): {self.m_count}, message: {message.text}')

    def on_command_failure(self, message, err=None):  # When command fails
        if err is None:
            self.bot.send_message(message.chat.id, 'Command failed to bind arguments!')
        else:
            self.bot.send_message(message.chat.id, f'Error in command:\n{err}')


def main():
    # set torch default device to gpu 1
    import torch
    import tensorflow as tf

    torch.cuda.set_device(0)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs = {gpus}")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[1:], 'GPU')
            for gpu in gpus[1:]:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    try:
        with open('token.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        token = input('Enter bot token: ')
        with open('token.txt', 'w') as f:
            f.write(token)

    chats = {}
    bot = Bot(token)
    lm_worker = ObjectWrapper(LMWorker)
    try:
        bot.add_listener(MessageListener(bot, lm_worker, chats))
        bot.add_commands(BotsCommands(bot, lm_worker, chats))
        bot.start()   # start bot's threads
        print('Bot started')
        while True:
            sleep(1)
    finally:
        lm_worker.join_thread()


if __name__ == '__main__':
    with multithreading_mode():
        main()
