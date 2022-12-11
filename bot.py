from sys import argv
from time import sleep

from origamibot import OrigamiBot as Bot
from origamibot.listener import Listener

from matplotlib import pyplot as plt


class BotsCommands:
    def __init__(self, bot: Bot):  # Can initialize however you like
        self.bot = bot

    def start(self, message):   # /start command
        self.bot.send_message(
            message.chat.id,
            'Hello user!\nThis is an example bot.')

    def echo(self, message, value: str):  # /echo [value: str] command
        self.bot.send_message(
            message.chat.id,
            value
            )

    def stop(self, message):  # /add [a: float] [b: float]
        self.bot.send_message(message.chat.id, 'Stopping training (bot in emulation mode)')

    def accuracy(self, message):
        self.bot.send_message(message.chat.id, 'Accuracy: 0.5 (bot in emulation mode)')

    def plot(self, message):
        # draw plot with matplotlib and send it to user
        epochs = list(range(1, 11))
        train_loss = [0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        val_loss = [0.6, 0.5, 0.4, 0.3, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25]
        train_acc = [0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
        val_acc = [0.4, 0.5, 0.6, 0.7, 0.8, 0.79, 0.78, 0.77, 0.76, 0.75]
        # draw plot in 2 columns (left - loss, right - accuracy)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
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


class MessageListener(Listener):  # Event listener must inherit Listener
    def __init__(self, bot):
        self.bot = bot
        self.m_count = 0

    def on_message(self, message):   # called on every message
        self.m_count += 1
        print(f'Total messages: {self.m_count}')
        self.bot.send_message(message.chat.id, f'Current message count (after bot restart): {self.m_count}')

    def on_command_failure(self, message, err=None):  # When command fails
        if err is None:
            self.bot.send_message(message.chat.id, 'Command failed to bind arguments!')
        else:
            self.bot.send_message(message.chat.id, 'Error in command:\n{err}')


if __name__ == '__main__':
    try:
        with open('token.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        token = input('Enter bot token: ')
        with open('token.txt', 'w') as f:
            f.write(token)

    bot = Bot(token)
    bot.add_listener(MessageListener(bot))
    bot.add_commands(BotsCommands(bot))
    bot.start()   # start bot's threads
    while True:
        sleep(1)
