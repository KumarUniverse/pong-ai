# A Q learning Pong AI agent built using Python.
# The depth dimension of the neural network comes first. depth = k frames.

import pyglet
from pyglet import shapes
from pyglet import text

import math
import random
import time
#import multiprocessing
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import io
import os
#import sys
#import resource
#import logging
#import pickle

from collections import deque
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import array_to_img
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # suppress tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only log the FATAL errors.
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.get_logger().setLevel('INFO')

# Global variables
DIRECTION = {"IDLE": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
# winning_score = 11

# def get_memory():
#     with open('/proc/meminfo', 'r') as mem:
#         free_memory = 0
#         for i in mem:
#             sline = i.split()
#             if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
#                 free_memory += int(sline[1])
#     return free_memory

# def memory_limit():
#     _, hard = resource.getrlimit(resource.RLIMIT_AS)
#     resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 // 4, hard))


class Ball():
    """The ball object (The cube that bounces back and forth)."""

    def __init__(self, pong_game):
        self.width = 18 #1000 // 10 # for debugging, need to change back to 18.
        self.height = 18 #1000 // 10 # for debugging, need to change back to 18.
        self.x = pong_game.canvas_width // 2
        self.y = pong_game.canvas_height // 2
        self.move_x = DIRECTION["IDLE"]
        self.move_y = DIRECTION["IDLE"]
        self.speed = 10.8166538264 #9


class Paddle():
    """The paddle object (The 2 lines that move up and down)."""

    def __init__(self, pong_game, side):
        self.width = 18
        self.height = 70
        self.x = 150 if side == 'left' else pong_game.canvas_width - 150
        self.y = (pong_game.canvas_width // 2) - (self.height // 2)
        self.score = 0
        self.move = DIRECTION["IDLE"]
        self.speed = 6 # (ball.speed / 1.5)


class PongGame(pyglet.window.Window):
    """The Pong game."""

    def __init__(self):
        self.canvas_width = 1400  # 700
        self.canvas_height = 1000  # 500

        self.ai1 = Paddle(self, 'left')  # Q agent
        self.ai2 = Paddle(self, 'right')
        self.ball = Ball(self)
        #self.ai1.height = self.canvas_height // 3 # for debugging, need to remove.
        # self.ai1.speed = 9 # comment out after debugging.
        # self.ai2.speed = 9 # comment out after debugging.
        self.original_ball_speed = self.ball.speed
        self.winning_score = 11
        self.qagent = None

        # self.ai2.speed = 8  # make ai2's paddle speed slower than ai1.
        self.turn = self.ai2  # it's the ai2's turn first.
        self.qlearn_mode = False
        # self.sim_sample_nums = []  # x values of visited percents graph.
        # self.visited_percents = [] # y values of visited percents graph.
        self.qagent_action = 0
        self.ball_init_direction = 1 # 1 means UP, 2 means DOWN

        ########################vvv PYGLET CODE vvv################################################

        super().__init__(width=self.canvas_width//6, height=self.canvas_height//6, caption='Q-learning Pong',
            resizable=True)
        #super().set_visible(False) # hide the Pyglet window. COMMENT OUT TO DISPLAY PONG GAME.
        # create the paddles and the ball.
        self.paddle_colors = (255,255,255) # paddle color is white
        self.ai1_rect = shapes.Rectangle(self.ai1.x, self.canvas_height-self.ai1.height-self.ai1.y,
            self.ai1.width, self.ai1.height, color=self.paddle_colors)
        self.ai2_rect = shapes.Rectangle(self.ai2.x, self.canvas_height-self.ai2.height-self.ai2.y,
            self.ai2.width, self.ai2.height, color=self.paddle_colors)
        self.ball_rect = shapes.Rectangle(self.ball.x, self.canvas_height-self.ball.height-self.ball.y,
            self.ball.width, self.ball.height, color=self.paddle_colors)
        self.line = shapes.Line(self.canvas_width//2, 0, self.canvas_width//2, self.canvas_height)

        # AI1's score:
        self.font = 'Courier New'
        self.fsize = 50
        self.font_ratio = self.fsize / self.canvas_height
        self.ai1_scoreboard_x = (self.canvas_width // 2) - 500
        self.ai2_scoreboard_x = (self.canvas_width // 2) + 200
        self.ai_scoreboard_y = 800
        self.ai1_scoreboard_x_ratio = self.ai1_scoreboard_x / self.canvas_width
        self.ai2_scoreboard_x_ratio = self.ai2_scoreboard_x / self.canvas_width
        self.ai_scoreboard_y_ratio = self.ai_scoreboard_y / self.canvas_height
        self.ai1_scoreboard = text.Label("AI1: " + str(self.ai1.score), font_name=self.font, font_size=self.fsize,
            x=self.ai1_scoreboard_x, y=self.ai_scoreboard_y)
        # AI2's score:
        self.ai2_scoreboard = text.Label("AI2: " + str(self.ai2.score), font_name=self.font, font_size=self.fsize,
            x=self.ai2_scoreboard_x, y=self.ai_scoreboard_y)

        # AI1's current action:
        qaction = "IDLE 0"
        if self.qagent_action < 0:
            qaction = "UP  1"
        elif self.qagent_action > 0:
            qaction = "DOWN   2"
        self.ai1_action_x = self.canvas_width // 2 - 300
        self.ai1_action_y = 200
        self.ai1_action_x_ratio = self.ai1_action_x / self.canvas_width
        self.ai1_action_y_ratio = self.ai1_action_y / self.canvas_height
        self.ai1_action = text.Label("Action: " + qaction, font_name=self.font, font_size=self.fsize,
            x=self.ai1_action_x, y=self.ai1_action_y)


    def on_draw(self):
        """Draws elements of the Pong Pyglet canvas every time the canvas is asked to be rendered."""
        self.clear()

        self.ai1_rect.draw()
        self.ai2_rect.draw()
        self.ball_rect.draw()
        self.line.draw()

        self.ai1_scoreboard.draw()
        self.ai2_scoreboard.draw()
        self.ai1_action.draw()

    def update(self):
        """Updates the positions and sizes of elements in the Pong canvas."""
        curr_game_width, curr_game_height = self.get_size()

        # Make the Pong game GUI dynamically scale based on the size of the pyglet window.
        game_width_ratio = curr_game_width / self.canvas_width
        game_height_ratio = curr_game_height / self.canvas_height

        self.ai1_rect.width = self.ai1.width * game_width_ratio
        self.ai1_rect.height = self.ai1.height * game_height_ratio
        self.ai1_rect.x = self.ai1.x * game_width_ratio
        self.ai1_rect.y = curr_game_height - self.ai1_rect.height \
            - (self.ai1.y * game_height_ratio)

        self.ai2_rect.width = self.ai2.width * game_width_ratio
        self.ai2_rect.height = self.ai2.height * game_height_ratio
        self.ai2_rect.x = self.ai2.x * game_width_ratio
        self.ai2_rect.y = curr_game_height - self.ai2_rect.height \
            - (self.ai2.y * game_height_ratio)

        self.ball_rect.width = self.ball.width * game_width_ratio
        self.ball_rect.height = self.ball.height * game_height_ratio
        self.ball_rect.x = self.ball.x * game_width_ratio
        self.ball_rect.y = curr_game_height - self.ball_rect.height \
            - (self.ball.y * game_height_ratio)

        self.line.x = curr_game_width / 2
        self.line.x2 = self.line.x
        #self.line.y = 0
        self.line.y2 = curr_game_height

        self.ai1_scoreboard.text = "AI1: " + str(self.ai1.score)
        self.ai1_scoreboard.font_size = curr_game_height * self.font_ratio
        self.ai1_scoreboard.x = curr_game_width * self.ai1_scoreboard_x_ratio
        self.ai1_scoreboard.y = curr_game_height * self.ai_scoreboard_y_ratio

        self.ai2_scoreboard.text = "AI2: " + str(self.ai2.score)
        self.ai2_scoreboard.font_size = self.ai1_scoreboard.font_size
        self.ai2_scoreboard.x = curr_game_width * self.ai2_scoreboard_x_ratio
        self.ai2_scoreboard.y = curr_game_height * self.ai_scoreboard_y_ratio

        qaction = "IDLE 0"
        if self.qagent_action is None or self.qagent_action < 0:
            qaction = "UP  1"
        elif self.qagent_action > 0:
            qaction = "DOWN   2"
        self.ai1_action.text = "Action: " + qaction
        self.ai1_action.font_size = self.ai1_scoreboard.font_size
        self.ai1_action.x = curr_game_width * self.ai1_action_x_ratio #self.ai1_action.x * game_width_ratio
        self.ai1_action.y = curr_game_height * self.ai1_action_y_ratio #self.ai1_action.y * game_height_ratio

    ##################################^^^ PYGLET CODE ^^^##########################################

    # def plot_visited_states_percents(self):
    #     fig = plt.figure()
    #     plt.plot(self.sim_sample_nums, self.visited_percents)
    #     plt.title("Percent of states visited vs. Number of Trials")
    #     plt.xlabel('Number of Trials')
    #     plt.ylabel('Percent of states visited')
    #     #plt.show()  # use this to display the graph.
    #     plt.savefig("pong-ai-visited-graph.png")
    #     plt.close(fig)

    def reset_turn(self, victor, loser):
        """Resets the turn to the loser once the ball goes past the loser's paddle."""
        self.ball = Ball(self)
        self.turn = loser
        victor.score += 1
        self.qagent.ball = self.ball

    def render_pong_canvas(self):
        """Renders the pong canvas in Pyglet."""
        pyglet.clock.tick()
        # print(f"FPS is {pyglet.clock.get_fps()}") # Use to view the FPS of the game.
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.update()
            window.flip()

    def get_img_array(self):
        """Returns the image array of the current Pong canvas."""
        img_buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        raw_img_data = img_buffer.get_image_data()
        img_data_format = 'L' #'RGB'
        pitch = raw_img_data.width * len(img_data_format)
        pixel_data = raw_img_data.get_data(img_data_format, pitch)
        img = Image.frombytes(img_data_format, (img_buffer.width, img_buffer.height), pixel_data)
        img_arr = np.asarray(img).tolist() # dimension: 166 x 233
        img_arr = img_arr[::-1] # flip the image vertically (along the x axis)
        # For debugging:
        #img = Image.fromarray(img_arr)
        #img.save('how_pong_ai_sees_the_game.png')
        #print(len(img_arr), len(img_arr[0]))

        # Less efficient way to get Pyglet image array:
        # pyglet.image.get_buffer_manager().get_color_buffer().save('pong-canvas-img.png')
        # img_arr = np.asarray(Image.open('pong-canvas-img.png').convert('L')).tolist()

        # Another alternative solution using np.frombuffer():
        # img_arr = np.frombuffer(raw_img_data.get_data(), dtype=np.uint8)
        # img_arr = img_arr.reshape(color_buffer.height, color_buffer.width, 4)
        # img_arr = img_arr[::-1, :, 3].tolist()
        # print(type(img_arr), len(img_arr), len(img_arr[0]), len(img_arr[1]))

        # For debugging:
        # b = False
        # num_non_zeros = 0
        # for i in range(len(img_arr)):
        #     for j in range(len(img_arr[0])):
        #         if img_arr[i][j] != 0:
        #             num_non_zeros += 1
        #             # print("Non 0 found!", img_arr[i][j])
        #             # print(str(i), str(j))
        #             b = True
        #             #break
        #     # if b:
        #     #     break
        # if not b:
        #     print("ALL 0s!")
        # else:
        #     print("Number of non-zeros:", num_non_zeros)

        return img_arr

    def play(self, winning_score=11, qlearn_mode=False, neuralnet_filename='pong-ai-deep-q-network.h5'):
        """
        Plays the Pong game and keep playing until one of the players reaches the winning score
        or the episode limit while Q-learning is reached.
        """
        self.winning_score = winning_score
        self.qlearn_mode = qlearn_mode
        #super().set_visible(False)
        if self.qlearn_mode:
            super().set_visible(False) # hide the Pyglet window during learning.
        else:
            super().set_visible(True)
        self.qagent = Qagent(self, self.ai1, winning_score)
        if os.path.isfile(neuralnet_filename):
            #self.qagent.load_target_model(neuralnet_filename)
            self.qagent.load_target_model_weights(neuralnet_filename)
        opponent_prob = 1 #0.85  # probability the opponent paddle hits the ball.
        # middle_state = self.qagent.num_paddle_states // 2
        self.qagent_action = 0 # default action is to idle.

        # Keep track of the percent of visited states as the number of sims increases
        # and later plot these percents in a graph.
        num_samples = 100  # max number of samples we want to sample.
        sample_mod_num = 0
        prev_sim_num = -1
        if winning_score > num_samples:
            sample_mod_num = winning_score / num_samples
        else:
            sample_mod_num = 1

        # if self.qlearn_mode:
        #     self.sim_sample_nums = []
        #     self.visited_percents = []

        # Stop the q-learning if a trial exceeds the given time limit (in minutes).
        #TRIAL_TIME_LIMIT = 5
        #TRIAL_TIME_LIMIT_SECS = TRIAL_TIME_LIMIT * 60
        #trial_start_time = time.time()

        # NOT NEEDED. But may be used in future to make Pyglet GUI more responsive.
        # pyglet.clock.schedule_interval(self.update, 1/120.0)
        # pyglet.app.run()
        #max_y_diff = (self.ai1.height) // 2 + (self.ball.height // 2)
        ball_vector_angle = (math.pi - 2*math.atan(9/6)) / 2 #33.69deg # in rads, angle is between 0 to 70 degrees
        # +/-90deg vector angle means the ball will only move in the y direction (move up and down).
        # color_buffer = pyglet.image.get_buffer_manager().get_color_buffer() #.save('pong-canvas-img.png')
        # raw_img_data = color_buffer.get_image_data()
        # img_data_format = 'RGB'
        # pitch = raw_img_data.width * 3
        self.ball.move_y = DIRECTION["UP"] # throw the ball from the bottom up
        self.ball.y = self.canvas_height - 150
        # Render the Pong env for a few frames before starting the game.
        # This ensures that the Q agent can see the Pong frames as soon as the game starts.
        for _ in range(5):
            self.render_pong_canvas()

        #i = 0
        reward = 0
        while True: #self.ai1.score < winning_score and self.ai2.score < winning_score:
            # Keep looping until the episode limit is reached.
            if self.qlearn_mode:
                if self.qagent.episode_count == self.qagent.episode_limit:
                    self.qagent.episode_count = 0
                    self.qagent.frame_count = 0
                    break
            else: # play mode
                if self.ai1.score >= winning_score or self.ai2.score >= winning_score:
                    self.qagent.episode_count = 0
                    self.qagent.frame_count = 0
                    break

            self.render_pong_canvas()
            pong_frame = self.get_img_array()

            if self.qlearn_mode:
                #trial_curr_time = time.time()
                #trial_elapsed_time = trial_curr_time - trial_start_time
                # Stop the q-learning if a trial exceeds the given time limit (in minutes).
                # if trial_elapsed_time >= TRIAL_TIME_LIMIT_SECS:
                #     print("Time limit reached!")
                #     break

                self.qagent_action = self.qagent.q_learn(pong_frame, reward)
                reward = 0 # reset reward.
                if self.ai2.score > prev_sim_num and self.ai2.score % sample_mod_num == 0:
                    # self.sim_sample_nums.append(self.ai2.score)
                    # self.visited_percents.append(self.qagent.get_percent_of_states_explored())
                    prev_sim_num = self.ai2.score
            else:
                self.qagent_action = self.qagent.play_game(pong_frame, reward)

            # On new serve (start of each turn), reset the paddles to their
            # center position and move the ball to the correct side.
            if self.turn:
                #print("your turn")
                if self.qlearn_mode:
                    # Initialize AI1's paddle to a random possible position:
                    #self.ai1.y = random.randint(0, self.canvas_height - self.ai1.height)
                    self.ai1.y = random.choice(self.qagent.possible_pad_states)
                else:
                    self.ai1.y = (self.canvas_height // 2) - (self.ai1.height // 2)
                self.ai2.y = (self.canvas_height // 2) - (self.ai2.height // 2)
                self.ball.move_x = DIRECTION["LEFT"] if self.turn == self.ai1 else DIRECTION["RIGHT"]
                if self.ball_init_direction == 1:
                    self.ball.move_y = DIRECTION["UP"] # throw the ball from the bottom up
                    self.ball.y = self.canvas_height - 150 # ball starts from the bottom
                    self.ball_init_direction = 2
                else:
                    self.ball.move_y = DIRECTION["DOWN"] # throw the ball from the top down
                    self.ball.y = 150 #154 # ball starts from the top
                    self.ball_init_direction = 1
                self.qagent_action = 0
                ball_vector_angle = (math.pi - 2*math.atan(9/6)) / 2
                self.turn = None

            # Handle ball movement.
            # Move ball in intended direction based on move_y and move_x values.
            # The ball travels faster in the x direction than in the y direction.
            bounce_angle = (math.pi - 2*ball_vector_angle) / 2
            ball_x_speed = int(self.ball.speed * math.sin(bounce_angle))
            ball_y_speed = int(self.ball.speed * math.cos(bounce_angle))
            #print(ball_x_speed, ball_y_speed)

            # For varying ball angle: (overall ball speed remains constant)
            if self.ball.move_x == DIRECTION["LEFT"]:
                self.ball.x -= ball_x_speed #self.ball.speed
            elif self.ball.move_x == DIRECTION["RIGHT"]:
                self.ball.x += ball_x_speed #self.ball.speed

            if self.ball.move_y == DIRECTION["UP"]:
                self.ball.y -= ball_y_speed #int(self.ball.speed / 1.5)
            elif self.ball.move_y == DIRECTION["DOWN"]:
                self.ball.y += ball_y_speed #int(self.ball.speed / 1.5)

            # If the ball collides with the top and bottom bound limits, bounce it.
            if self.ball.y <= 0:
                self.ball.y = 0
                self.ball.move_y = DIRECTION["DOWN"]
            elif self.ball.y >= self.canvas_height - self.ball.height:
                self.ball.y = self.canvas_height - self.ball.height
                self.ball.move_y = DIRECTION["UP"]

            # If the ball makes it past either of the paddles,
            # add a point to the winner and reset the turn to the loser.
            if self.ball.x <= 0:  # ai1 lost, ai2 won the round.
                self.render_pong_canvas()
                pong_frame = self.get_img_array()
                if self.qlearn_mode:
                    self.qagent.q_learn(pong_frame, reward)
                else:
                    self.qagent.play_game(pong_frame, reward)
                #print("AI2 scored a goal.")
                self.reset_turn(self.ai2, self.ai1)
                reward = -1
                trial_start_time = time.time()
                # Punish the AI every time it misses the ball.
                # if qlearn_mode and self.qagent.prev_state is not None:
                #     self.qagent.update_reward(-1)
            elif self.ball.x >= self.canvas_width - self.ball.width: # ai1 won, ai2 lost.
                self.render_pong_canvas()
                pong_frame = self.get_img_array()
                if self.qlearn_mode:
                    self.qagent.q_learn(pong_frame, reward)
                else:
                    self.qagent.play_game(pong_frame, reward)
                #print("AI1 scored a goal.")
                self.reset_turn(self.ai1, self.ai2)
                reward = 1

            # For varying ball speed:
            # if self.ball.move_x == DIRECTION["LEFT"]:
            #     self.ball.x -= self.ball.speed
            # elif self.ball.move_x == DIRECTION["RIGHT"]:
            #     self.ball.x += self.ball.speed

            # if self.ball.move_y == DIRECTION["UP"]:
            #     self.ball.y -= int(self.ball.speed / 1.5)
            # elif self.ball.move_y == DIRECTION["DOWN"]:
            #     self.ball.y += int(self.ball.speed / 1.5)

            # Handle ai1 UP and DOWN movement.
            self.ai1.y += self.qagent_action
            # ball_center = (self.ball.y + self.ball.y + self.ball.height) // 2
            # pad_center = (self.ai1.y + self.ai1.y + self.ai1.height) // 2
            # y_diff = abs(ball_center - pad_center)
            # if y_diff >= self.ai1.speed:
            #     if self.ai1.y + (self.ai1.height // 2) > self.ball.y:
            #         self.ai1.y -= self.ai1.speed
            #     elif self.ai1.y + (self.ai1.height // 2) < self.ball.y:
            #         self.ai1.y += self.ai1.speed

            # Handle ai2 UP and DOWN movement.
            # The ai2 paddle's y always follows the y position of the ball.
            ball_center = (self.ball.y + self.ball.y + self.ball.height) // 2
            pad_center = (self.ai2.y + self.ai2.y + self.ai2.height) // 2
            y_diff = abs(ball_center - pad_center)
            if y_diff >= self.ai2.speed:
                if self.ai2.y + (self.ai2.height // 2) > self.ball.y:
                    self.ai2.y -= self.ai2.speed
                elif self.ai2.y + (self.ai2.height // 2) < self.ball.y:
                    self.ai2.y += self.ai2.speed

            # Handle ai1 wall collision.
            if self.ai1.y <= 0:
                self.ai1.y = 0
            elif self.ai1.y >= self.canvas_height - self.ai1.height:
                self.ai1.y = self.canvas_height - self.ai1.height

            # Handle ai2 wall collision.
            if self.ai2.y <= 0:
                self.ai2.y = 0
            elif self.ai2.y >= self.canvas_height - self.ai2.height:
                self.ai2.y = self.canvas_height - self.ai2.height

            # Handle ai1 (q agent) ball collision.
            if self.ball.x <= self.ai1.x + self.ai1.width and \
                self.ball.x + self.ball.width >= self.ai1.x:
                if self.ball.y <= self.ai1.y + self.ai1.height and \
                    self.ball.y + self.ball.height >= self.ai1.y:
                    self.ball.x = self.ai1.x + self.ball.width
                    self.ball.move_x = DIRECTION["RIGHT"]
                    ball_center = self.ball.y + (self.ball.height // 2)
                    pad_center = self.ai1.y + (self.ai1.height // 2)
                    if ball_center < pad_center:
                        self.ball.move_y = DIRECTION["UP"]
                    elif ball_center > pad_center:
                        self.ball.move_y = DIRECTION["DOWN"]
                    angle_factor = abs(ball_center - pad_center) / (self.ball.height / 2 + self.ai1.height / 2) # 0 - 1
                    ball_vector_angle = 70 * math.pi / 180 * angle_factor

            # Handle ai2 ball collision.
            if self.ball.x <= self.ai2.x + self.ai2.width and \
                self.ball.x + self.ball.width >= self.ai2.x:
                if self.ball.y <= self.ai2.y + self.ai2.height and \
                    self.ball.y + self.ball.height >= self.ai2.y:
                    # Q agent learns or plays the game.
                    #if self.qlearn_mode:
                    self.ball.x = self.ai2.x - self.ball.width
                    self.ball.move_x = DIRECTION["LEFT"]
                    ball_center = self.ball.y + (self.ball.height // 2)
                    pad_center = self.ai2.y + (self.ai2.height // 2)
                    if ball_center < pad_center:
                        self.ball.move_y = DIRECTION["UP"]
                    elif ball_center > pad_center:
                        self.ball.move_y = DIRECTION["DOWN"]
                    # angle_factor = abs(ball_center - pad_center) / (self.ball.height / 2 + self.ai2.height / 2) # 0 - 1
                    angle_factor = random.random()
                    ball_vector_angle = 70 * math.pi / 180 * angle_factor

            #i += 1

        if qlearn_mode:
            #self.qagent.save_target_model(neuralnet_filename) # UNCOMMENT! save the Q network in a file.
            self.qagent.save_target_model_weights(neuralnet_filename) # UNCOMMENT!
            self.qagent.plot_loss('pong-ai-deep-q-loss.png')
            print("Q learning finished!")
        else:
            print("Pong game finished!")
        if self.ai1.score == winning_score:
            print("AI1 is the winner!")
        elif self.ai2.score == winning_score:
            print("AI2 is the winner!")
        print("AI1 score (Q agent): " + str(self.ai1.score))
        print(f"AI2 score ({int(opponent_prob*100)}% perfect agent): " + str(self.ai2.score))
        self.ai1.score = 0
        self.ai2.score = 0
        self.turn = self.ai2
        print()


class Qagent():
    """The Q agent playing Pong using deep Q networks."""

    def __init__(self, pong_game, paddle, episode_limit):
        self.pong_game = pong_game
        self.paddle = paddle
        # Hyperparameters:
        self.gamma = 0.99 # discount factor for future rewards.
        self.epsilon = 1.0 # randomness factor. e=0 makes the agent greedy.
        self.epsilon_min = 0.1 # minimum epsilon greedy parameter.
        self.batch_size = 32
        self.max_steps_per_episode = 20000
        self.update_every_kth_frame = 4 # Q agent updates its action every kth frame.
        # Input image dimensions:
        self.state_width = 233
        self.state_height = 166
        self.state_depth = self.update_every_kth_frame
        # Experience replay buffer:
        self.max_buffer_len = 25000 #100000
        self.curr_buffer_len = 0
        self.frame_history = [] # has a max size of k.
        self.state_history = deque([])
        self.action_history = deque([])
        self.rewards_history = deque([])
        self.next_state_history = deque([])
        # self.state = np.asarray(
        #     [
        #         [
        #             [0 for _ in range(self.state_width)]
        #             for _ in range(self.state_height)
        #         ]
        #         for _ in range(self.state_depth)
        #     ]
        # , dtype=np.int32) # dimensions: 4x166x233
        self.state = np.asarray(
            [
                [
                    [0 for _ in range(self.state_width)]
                    for _ in range(self.state_height)
                ]
                for _ in range(self.state_depth)
            ]
        , dtype=np.int32) # dimensions: 4x166x233
        self.state = tf.convert_to_tensor(self.state, dtype=tf.int32)
        self.action = 0
        # Q models:
        self.model = self.create_q_model() # makes the predictions for Q-values which are used to make an action.
        # The target model is used for the prediction of future rewards.
        # The weights of the target model get updated every self.update_target_model steps.
        # This is so that when the loss between the Q-values is calculated, the target Q-value is stable.
        self.target_model = self.create_q_model()
        self.loss_func = keras.losses.CategoricalCrossentropy()
        # Alternative loss function: Huber loss
        # Note: Huber loss is generally used for regression, not classification.
        #self.loss_func = keras.losses.Huber() # using huber loss for stability.
        self.update_target_model = 10000 # how often to update the target network.
        self.alpha = 0.01 #0.00025 # learning rate.
        self.optimizer = keras.optimizers.Adam(learning_rate=self.alpha, clipnorm=1.0)
        # Counts and limits:
        self.num_actions = 3 # idle (0), up (1), down (2)
        self.frame_count = 0 # number of frames seen in our current episode.
        self.episode_count = 0
        self.episode_limit = episode_limit #1000000 # number of episodes to train the Q agent for.
        self.pad_actions = {0: 0, 1: -self.paddle.speed, 2: self.paddle.speed}

        self.possible_pad_states = []
        for y in range(0, self.pong_game.canvas_height-self.paddle.height+1, self.paddle.speed):
            self.possible_pad_states.append(y)
        for y in range(self.pong_game.canvas_height-self.paddle.height, -1, -self.paddle.speed):
            self.possible_pad_states.append(y)
        starting_pos = (self.pong_game.canvas_height // 2) - (self.paddle.height // 2)
        for y in range(starting_pos, self.pong_game.canvas_height-self.paddle.height+1, self.paddle.speed):
            self.possible_pad_states.append(y)
        for y in range(starting_pos, -1, -self.paddle.speed):
            self.possible_pad_states.append(y)
        self.possible_pad_states.sort()
        #print(self.possible_pad_states)

        self.loss = 0
        self.losses = []
        self.episode_counts = []

    def create_q_model(self):
        """Create Deep Q network."""
        # Input layer: input image is 4x166x233 pixels.
        inputs = layers.Input(shape=(self.state_depth, self.state_height, self.state_width))
        # Hidden layers: (# of filters, kernal size, stride, activation). filter size > stride
        layer1 = layers.MaxPool2D(pool_size=(4,1))(inputs)
        layer2 = layers.Reshape((166,233,1))(layer1)
        layer3 = layers.Conv2D(16, 8, strides=(4,4), activation="relu")(layer2) #(inputs)
        layer4 = layers.Conv2D(32, 4, strides=(2,2), activation="relu")(layer3)
        layer5 = layers.Flatten()(layer4) # works better with square images.
        layer6 = layers.Dense(128, activation="relu")(layer5)

        # Output layer:
        num_pad_actions = 3 # self.num_actions is not included for some reason.
        action_layer = layers.Dense(num_pad_actions, activation="softmax")(layer6)

        model = keras.Model(inputs=inputs, outputs=action_layer)
        # For debugging: COMMENT OUT
        # for layer in model.layers:
        #     print(layer.output_shape)

        return model

    def preprocess_frames(self):
        """Preprocess the last k frames of a history and stack them together."""
        # processed_frames = tf.convert_to_tensor(np.asarray(self.frame_history))
        # nframes = len(self.frame_history)
        # if nframes > self.update_every_kth_frame: # For debuggging. COMMENT OUT.
        #     print(f"{nframes} frames in frame history!")
        # state_tensor = tf.convert_to_tensor(self.frame_history[:self.update_every_kth_frame])

        state_tensor = tf.convert_to_tensor(np.asarray(self.frame_history, dtype=np.int32), dtype=tf.int32)
        self.frame_history.clear()
        return state_tensor

        # k_frames = np.asarray(self.frame_history, dtype=np.int32)
        # k_frames = np.stack(k_frames, axis=-1)
        # #print("k_frames:", k_frames.shape) k_frames.shape: (166, 233, 4)
        # state_tensor = tf.convert_to_tensor(k_frames, dtype=tf.int32)
        # self.frame_history.clear()
        # return state_tensor

    def update_replay_buffer(self, next_state, reward):
        """Add state transition to the replay buffer."""
        self.state_history.append(self.state)
        self.action_history.append(self.action)
        self.next_state_history.append(next_state)
        self.rewards_history.append(reward)
        self.curr_buffer_len += 1

        if self.curr_buffer_len > self.max_buffer_len: # limit the size of the history.
            self.remove_first_history()

    def remove_first_history(self):
        """Remove the first entry in the reply buffer."""
        self.state_history.popleft()
        self.action_history.popleft()
        self.rewards_history.popleft()
        self.next_state_history.popleft()
        self.curr_buffer_len -= 1

    def train_q_network(self):
        # Update action every kth frame and once the replay buffer's length is greater than the batch size.
        # Get indices of samples for replay buffers.
        indices = np.random.choice(range(self.curr_buffer_len), size=self.batch_size)

        # Sample from the replay buffer.
        state_sample = tf.convert_to_tensor(np.asarray([self.state_history[i] for i in indices], dtype=np.int32), dtype=tf.int32)
        #state_sample = [self.state_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]
        rewards_sample = [self.rewards_history[i] for i in indices]
        next_state_sample = tf.convert_to_tensor(np.asarray([self.next_state_history[i] for i in indices], dtype=np.int32), dtype=tf.int32)
        #next_state_sample = [self.next_state_history[i] for i in indices]

        # Build the updated Q-values for the sampled future states.
        # Use the target model for stability.
        future_q_vals = self.target_model.predict(next_state_sample, batch_size=self.batch_size)
        #print("future_q_vals.shape:", future_q_vals.shape) # dim: 32x3
        #future_q_vals = self.target_model.predict_on_batch(next_state_sample)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + self.gamma * tf.math.reduce_max(future_q_vals, axis=1) # this is correct
        #print("updated_q_values.shape:", updated_q_values.shape) # dim: 32

        # Create a mask so we only calculate loss on the updated Q-values.
        masks = tf.one_hot(action_sample, self.num_actions)
        #print("masks.shape:", masks.shape) # dim: 32x3

        # Train the model on the states and updated Q-values.
        with tf.GradientTape() as tape:
            q_values = self.model(state_sample)
            #print("q_values.shape:", q_values.shape) # dim: 32x3
            # Apply the masks to the Q-values to get the Q-value for the action taken.
            q_values = tf.math.reduce_sum(tf.math.multiply(q_values, masks), axis=1)
            #print("q_values.shape:", q_values.shape) # dim: 32
            # Calculate the loss between the new and old Q-values.
            self.loss = self.loss_func(updated_q_values, q_values)
            #self.loss = loss #round(np.mean(loss), 3)

            # Backpropagation
            grads = tape.gradient(self.loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.frame_count % self.update_target_model == 0:
            # update the the target network with new weights
            self.target_model.set_weights(self.model.get_weights())

    def q_learn(self, pong_frame, reward):
        """Make the Q agent learn about its environment."""
        last_frame = False # indicates whether the current frame is the last frame of the episode.
        if self.frame_count > self.max_steps_per_episode: # Episode finished.
            self.pong_game.reset_turn(self.pong_game.ai2, self.pong_game.ai1)
            #self.episode_count += 1
            #print(f"Episode count: {self.episode_count} (Max steps exceeded)")
            #self.frame_count = 0
            last_frame = True
        elif abs(reward) == 1: # Pong episode finished.
            #self.episode_count += 1
            #print(f"Episode count: {self.episode_count}")
            #self.frame_count = 0
            last_frame = True

        self.frame_history.append(pong_frame)
        self.frame_count += 1

        if last_frame:
            # Ensure that there are k frames in the history:
            while self.frame_count % self.update_every_kth_frame != 0:
                self.frame_history.append(pong_frame)
                self.frame_count += 1

        if self.frame_count % self.update_every_kth_frame == 0:
            next_state = self.preprocess_frames()
            #next_state_tensor = tf.convert_to_tensor(self.state)
            self.update_replay_buffer(next_state, reward)
            self.state = next_state
            #self.state = next_state_tensor
            # Epsilon-greedy exploration.
            # Epsilon decay - decay probability of taking random action.
            self.epsilon = 1 - round(self.episode_count / self.episode_limit, 3)
            self.epsilon = max(self.epsilon, self.epsilon_min)
            rand_num = round(random.random(), 2)
            if rand_num < self.epsilon or self.state is None: # exploration
                self.action = np.random.choice(self.num_actions) # take a random action.
            else:                       # exploitation
                #state_tensor = tf.convert_to_tensor(self.state)
                expanded_state = tf.expand_dims(self.state, 0) # add extra dimension to the first axis.
                # Determine probabilities of actions given the environment state.
                action_probs = self.model(expanded_state, training=False)
                # Take the best known action.
                self.action = tf.argmax(action_probs[0]).numpy()

            if self.curr_buffer_len > self.batch_size:
                self.train_q_network()

            #self.state = next_state
            #self.action = action

        if last_frame:
            self.episode_counts.append(self.episode_count)
            avg_loss = round(np.mean(self.loss), 3)
            print(f"loss: {avg_loss}")
            self.losses.append(avg_loss)
            self.episode_count += 1
            self.frame_count = 0
            print(f"Episode count: {self.episode_count}")

        return self.pad_actions[self.action]

    def play_game(self, pong_frame, reward):
        """
        Make the Q agent play the Pong game after having
        learned all the weights of the Deep Q network.
        """
        self.frame_history.append(pong_frame)
        self.frame_count = (self.frame_count + 1) % self.update_every_kth_frame
        if abs(reward) == 1: # Pong episode finished.
             while self.frame_count % self.update_every_kth_frame != 0:
                self.frame_history.append(pong_frame)
                self.frame_count += 1

        if self.frame_count % self.update_every_kth_frame == 0:
            # Update the state:
            self.state = self.preprocess_frames()
            #state_tensor = tf.convert_to_tensor(self.state)
            expanded_state = tf.expand_dims(self.state, 0) # add extra dimension to the first axis.
            # Determine probabilities of actions given the environment state.
            action_probs = self.target_model(expanded_state, training=False)
            # Take the best action.
            self.action = tf.argmax(action_probs[0]).numpy()

        return self.pad_actions[self.action]

    def simple_moving_avg(a, window=3):
        """
        Calculates the simple moving average of a list.
        Useful for smoothening short-term variations in graphs.
        """
        res = a.copy()
        a = np.pad(a, (window//2, window//2 + 1), 'edge')
        su = sum(a[:window])
        for i in range(len(res)):
            res[i] = su / window
            su = su - a[i] + a[i + window]

        return res

    def plot_loss(self, filename):
        """Plot the loss of the Q network and save the plot as a file."""
        fig = plt.figure()
        sma_losses = self.simple_moving_avg(self.losses, 25)
        plt.plot(self.episode_counts, sma_losses) #self.losses)
        plt.title("Loss vs. Episode Count")
        plt.xlabel('Episode Count')
        plt.ylabel('Loss')
        #plt.show()  # use this to display the graph.
        plt.savefig(filename) # Ex: "pong-ai-loss-graph.png"
        plt.close(fig)

    def save_target_model(self, filename): # Eg. filename: 'pong-ai-deep-q-network.h5'
        """Save the Deep Q network to a HDF5 file after finishing with the training."""
        self.target_model.save(filename)

    def load_target_model(self, filename): # Eg. filename: 'pong-ai-deep-q-network.h5'
        """Load a previously saved Deep Q network from a HDF5 file, if it exists."""
        self.target_model = load_model(filename, compile=False)

    def save_target_model_weights(self, filename):
        """Save the weights of the Deep Q network to a HDF5 file after finishing with the training."""
        self.target_model.save_weights(filename)

    def load_target_model_weights(self, filename):
        """Load the weights of a previously saved Deep Q network from a HDF5 file, if it exists."""
        self.target_model.load_weights(filename)


def main():
    num_episodes = 100
    network_filename = 'pong-ai-deep-q-network.h5'
    if os.path.isfile(network_filename):
        print("Game started.")
        pong_game.play(neuralnet_filename=network_filename)
        print("Game finished.")
    else:
        print("Q learning started.")
        start_time = time.time()
        pong_game.play(winning_score=num_episodes, qlearn_mode=True,
            neuralnet_filename=network_filename)
        end_time = time.time()
        total_time_elapsed = end_time - start_time
        print(f"Total time elapsed for {num_episodes} episodes: %.2fs" % (total_time_elapsed))
        avg_time_per_simulation = round(total_time_elapsed / num_episodes, 7)
        print(f"Avg. time per episode: {avg_time_per_simulation}s")

        print("Game started.")
        pong_game.play(neuralnet_filename=network_filename)
        print("Game finished.")

        #pong_game.plot_visited_states_percents()

if __name__ == '__main__':
    pong_game = PongGame()
    # If the Q table already exists, then load the table and make the Q agent play the game.
    # Else train the Q agent by playing n games.
    # memory_limit() # limit the amount of memory used.
    # try:
    #     main()
    # except MemoryError:
    #     sys.stderr.write('\n\nERROR: Memory Exception\n')
    #     sys.exit(1)
    main()
