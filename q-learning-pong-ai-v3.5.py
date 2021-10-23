# A Q learning Pong AI agent built using Python.

import pyglet
from pyglet import shapes
from pyglet import text

import math
import random
import time
#import multiprocessing
import matplotlib.pyplot as plt
import os
import pickle

# Global variables
DIRECTION = {"IDLE": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
# winning_score = 11


class Ball():
    """The ball object (The cube that bounces back and forth)."""

    def __init__(self, pong_game):
        self.width = 18 #1000 // 10 # for debugging, need to change back to 18.
        self.height = 18 #1000 // 10 # for debugging, need to change back to 18.
        self.x = pong_game.canvas_width // 2
        self.y = pong_game.canvas_height // 2
        self.move_x = DIRECTION["IDLE"]
        self.move_y = DIRECTION["IDLE"]
        self.speed = 9


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
        #self.ai1.height = self.canvas_height // 3 # for debugging, need to remove.
        #self.ai1.speed = 18 # comment out after debugging.
        self.ai2 = Paddle(self, 'right')
        self.ball = Ball(self)
        self.winning_score = 11
        self.qagent = Qagent(self, self.ai1, self.ai2, self.ball)

        # self.ai2.speed = 8  # make ai2's paddle speed slower than ai1.
        self.turn = self.ai2  # it's the ai2's turn first.
        self.qlearn_mode = False
        self.sim_sample_nums = []  # x values of visited percents graph.
        self.visited_percents = [] # y values of visited percents graph.
        self.qagent_action = 0
        self.ball_init_direction = 1 # 1 means UP, 2 means DOWN

        ########################vvv PYGLET CODE vvv################################################

        super().__init__(width=self.canvas_width//2, height=self.canvas_height//2, caption='Q-learning Pong',
            resizable=True)
        # create the paddles and the ball.
        self.paddle_colors = (255,255,255) # paddle color is white
        # self.ai1_rect = shapes.Rectangle(self.ai1.x, self.ai1.y, self.ai1.width, self.ai1.height,
        #     color=self.paddle_colors)
        # #self.ai1_rect.opacity = 255
        # self.ai2_rect = shapes.Rectangle(self.ai2.x, self.ai2.y, self.ai2.width, self.ai2.height,
        #     color=self.paddle_colors)
        # self.ball_rect = shapes.Rectangle(self.ball.x, self.ball.y, self.ball.width, self.ball.height,
        #     color=self.paddle_colors)
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
        self.clear()

        self.ai1_rect.draw()
        self.ai2_rect.draw()
        self.ball_rect.draw()
        self.line.draw()

        self.ai1_scoreboard.draw()
        self.ai2_scoreboard.draw()
        self.ai1_action.draw()

    def update(self):
        curr_game_width, curr_game_height = self.get_size()

        # self.ai1_rect.x = self.ai1.x
        # #self.ai1_rect.y = self.ai1.y
        # self.ai1_rect.y = self.canvas_height-self.ai1.height-self.ai1.y

        # self.ai2_rect.x = self.ai2.x
        # #self.ai2_rect.y = self.ai2.y
        # self.ai2_rect.y = self.canvas_height-self.ai2.height-self.ai2.y

        # self.ball_rect.x = self.ball.x
        # #self.ball_rect.y = self.ball.y
        # self.ball_rect.y = self.canvas_height-self.ball.height-self.ball.y
        # self.ai1_scoreboard.text = "AI1: " + str(self.ai1.score)
        # self.ai2_scoreboard.text = "AI2: " + str(self.ai2.score)
        # qaction = "IDLE 0"
        # if self.qagent_action < 0:
        #     qaction = "UP  1"
        # elif self.qagent_action > 0:
        #     qaction = "DOWN   2"
        # self.ai1_action.text = "Action: " + qaction

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

        self.line.x = curr_game_width / 2 #int(self.line.x * game_width_ratio)
        self.line.x2 = self.line.x
        #self.line.y = self.line.y * game_height_ratio
        self.line.y2 = curr_game_height #int(self.line.y2 * game_height_ratio)

        self.ai1_scoreboard.text = "AI1: " + str(self.ai1.score)
        self.ai1_scoreboard.font_size = curr_game_height * self.font_ratio #prev_font_size * game_height_ratio
        self.ai1_scoreboard.x = curr_game_width * self.ai1_scoreboard_x_ratio #self.ai1_scoreboard.x * game_width_ratio
        self.ai1_scoreboard.y = curr_game_height * self.ai_scoreboard_y_ratio #self.ai1_scoreboard.y * game_height_ratio

        self.ai2_scoreboard.text = "AI2: " + str(self.ai2.score)
        self.ai2_scoreboard.font_size = self.ai1_scoreboard.font_size
        self.ai2_scoreboard.x = curr_game_width * self.ai2_scoreboard_x_ratio #self.ai2_scoreboard.x * game_width_ratio
        self.ai2_scoreboard.y = curr_game_height * self.ai_scoreboard_y_ratio #self.ai2_scoreboard.y * game_height_ratio

        qaction = "IDLE 0"
        if self.qagent_action < 0:
            qaction = "UP  1"
        elif self.qagent_action > 0:
            qaction = "DOWN   2"
        self.ai1_action.text = "Action: " + qaction
        self.ai1_action.font_size = self.ai1_scoreboard.font_size
        self.ai1_action.x = curr_game_width * self.ai1_action_x_ratio #self.ai1_action.x * game_width_ratio
        self.ai1_action.y = curr_game_height * self.ai1_action_y_ratio #self.ai1_action.y * game_height_ratio

    ##################################^^^ PYGLET CODE ^^^##########################################

    def plot_visited_states_percents(self):
        fig = plt.figure()
        plt.plot(self.sim_sample_nums, self.visited_percents)
        plt.title("Percent of states visited vs. Number of Trials")
        plt.xlabel('Number of Trials')
        plt.ylabel('Percent of states visited')
        #plt.show()  # use this to display the graph.
        plt.savefig("pong-ai-visited-graph.png")
        plt.close(fig)

    def reset_turn(self, victor, loser):
        """Reset the turn to the loser once the ball goes past the loser's paddle."""
        self.ball = Ball(self)
        self.turn = loser
        victor.score += 1
        self.qagent.ball = self.ball

    def play(self, winning_score=11, qlearn_mode=False):
        """
        Play the Pong game and keep playing until one of the players reaches the winning score.
        """
        self.winning_score = winning_score
        self.qlearn_mode = qlearn_mode
        opponent_prob = 0.85  # probability the opponent paddle hits the ball.
        # middle_state = self.qagent.num_paddle_states // 2
        self.qagent_action = 0

        # Keep track of the percent of visited states as the number of sims increases
        # and later plot these percents in a graph.
        num_samples = 100  # max number of samples we want to sample.
        sample_mod_num = 0
        prev_sim_num = -1
        if winning_score > num_samples:
            sample_mod_num = winning_score / num_samples
        else:
            sample_mod_num = 1

        if self.qlearn_mode:
            self.sim_sample_nums = []
            self.visited_percents = []

        # Stop the q-learning if a trial exceeds the given time limit (in minutes).
        TRIAL_TIME_LIMIT = 5
        TRIAL_TIME_LIMIT_SECS = TRIAL_TIME_LIMIT * 60
        trial_start_time = time.time()

        # NOT NEEDED. But may be used in future to make Pyglet GUI more responsive.
        # pyglet.clock.schedule_interval(self.update, 1/120.0)
        # pyglet.app.run()

        while self.ai1.score < winning_score and self.ai2.score < winning_score:
            if self.qlearn_mode:
                trial_curr_time = time.time()
                trial_elapsed_time = trial_curr_time - trial_start_time
                # Stop the q-learning if a trial exceeds the given time limit (in minutes).
                if trial_elapsed_time >= TRIAL_TIME_LIMIT_SECS:
                    break

                self.qagent_action = self.qagent.qlearn(self.ball.y, self.ai1.y)
                if self.ai2.score > prev_sim_num and self.ai2.score % sample_mod_num == 0:
                    self.sim_sample_nums.append(self.ai2.score)
                    self.visited_percents.append(self.qagent.get_percent_of_states_explored())
                    prev_sim_num = self.ai2.score
            else:
                self.qagent_action = self.qagent.play_game(self.ball.y, self.ai1.y)

            # PYGLET CODE: (Uncomment to display Pong GUI)
            if not self.qlearn_mode:
                pyglet.clock.tick()

                for window in pyglet.app.windows:
                    window.switch_to()
                    window.dispatch_events()
                    window.dispatch_event('on_draw')
                    window.update()
                    window.flip()

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
                # Switch the initial direction of the ball
                if self.ball_init_direction == 1:
                    self.ball.move_y = DIRECTION["UP"] # throw the ball from the bottom up
                    self.ball.y = self.canvas_height - 150 # ball starts from the bottom
                    self.ball_init_direction = 2
                else:
                    self.ball.move_y = DIRECTION["DOWN"] # throw the ball from the top down
                    self.ball.y = 150 #154 # ball starts from the top
                    self.ball_init_direction = 1
                self.qagent_action = 0
                self.turn = None

            # If the ball makes it past either of the paddles,
            # add a point to the winner and reset the turn to the loser.
            if self.ball.x <= 0:  # ai1 lost, ai2 won the round.
                self.reset_turn(self.ai2, self.ai1)
                trial_start_time = time.time()
                # Punish the AI every time it misses the ball.
                # if qlearn_mode and self.qagent.prev_state is not None:
                #     self.qagent.update_reward(-1)
            elif self.ball.x >= self.canvas_width - self.ball.width: # ai1 won, ai2 lost.
                #print("AI1 scored a goal.")
                self.reset_turn(self.ai1, self.ai2)

            # # If the ball collides with the top and bottom bound limits, bounce it.
            # if self.ball.y <= 0:
            #     self.ball.y = 0
            #     self.ball.move_y = DIRECTION["DOWN"]
            # elif self.ball.y >= self.canvas_height - self.ball.height:
            #     self.ball.y = self.canvas_height - self.ball.height
            #     self.ball.move_y = DIRECTION["UP"]

            # # Handle ai1 wall collision.
            # if self.ai1.y <= 0:
            #     self.ai1.y = 0
            # elif self.ai1.y >= self.canvas_height - self.ai1.height:
            #     self.ai1.y = self.canvas_height - self.ai1.height

            # # Handle ai2 wall collision.
            # if self.ai2.y <= 0:
            #     self.ai2.y = 0
            # elif self.ai2.y >= self.canvas_height - self.ai2.height:
            #     self.ai2.y = self.canvas_height - self.ai2.height

            # Handle ball movement.
            # Move ball in intended direction based on move_y and move_x values.
            # The ball travels faster in the x direction than in the y direction.
            if self.ball.move_y == DIRECTION["UP"]:
                self.ball.y -= int(self.ball.speed / 1.5)
            elif self.ball.move_y == DIRECTION["DOWN"]:
                self.ball.y += int(self.ball.speed / 1.5)

            if self.ball.move_x == DIRECTION["LEFT"]:
                self.ball.x -= self.ball.speed
            elif self.ball.move_x == DIRECTION["RIGHT"]:
                self.ball.x += self.ball.speed

            # If the ball collides with the top and bottom bound limits, bounce it.
            if self.ball.y <= 0:
                self.ball.y = 0
                self.ball.move_y = DIRECTION["DOWN"]
            elif self.ball.y >= self.canvas_height - self.ball.height:
                self.ball.y = self.canvas_height - self.ball.height
                self.ball.move_y = DIRECTION["UP"]

            # Handle ai1 UP and DOWN movement.
            self.ai1.y += self.qagent_action

            # Handle ai2 UP and DOWN movement.
            # The ai2 paddle's y always follows the y position of the ball.
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
                    # Reward the Q agent every time it hits the ball.
                    # if qlearn_mode and self.qagent.prev_state is not None:
                    #     self.qagent.update_reward(1)
                    # Move the Q agent's paddle back to the center.
                    # self.qagent_action = middle_state

            # Handle ai2 ball collision.
            if self.ball.x <= self.ai2.x + self.ai2.width and \
                self.ball.x + self.ball.width >= self.ai2.x:
                if self.ball.y <= self.ai2.y + self.ai2.height and \
                    self.ball.y + self.ball.height >= self.ai2.y:
                    # Q agent learns or plays the game.
                    if self.qlearn_mode:
                        self.ball.x = self.ai2.x - self.ball.width
                        self.ball.move_x = DIRECTION["LEFT"]
                        #print("Q learning method called.")
                        #self.qagent_action = self.qagent.qlearn()
                    else: # In actual gameplay, the opponent hits the ball with probability p.
                        rand_num = round(random.random(), 1)
                        if rand_num <= opponent_prob:
                            self.ball.x = self.ai2.x - self.ball.width
                            self.ball.move_x = DIRECTION["LEFT"]
                            #self.qagent_action = self.qagent.play_game()
                        else: # misses ball
                            self.ball.x += self.ball.width * 2 + 1

        if qlearn_mode:
            self.qagent.write_q_table('pong-qtable.dat')  # Save the Q table in a file.
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
        #self.qagent.glimpse_qtable()


class Qagent():
    """The Q agent playing the Pong game."""

    def __init__(self, pong_game, paddle, opponent, ball):
        self.pong_game = pong_game
        self.paddle = paddle
        self.opponent = opponent
        self.ball = ball
        self.alpha = 0.1  # learning rate.
        self.gamma = 0.8 # discount factor. # Before: 0.8
        self.epsilon = 1  # randomness factor. e=0 makes the agent greedy.
        self.num_y_directions = 2
        self.num_paddle_states = math.ceil(pong_game.canvas_height / self.paddle.height)  # 15
        self.num_ball_states = math.ceil(pong_game.canvas_height / self.ball.height)      # 56
        self.rewards = [
            [0 for _ in range(self.pong_game.canvas_height) # y position of paddle.
            ] for _ in range(self.pong_game.canvas_height)  # y position of ball.
        ]
        self.ball_actions = {0: 0, 1: -self.ball.speed, 2: self.ball.speed}
        self.pad_actions = {0: 0, 1: -self.paddle.speed, 2: self.paddle.speed}
        self.min_visits = 3 # The minimum number of times every state in the env should be visited.

        # Initialize positive rewards in the reward table.
        # Reward the agent whenever the ball's center is in the range of its paddle.
        # for ball_y in range(len(self.rewards)):
        #     for pad_y in range(len(self.rewards[0])):
        #         ball_center = (ball_y + self.ball.height) // 2
        #         pad_right = pad_y + self.paddle.height
        #         if ball_center <= pad_right and ball_center >= pad_y:
        #             self.rewards[ball_y][pad_y] = 1

        # Reward the agent more when the ball's center is aligned with the paddle's center.
        # (This reward system seems to make Q agent 20% faster at learning for large # of trails.)
        # for ball_y in range(len(self.rewards)):
        #     for pad_y in range(len(self.rewards[0])):
        #         ball_center = (ball_y + self.ball.height) // 2
        #         pad_center = (pad_y + self.paddle.height) // 2
        #         abs_y_diff = abs(ball_center - pad_center)
        #         reward = 1 / (abs_y_diff + 1)
        #         self.rewards[ball_y][pad_y] = reward

        # Reward the agent more when the ball's center is aligned with the paddle's center.
        # But also punish the agent more the further the paddle is from the ball.
        for ball_y in range(len(self.rewards)):
            for pad_y in range(len(self.rewards[0])):
                ball_right = ball_y + self.ball.height
                ball_center = (ball_y + ball_right) // 2
                pad_right = pad_y + self.paddle.height
                pad_center = (pad_y + pad_right) // 2
                abs_y_diff = abs(ball_center - pad_center)
                reward = 1 / (abs_y_diff + 1)
                # if ball_center <= pad_right and ball_center >= pad_y:
                #     reward = 1 / (abs_y_diff + 1)
                #     reward *= 100
                # else:
                #     reward = -abs_y_diff / 10
                self.rewards[ball_y][pad_y] = reward

        self.qtable = None
        if os.path.isfile('pong-qtable.dat'):
            self.read_q_table('pong-qtable.dat')
        else:
            self.qtable = [
                [[0 for _ in range(3) # since 3 possible actions: up, down, idle.
                    ] for _ in range(self.pong_game.canvas_height)     # y position of paddle.
                ] for _ in range(self.pong_game.canvas_height)         # y position of ball.
            ]
        self.visited_states = [ # used to keep track of previously visited states.
            [0 for _ in range(self.pong_game.canvas_height) # y position of paddle.
            ] for _ in range(self.pong_game.canvas_height)  # y position of ball.
        ]

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

    def r(self, s, a):
        """
        A reward function R(s,a) that gives the agent a reward for taking
        action a in state s.
        """
        ball_y, pad_y = s
        new_ball_y = ball_y + self.ball_actions[self.ball.move_y]
        new_pad_y = pad_y + self.pad_actions[a]

        # Make sure the new states are within the bounds of the Pong env.
        if new_ball_y < 0:
            new_ball_y = 0
        elif new_ball_y > self.pong_game.canvas_height - self.ball.height:
            new_ball_y = self.pong_game.canvas_height - self.ball.height
        if new_pad_y < 0:
            new_pad_y = 0
        elif new_pad_y > self.pong_game.canvas_height - self.paddle.height:
            new_pad_y = self.pong_game.canvas_height - self.paddle.height

        return self.rewards[new_ball_y][new_pad_y]

        # Reward the Q agent whenever the y distance between the paddle and ball is reduced
        # and punish it otherwise.
        # ball_center = (ball_y + self.ball.height) // 2
        # pad_center = (pad_y + self.paddle.height) // 2
        # ball_pad_diff = abs(ball_center - pad_center)
        # new_ball_center = (new_ball_y + self.ball.height) // 2
        # new_pad_center = (new_pad_y + self.paddle.height) // 2
        # new_ball_pad_diff = abs(new_ball_center - new_pad_center)

        # range_reward = 0
        # new_ball_right_y = new_ball_y + self.ball.height
        # new_pad_right_y = new_pad_y + self.paddle.height
        # if (new_ball_y >= new_pad_y and new_ball_y <= new_pad_right_y) or \
        #     (new_ball_right_y <= new_pad_right_y and new_ball_right_y >= new_pad_y):
        #     range_reward = 1000

        # diff_reward = 0
        # if new_ball_pad_diff < ball_pad_diff:
        #     diff_reward = 1
        # elif new_ball_pad_diff > ball_pad_diff:
        #     diff_reward = -1 #-1

        # return diff_reward + range_reward

    def exploration_fn(self, ball_y, pad_y):
        """
            Returns the best action to take in order to explore unseen states in the environment
            while reducing the probability of exploring bad states.
        """
        new_ball_y = ball_y + self.ball_actions[self.ball.move_y]
        if new_ball_y < 0:
            new_ball_y = 0
        elif new_ball_y > self.pong_game.canvas_height - self.ball.height:
            new_ball_y = self.pong_game.canvas_height - self.ball.height
        possible_next_states = []
        for pad_action, pad_y_change in self.pad_actions.items():
            new_pad_y = pad_y + pad_y_change
            if new_pad_y < 0:
                new_pad_y = 0
            elif new_pad_y > self.pong_game.canvas_height - self.paddle.height:
                new_pad_y = self.pong_game.canvas_height - self.paddle.height
            possible_next_states.append((new_ball_y, new_pad_y, pad_action))

        # Stochastic exploration:
        moves = [0,1,2] # idle, up, down
        move_weights = [0,0,0]
        max_q = max(self.qtable[new_ball_y][new_pad_y])
        for (new_ball_y, new_pad_y, pad_action) in possible_next_states:
            qval = self.qtable[new_ball_y][new_pad_y][pad_action]
            move_weight = 0
            if self.visited_states[new_ball_y][new_pad_y] < self.min_visits:
                move_weight = 10
            elif qval == max_q:
                move_weight = 1
            move_weights[pad_action] = move_weight
        move = random.choices(moves, weights=move_weights)[0]

        # Deterministic exploration: (for quicker training?)
        # Always prefers unexplored states to optimal states.
        # Doesn't seem to work. Agent is not exploring the environment.
        # move = 0
        # max_q = max(self.qtable[new_ball_y][new_pad_y])
        # for (new_ball_y, new_pad_y, pad_action) in possible_next_states:
        #     if self.visited_states[new_ball_y][new_pad_y] < self.min_visits:
        #         move = pad_action
        #         # if move != 0:
        #         #     print(new_ball_y, new_pad_y, move)
        #         #     print(f"Num visits: {self.visited_states[new_ball_y][new_pad_y]}")
        #         break
        #     qval = self.qtable[new_ball_y][new_pad_y][pad_action]
        #     if qval == max_q:
        #         move = pad_action

        return move

    def get_percent_of_states_explored(self):
        total_num_states = self.pong_game.canvas_height*self.pong_game.canvas_height
        num_visited_states = 0
        for i in range(self.pong_game.canvas_height):
            for j in range(self.pong_game.canvas_height):
                if self.visited_states[i][j] != 0:
                    num_visited_states += 1
        percent_visited = num_visited_states / total_num_states * 100

        return percent_visited

    def q(self, s, a):
        """The Q function Q(s,a) gives the quality of taking action a in state s."""
        ball_y = s[0]
        pad_y = s[1]
        try:
            new_ball_y = ball_y + self.ball_actions[self.ball.move_y]
            if new_ball_y < 0:
                new_ball_y = 0
            elif new_ball_y > self.pong_game.canvas_height - self.ball.height:
                new_ball_y = self.pong_game.canvas_height - self.ball.height

            pad_y_change = self.pad_actions[a]
            new_pad_y = pad_y + pad_y_change
            if new_pad_y < 0:
                new_pad_y = 0
            elif new_pad_y > self.pong_game.canvas_height - self.paddle.height:
                new_pad_y = self.pong_game.canvas_height - self.paddle.height

            next_state_q_values = self.qtable[new_ball_y][new_pad_y]
        except IndexError:
            raise Exception("One of the 2 indices for the Q table is out of bounds.")
        next_state_max_q = max(next_state_q_values)
        # Q-value equation for deterministic environment:
        self.qtable[ball_y][pad_y][a] = self.r(s, a) + self.gamma * next_state_max_q
        # print(f"ball_y: {ball_y}, pad_y: {pad_y}") # for debugging.
        # print(self.r(s,a), self.qtable[ball_y][pad_y]) # for debugging.

        return self.qtable[ball_y][pad_y][a]

    def qlearn(self, ball_y, pad_y):
        """Make the Q agent learn about its environment."""
        self.visited_states[ball_y][pad_y] += 1
        # Remove the following line to exclude epsilon decay:
        #self.epsilon = max(0.1, round(1 - (self.opponent.score+1)/self.pong_game.winning_score, 2))
        self.epsilon = round(1 - self.opponent.score/self.pong_game.winning_score, 10)
        rand_num = round(random.random(), 2) #0
        move = 0

        # move is the next state the paddle will go to.
        if rand_num < self.epsilon:  # exploration. Always true if rand_num = 0 and self.epsilon = 1
            #move = random.randint(0,2)  # 3 possible moves.
            move = self.exploration_fn(ball_y, pad_y)
            state = (ball_y, pad_y)
            self.q(state, move) # Update Q value in Q-table.
        else:                        # exploitation
            return self.play_game(ball_y, pad_y)

        return self.pad_actions[move]

    def play_game(self, ball_y, pad_y):
        """
        Make the Q agent play the Pong game after having
        learned all the Q values.
        """
        best_next_action = 0
        actions_q_values = self.qtable[ball_y][pad_y]
        # try:
        #     new_ball_y = ball_y + self.ball_actions[self.ball.move_y]
        #     if new_ball_y < 0:
        #         new_ball_y = 0
        #     elif new_ball_y > self.pong_game.canvas_height - self.ball.height:
        #         new_ball_y = self.pong_game.canvas_height - self.ball.height

        #     # pad_y_change = self.pad_actions[a]
        #     # new_pad_y = pad_y + pad_y_change
        #     # if new_pad_y < 0:
        #     #     new_pad_y = 0
        #     # elif new_pad_y > self.pong_game.canvas_height - self.paddle.height:
        #     #     new_pad_y = self.pong_game.canvas_height - self.paddle.height

        #     next_state_q_values = self.qtable[new_ball_y][pad_y]
        # except IndexError:
        #     raise Exception("One of the 2 indices for the Q table is out of bounds.")
        curr_q = actions_q_values[best_next_action]
        for action in range(len(actions_q_values)):
            action_q = actions_q_values[action]
            if action_q > curr_q:
                curr_q = action_q
                best_next_action = action

        return self.pad_actions[best_next_action]

    def reset_q_table(self):
        """Reset all Q values in the Q table to 0."""
        for i in range(self.num_y_directions):
            for j in range(self.num_ball_states):
                for k in range(self.num_paddle_states):
                    self.qtable[i][j][k] = 0
        self.epsilon = 1  # reset espilon as well.

    def read_q_table(self, filename):
        """Read the Q table from a file, if such a file exists."""
        with open(filename, 'rb') as fp:
            try:
                self.qtable = pickle.load(fp)
            except EOFError:
                print("No objects in the data file.")

    def write_q_table(self, filename):
        """Save the contents of the Q table to a file."""
        with open(filename, 'wb') as fp:
            pickle.dump(self.qtable, fp)

    def glimpse_qtable(self):
        """Print out a small part of the Q table."""
        print("Q table:")
        ball_y_states = [84, 324, 780]
        for ball_y_state in ball_y_states:
            for pad_y_state in range(0, len(self.qtable[1])+1, self.paddle.speed*10):
                # num_ball_states, num_paddle_states, num_actions
                print(f"State ({ball_y_state},{pad_y_state}): \
                    {self.qtable[ball_y_state][pad_y_state]}")  # take a glimpse at the q table.
        #print(self.qtable)


if __name__ == '__main__':
    pong_game = PongGame()
    # If the Q table already exists, then load the table and make the Q agent play the game.
    # Else train the Q agent by playing n games.
    num_simulations = 100000 # 100000000
    if os.path.isfile('pong-qtable.dat'):
        print("Game started.")
        pong_game.play()
        print("Game finished.")
    else:
        print("Q learning started.")
        start_time = time.time()
        pong_game.play(winning_score=num_simulations, qlearn_mode=True)
        end_time = time.time()
        total_time_elapsed = end_time - start_time
        print(f"Total time elapsed for {num_simulations} simulations: %.2fs" % (total_time_elapsed))
        avg_time_per_simulation = round(total_time_elapsed / num_simulations, 7)
        print(f"Avg. time per simulation: {avg_time_per_simulation}s")

        print("Game started.")
        pong_game.play()
        print("Game finished.")

        pong_game.plot_visited_states_percents()
