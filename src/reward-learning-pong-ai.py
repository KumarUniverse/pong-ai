# A Q learning Pong AI agent built using Python.

import math
import random
import time
import os
import pickle
from collections import defaultdict
from typing import DefaultDict

# Global variables
DIRECTION = {"IDLE": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}


class Ball():
    """The ball object (The cube that bounces back and forth)."""

    def __init__(self, pong_game):
        self.width = 18
        self.height = 18
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
        self.speed = 6


class PongGame():
    """The Pong game."""

    def __init__(self):
        # self.canvas = None
        # self.context = None
        self.canvas_width = 1400  # 700
        self.canvas_height = 1000  # 500

        self.ai1 = Paddle(self, 'left')
        #self.ai1.speed = 6  # 6
        self.ai2 = Paddle(self, 'right')
        self.ball = Ball(self)
        self.winning_score = 11
        self.qagent = Qagent(self, self.ai1, self.ai2, self.ball)

        #self.ai1.speed = 5
        # self.ai2.speed = 8  # make ai2's paddle speed slower than ai1.
        # self.running = False  # to check whether the game is running.
        self.turn = self.ai2  # it's the ai2's turn first.
        #self.round = 0
        self.qlearn_mode = False
        # self.timer = 0
        # self.color = '#000000'  # color the game background black.

        # self.menu()
        # self.listen()

    # No need for these methods:
    # def end_game_menu(text): pass
    # def menu(): pass
    # def draw(): pass
    # def listen(): pass

    def reset_turn(self, victor, loser):
        """Reset the turn to the loser once the ball goes past the loser's paddle."""
        #self.ball = Ball(self)
        self.turn = loser
        victor.score += 1

    def play(self, winning_score=11, qlearn_mode=False):
        """
        Play the Pong game and keep playing until one of the
        players reaches the winning score.
        """
        self.winning_score = winning_score
        self.qlearn_mode = qlearn_mode
        opponent_prob = 0.8  # probability the opponent paddle hits the ball.
        middle_state = self.qagent.num_paddle_states // 2
        qagent_next_state = self.qagent.get_curr_paddle_state(self.ai1.y)
        # ball_start_end_states = {}
        # ball_start_states = set()
        # hit_paddle_states = set()
        #visited_paddle_states = set()
        # opp_visited_paddle_states = set()
        #visited_ball_states = set()
        # hit_paddle_ys = set()
        # opp_hit_paddle_ys = set()
        ball_min_y = self.canvas_height
        ball_max_y = 0
        best_rally = 0  # best number of times ball went back and forth.
        rally = 0
        #ball_start_state = None
        #ball_end_state = None
        #q_start_state = None
        i = 0
        # ball_path = defaultdict(list)

        while self.ai1.score < winning_score and self.ai2.score < winning_score:
            # On new serve (start of each turn), reset the paddles to their
            # center position and move the ball to the correct side.
            if self.turn:
                self.ai1.y = (self.canvas_height // 2) - 35
                self.ai2.y = (self.canvas_height // 2) - 35
                self.ball.move_x = DIRECTION["LEFT"] if self.turn == self.ai1 else DIRECTION["RIGHT"]
                self.ball.move_y = DIRECTION["UP"]
                self.ball.x = pong_game.canvas_width // 2
                self.ball.y = self.canvas_height - 150
                qagent_next_state = self.qagent.get_curr_paddle_state(self.ai1.y)
                self.turn = None

            # If the ball makes it past either of the paddles,
            # add a point to the winner and reset the turn to the loser.
            if self.ball.x <= 0:  # ai1 lost, ai2 won the round.
                rally = 0
                #print("AI2 scored a goal.")
                #print("AI1 y: " + str(self.ai1.y))
                #print("Ball y: "  + str(self.ball.y))
                self.reset_turn(self.ai2, self.ai1)
                # Punish AI1 every time it misses the ball.
                # if qlearn_mode:
                #     self.qagent.update_reward(-1)
                # This seems to remove some unnecessary rewards:
                # self.qagent.prev_state = None
                # self.qagent.prev_action = None
            elif self.ball.x >= self.canvas_width - self.ball.width: # ai1 won, ai2 lost.
                rally = 0
                #print("AI1 scored a goal.")
                self.reset_turn(self.ai1, self.ai2)
                # This doesn't really do anything:
                # self.qagent.prev_state = None
                # self.qagent.prev_action = None

            # If the ball collides with the top and bottom bound limits, bounce it.
            if self.ball.y < 0:
                self.ball.y = 0 # I added this line in.
                self.ball.move_y = DIRECTION["DOWN"]
                # if qlearn_mode and q_start_state and q_start_state[0] == 0 and q_start_state[1] == 18 and q_start_state[2] == 7:
                    # ball_path[q_start_state].append((self.ball.x, self.ball.y))
            elif self.ball.y > self.canvas_height - self.ball.height:
                self.ball.y = self.canvas_height - self.ball.height # I added this line in.
                self.ball.move_y = DIRECTION["UP"]
                # if qlearn_mode and q_start_state and q_start_state[0] == 0 and q_start_state[1] == 18 and q_start_state[2] == 7:
                    # ball_path[q_start_state].append((self.ball.x, self.ball.y))

            # Handle ai1 wall collision.
            if self.ai1.y < 0:
                self.ai1.y = 0
            elif self.ai1.y > self.canvas_height - self.ai1.height:
                self.ai1.y = self.canvas_height - self.ai1.height

            # Handle ai2 wall collision.
            if self.ai2.y < 0:
                self.ai2.y = 0
            elif self.ai2.y > self.canvas_height - self.ai2.height:
                self.ai2.y = self.canvas_height - self.ai2.height

            # Keep track of visited ball states.
            #curr_ball_state = self.qagent.get_curr_ball_state(self.ball.y)
            # if curr_ball_state == 55: # NEED TO REMOVE.
            #     raise Exception("Ball is in its max's state.")
            #visited_ball_states.add(curr_ball_state)

            # Min and max y:
            ball_min_y = min(ball_min_y, self.ball.y)
            ball_max_y = max(ball_max_y, self.ball.y)

            # Handle ball movement.
            # Move ball in intended direction based on move_y and move_x values.
            # The ball travels faster in the x direction than in the y direction.
            if self.ball.move_y == DIRECTION["UP"]:
                #print("Ball is moving up!")
                self.ball.y -= 6 #int(self.ball.speed // 1.5) # 6
            if self.ball.move_y == DIRECTION["DOWN"]:
                self.ball.y += 6 #int(self.ball.speed // 1.5) # 6

            if self.ball.move_x == DIRECTION["LEFT"]:
                #print("Ball is moving left!")
                self.ball.x -= self.ball.speed
            if self.ball.move_x == DIRECTION["RIGHT"]:
                self.ball.x += self.ball.speed

            # Handle ai1 UP and DOWN movement.
            # Need to change.
            # if self.ai1.y + (self.ai1.height // 2) > self.ball.y:
            #     self.ai1.y -= self.ai1.speed
            # elif self.ai1.y + (self.ai1.height // 2) < self.ball.y:
            #     self.ai1.y += self.ai1.speed
            qagent_curr_state = self.qagent.get_curr_paddle_state(self.ai1.y)
            #visited_paddle_states.add(qagent_curr_state)
            if qagent_curr_state > qagent_next_state:
                #print("AI1 is moving up")
                self.ai1.y -= self.ai1.speed
            elif qagent_curr_state < qagent_next_state:
                #print("AI1 is moving down")
                self.ai1.y += self.ai1.speed

            # Handle ai2 UP and DOWN movement.
            # The ai2 paddle's y always follows the y position of the ball.
            #opp_curr_state = self.qagent.get_curr_opp_state(self.ai2.y)
            #opp_visited_paddle_states.add(opp_curr_state)
            if self.ai2.y > self.ball.y - (self.ai2.height // 2):
                #print("AI2 is moving up", self.ball.x, self.ball.y)
                self.ai2.y -= self.ai2.speed
            elif self.ai2.y < self.ball.y - (self.ai2.height // 2):
                #print("AI2 is moving down", self.ai2.y, self.ball.x, self.ball.y)
                self.ai2.y += self.ai2.speed

            # Handle ai1 (q agent) ball collision.
            if (self.ball.x <= self.ai1.x + self.ai1.width) and \
                (self.ball.x + self.ball.width >= self.ai1.x):
                if (self.ball.y <= self.ai1.y + self.ai1.height) and \
                    (self.ball.y + self.ball.height >= self.ai1.y):
                    rally += 1
                    # if qagent_curr_state != qagent_next_state:
                    #     print("Current state is not equal to destination state.")
                    # if qlearn_mode and q_start_state and q_start_state[0] == 0 and q_start_state[1] == 18 and q_start_state[2] == 7:
                        # ball_path[q_start_state].append(str((self.ball.x, self.ball.y)) + "HIT " + str((self.ai1.x, self.ai1.y)) + " " + str(qagent_curr_state))
                    # Reward the Q agent (AI1) every time it hits the ball.
                    if qlearn_mode and self.qagent.prev_state is not None:# and \
                        #self.qagent.prev_action is not None:
                        #print(f"adding reward to {self.qagent.prev_state}")
                        #if sum(self.qagent.reward[self.qagent.prev_state[0]][self.qagent.prev_state[1]][self.qagent.prev_state[2]]) < 1:
                        self.qagent.update_reward(1, self.ai1.y)
                    # THE BALL MOVES PREDICTABLY!
                    #ball_end_state = self.qagent.get_curr_ball_state(self.ball.y)
                    #hit_paddle_states.add(self.qagent.get_curr_paddle_state())
                    #hit_paddle_ys.add(self.ai1.y)
                    # if ball_start_state in ball_start_end_states: # and ball_end_state != 18:
                    #     pass
                    #     # if (prev_end_state := ball_start_end_states[ball_start_state]) == ball_end_state:
                    #     #     raise NameError(f"Ball is not moving predictably. Start state: {ball_start_state}. Prev end state: {prev_end_state}. Curr end state: {ball_end_state}")
                    # elif ball_start_state is None:
                    #     pass
                    #     #print("Opponent hit ball." if opponent_hit_ball else "Opponent did not hit ball.")
                    # else:
                    #     #print(f"Adding start state {ball_start_state} and end state {ball_end_state} and x val {self.ball.x}")
                    #     ball_start_end_states[ball_start_state] = ball_end_state
                    #print("AI1 hit the ball.")
                    self.ball.x = self.ai1.x + self.ai1.width # might need to add this back.
                    self.ball.move_x = DIRECTION["RIGHT"]
                    # This does not seem to do anything:
                    # self.qagent.prev_state = None
                    # self.qagent.prev_action = None
                    # Move the Q agent's paddle back to the center after hitting the ball.
                    qagent_next_state = middle_state

            # Handle ai2 ball collision.
            if (self.ball.x <= self.ai2.x + self.ai2.width) and \
                (self.ball.x  + self.ball.width >= self.ai2.x):
                if (self.ball.y <= self.ai2.y + self.ai2.height) and \
                    (self.ball.y + self.ball.height >= self.ai2.y):
                    #print(f"Ball x when AI2 hits: {self.ball.x}")
                    #print("Paddle 2 hit ball!")
                    # self.ball.x = self.ai2.x - self.ball.width
                    # self.ball.move_x = DIRECTION["LEFT"]
                    # Q agent learns or plays the game.
                    #ball_start_state = self.qagent.get_curr_ball_state(self.ball.y)
                    #opp_hit_paddle_ys.add(self.ai2.y)
                    #opponent_hit_ball = True
                    #ball_start_states.add(ball_start_state)
                    # if ball_start_state == None:
                    #     raise Exception("BALL IS NONE!")
                    if self.qlearn_mode:
                        rally += 1
                        #print("AI2 hit the ball.")
                        #print("AI1 y: " + str(self.ai1.y))
                        #print("Q learning method called.")
                        self.ball.x = self.ai2.x - self.ball.width # might need to add this back.
                        self.ball.move_x = DIRECTION["LEFT"]
                        qagent_next_state = self.qagent.qlearn(self.ball, self.ai1)
                        # Need to remove this:
                        # ball_direction = self.ball.move_y - 1
                        # ball_y_state = self.qagent.get_curr_ball_state(self.ball.y)
                        # paddle_state = self.qagent.get_curr_paddle_state(self.ai1.y)
                        # q_start_state = (ball_direction, ball_y_state, paddle_state, i)
                        # if ball_direction == 0 and ball_y_state == 18 and paddle_state == 7:
                            # i += 1
                            # ball_path[q_start_state].append((self.ball.x, self.ball.y))
                    else:  # play mode
                        rand_num = round(random.random(), 1)
                        if rand_num <= opponent_prob:
                            rally += 1
                            self.ball.x = self.ai2.x - self.ball.width # might need to add this back.
                            self.ball.move_x = DIRECTION["LEFT"]
                            qagent_next_state = self.qagent.play_game(self.ball, self.ai1)
                        else: # misses ball
                            self.ball.x += self.ai2.width * 2 + 1

                best_rally = max(best_rally, rally)


        if qlearn_mode:
            #self.qagent.write_reward_table('pong-reward-table.dat')
            print("Reward learning finished!")
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
        self.qagent.glimpse_reward_table()
        # print(f"ball start and end states: {ball_start_end_states}")
        # print(f"ball start states: {ball_start_states}")
        # print(f"ball end states: {hit_paddle_states}")
        #print(f"Q agent's visited paddle states: {visited_paddle_states}")  # All paddle state get visited.
        # print(f"Opponent's visited states: {opp_visited_paddle_states}")
        #print(f"Ball's visited states: {visited_ball_states}")  # All ball states get visited.
        print(f"Ball min y: {ball_min_y}, Ball max y: {ball_max_y}")
        # print(f"ys of ai1 paddle hits: {hit_paddle_ys}")
        # print(f"ys of opp paddle hits: {opp_hit_paddle_ys}")
        print(f"Best rally: {best_rally}")
        # print(f"Ball path from (0,18,7): {ball_path}")
        print()


class Qagent():
    """The Q agent playing the Pong game."""

    def __init__(self, pong_game, paddle, opponent, ball):
        self.pong_game = pong_game
        self.paddle = paddle
        self.opponent = opponent
        self.ball = ball
        self.alpha = 0.1  # learning rate.
        self.gamma = 0.8  # discount factor. # Before: 0.8
        self.epsilon = 1  # randomness factor. e=0 makes the agent greedy.
        self.num_y_directions = 2
        self.num_paddle_states = math.ceil(pong_game.canvas_height / self.paddle.height)  # 15
        self.num_ball_states = math.ceil(pong_game.canvas_height / self.ball.height)      # 56
        #self.num_ball_ys = pong_game.canvas_height
        # print(f"Num of paddle states: {self.num_paddle_states}")
        # print(f"Num of ball states: {self.num_ball_states}")
        # self.reward = [
        #         [[[0]*self.num_paddle_states] * self.num_paddle_states] * self.num_ball_states
        # ] * self.num_y_directions
        self.reward = [
            [[[0 for _ in range(self.num_paddle_states)
                    ] for _ in range(self.num_paddle_states)
                ] for _ in range(self.num_ball_states) #self.num_ball_ys
            ] for _ in range(self.num_y_directions)
        ]
        # if os.path.isfile('pong-reward-table.dat'):
        #     self.read_reward_table('pong-reward-table.dat')
        # else:
        #     self.reward = [
        #         [[0]*self.num_paddle_states] * self.num_ball_states
        # ] * self.num_y_directions
        #self.last_pad_state_size = self.pong_game.canvas_height % self.paddle.height
        # self.last_ball_state_size = self.pong_game.canvas_height % self.ball.height
        self.prev_state = None
        #self.prev_action = None

    def update_reward(self, val, paddle_y):
        """Update the reward table based on the outcome of the action taken."""
        curr_pad_state = self.get_curr_paddle_state(paddle_y)
        if self.prev_state is None:
            raise Exception("WARNING: prev_state is None!")
        # if self.prev_state is None and self.prev_action is None:
        #     raise NameError("WARNING: prev_state and prev_action are both None!")
        # if self.reward[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][curr_pad_state] != 1:
        #     print(f"state {self.prev_state} action {curr_pad_state} changed to 1!")
        #     print(f"paddle min y: {paddle_y}, paddle max y: {paddle_y + self.paddle.height}, ball y: {ball_y}, ball max y: {ball_y + self.ball.height}")
        self.reward[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][curr_pad_state] += val
        #print("updating reward")
        #self.reward[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_action] = val
        self.prev_state = None
        #self.prev_action = None

    def r(self, s, a):
        """
        A reward function R(s,a) that gives the agent a reward for taking
        action a in state s.
        """
        return self.reward[s[0]][s[1]][s[2]][a]

    # def q(self, s, a):
    #     """The Q function Q(s,a) gives the quality of taking action a in state s."""
    #     ball_direction = s[0]
    #     ball_x_state = s[1]
    #     try:
    #         next_state_q_values = self.reward[ball_direction][ball_x_state]
    #     except IndexError:
    #         print(f"ball direction value: {ball_direction}")
    #         print(f"ball x state: {ball_x_state}")
    #     next_state_max_q = max(next_state_q_values)
    #     self.qtable[ball_direction][ball_x_state][a] = (
    #         self.qtable[ball_direction][ball_x_state][a]
    #         + self.alpha * (self.r(s, a) + self.gamma * next_state_max_q
    #             - self.qtable[ball_direction][ball_x_state][a]
    #         )
    #     )

    #     return self.qtable[ball_direction][ball_x_state][a]

    def get_curr_paddle_state(self, paddle_y):
        """Return the current state of the paddle as an integer."""
        curr_state = (paddle_y + self.paddle.height - 1) // self.paddle.height
        if curr_state >= self.num_paddle_states:
            raise Exception("curr_state exceeds paddle states!")
            #return self.num_paddle_states - 1

        return curr_state

    def get_curr_opp_state(self, opponent_y):
        curr_state = (opponent_y + self.opponent.height - 1) // self.opponent.height
        if curr_state >= self.num_paddle_states:
            raise Exception("curr_state exceeds paddle states!")
            #return self.num_paddle_states - 1

        return curr_state

    def get_curr_ball_state(self, ball_y):
        curr_state = (ball_y + self.ball.height - 1) // self.ball.height
        #curr_state = self.ball.y // self.ball.height
        if curr_state >= self.num_ball_states:
            raise Exception("curr_state exceeds ball states!")
            #return self.num_ball_states - 1

        return curr_state

        # if ball_y > self.num_ball_ys:
        #     raise Exception("Ball state is out of bounds!")
        # return ball_y

    def qlearn(self, ball, paddle):
        """Make the Q agent learn about its environment."""
        self.epsilon = max(0.1, round(1 - (self.opponent.score+1)/self.pong_game.winning_score, 2))
        ball_direction = ball.move_y - 1
        ball_y_state = self.get_curr_ball_state(ball.y)
        paddle_state = self.get_curr_paddle_state(paddle.y)
        self.prev_state = (ball_direction, ball_y_state, paddle_state)
        # if self.prev_state != (1, 13, 7) and self.prev_state != (0, 18, 7) and self.prev_state != (0, 48, 7):
        #     print(self.prev_state)
        # Others: (0,28,7), (1,32,7)
        rand_num = round(random.random(), 2)
        move = 0

        # move is the next state the paddle will go to.
        if rand_num < self.epsilon:  # exploration
            move = random.randint(0, self.num_paddle_states-1)
            #print(f"random move made. Move: {move}")
        else:                        # exploitation
            #print("exploitation move made.")
            move = paddle_state
            next_state_reward_values = self.reward[ball_direction][ball_y_state][paddle_state]
            # move = max(
            #     range(len(next_state_reward_values)), key=next_state_reward_values.__getitem__)
            move = max(
                [(r, i) for i, r in enumerate(next_state_reward_values)])[1]
            #print(move)
        #self.prev_action = move
        #self.q(curr_state, move)

        return move

    def play_game(self, ball, paddle):
        """
        Make the Q agent play the Pong game after having
        learned all the Q values.
        """
        if ball.move_y != 1 and ball.move_y != 2:
            raise NameError("Ball is not moving up or down.")
        ball_direction = ball.move_y - 1
        ball_y_state = self.get_curr_ball_state(ball.y)
        paddle_state = self.get_curr_paddle_state(paddle.y)
        best_next_state = paddle_state
        try:
            next_state_reward_values = self.reward[ball_direction][ball_y_state][paddle_state]
        except IndexError:
                print(f"ball direction value: {ball_direction}")
                print(f"ball x state: {ball_y_state}")
        curr_reward = next_state_reward_values[best_next_state]
        #print(f"Next state rewards: {next_state_reward_values}")
        for state in range(len(next_state_reward_values)):
            state_reward = next_state_reward_values[state]
            if state_reward > curr_reward:
                curr_reward = state_reward
                best_next_state = state

        return best_next_state

    # def reset_q_table(self):
    #     """Reset all Q values in the Q table to 0."""
    #     for i in range(self.num_y_directions):
    #         for j in range(self.num_ball_states):
    #             for k in range(self.num_paddle_states):
    #                 self.qtable[i][j][k] = 0
    #     self.epsilon = 1  # reset espilon as well.

    def read_reward_table(self, filename):
        """Read the Q table from a file, if such a file exists."""
        with open(filename, 'rb') as fp:
            try:
                self.reward = pickle.load(fp)
            except EOFError:
                print("No objects in the data file.")

    def write_reward_table(self, filename):
        """Save the contents of the Q table to a file."""
        with open(filename, 'wb') as fp:
            pickle.dump(self.reward, fp)

    # def get_qtable_str(self):
    #     """Represent the Q table as a readable string."""
    #     output = "[\n"
    #     for i in range(self.num_y_directions):
    #         output += "[\n"
    #         for j in range(self.num_ball_states):
    #             for row in self.qtable[i][j]:
    #                 output += "\t" + str([round(x,2) for x in row]) + ",\n"
    #         output += "],\n"
    #     output += "]\n"

    #     return output

    def glimpse_reward_table(self):
        print("reward table:")
        for i in range(10):
            # num_y_directions, num_ball_states, num_paddle_states
            print(f"State (0,{i},0): {self.reward[0][i][0]}")  # take a glimpse at the reward table.
        print(f"State (0,18,7): {self.reward[0][18][7]}")  # take a glimpse at the reward table.
        print(f"State (1,13,7): {self.reward[1][13][7]}")  # take a glimpse at the reward table.
        print(f"State (0,47,8): {self.reward[0][47][8]}")  # take a glimpse at the reward table.

    def get_reward_str(self):
        """Represent the Q table as a readable string."""
        output = "[\n"
        for i in range(self.num_y_directions):
            output += "[\n"
            for j in range(self.num_ball_states):
                for row in self.reward[i][j]:
                    output += "\t" + str([round(x,2) for x in row]) + ",\n"
            output += "],\n"
        output += "]\n"

        return output

if __name__ == '__main__':
    pong_game = PongGame()
    # If the Q table already exists, then load the table and make the Q agent play the game.
    # Else train the Q agent by playing n games.
    num_simulations = 1000  # 11
    if os.path.isfile('pong-reward-table.dat'):
        print("Game started")
        pong_game.play()
    else:  # Reward learning.
        print("Reward learning started")
        start_time = time.time()
        pong_game.play(winning_score=num_simulations, qlearn_mode=True)
        end_time = time.time()
        total_time_elapsed = end_time - start_time
        print(f"Total time elapsed for {num_simulations} simulations: %.2fs" % (total_time_elapsed))
        avg_time_per_simulation = round(total_time_elapsed / num_simulations, 2)
        print(f"Avg. time per simulation: {avg_time_per_simulation}s\n")

        print("Game started")
        #pong_game.qagent.glimpse_reward_table()
        pong_game.play()
