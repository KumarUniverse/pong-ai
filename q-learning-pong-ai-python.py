# A Q learning Pong AI agent built using Python.

import math
import random
import time
#import multiprocessing
import os
import pickle

# Global variables
DIRECTION = {"IDLE": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
# winning_score = 11  # 250
# num_rounds = 10


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
        #self.ai1.speed = 5  # 6
        self.ai2 = Paddle(self, 'right')
        self.ball = Ball(self)
        self.winning_score = 11
        self.qagent = Qagent(self, self.ai1, self.ai2, self.ball)

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
        self.ball = Ball(self)
        self.turn = loser
        victor.score += 1

    def play(self, winning_score=11, qlearn_mode=False):
        """
        Play the Pong game and keep playing until one of the
        players reaches the winning score.
        """
        self.winning_score = winning_score
        self.qlearn_mode = qlearn_mode
        opponent_prob = 0.5  # probability the opponent paddle hits the ball.
        middle_state = self.qagent.num_paddle_states // 2
        qagent_next_state = self.qagent.get_curr_paddle_state()

        while self.ai1.score < winning_score and self.ai2.score < winning_score:
            # On new serve (start of each turn), reset the paddles to their
            # center position and move the ball to the correct side.
            if self.turn:
                self.ai1.y = (self.canvas_height // 2) - 35
                self.ai2.y = (self.canvas_height // 2) - 35
                self.ball.move_x = DIRECTION["LEFT"] if self.turn == self.ai1 else DIRECTION["RIGHT"]
                self.ball.move_y = DIRECTION["UP"]
                self.ball.y = self.canvas_height - 150
                qagent_next_state = self.qagent.get_curr_paddle_state()
                self.turn = None

            # If the ball makes it past either of the paddles,
            # add a point to the winner and reset the turn to the loser.
            if self.ball.x <= 0:  # ai1 lost, ai2 won the round.
                #print("AI2 scored a goal.")
                #print("AI1 y: " + str(self.ai1.y))
                #print("Ball y: "  + str(self.ball.y))
                self.reset_turn(self.ai2, self.ai1)
                # Punish the AI every time it misses the ball.
                if qlearn_mode:
                    self.qagent.update_reward(-1)
            elif self.ball.x >= self.canvas_width - self.ball.width: # ai1 won, ai2 lost.
                #print("AI1 scored a goal.")
                self.reset_turn(self.ai1, self.ai2)

            # If the ball collides with the top and bottom bound limits, bounce it.
            if self.ball.y <= 0:
                self.ball.move_y = DIRECTION["DOWN"]
            elif self.ball.y >= self.canvas_height - self.ball.height:
                self.ball.move_y = DIRECTION["UP"]

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

            # Handle ai1 UP and DOWN movement.
            # Need to change.
            # if self.ai1.y + (self.ai1.height // 2) > self.ball.y:
            #     self.ai1.y -= self.ai1.speed
            # elif self.ai1.y + (self.ai1.height // 2) < self.ball.y:
            #     self.ai1.y += self.ai1.speed
            qagent_curr_state = self.qagent.get_curr_paddle_state()
            if qagent_curr_state > qagent_next_state:
                self.ai1.y -= self.ai1.speed
            elif qagent_curr_state < qagent_next_state:
                self.ai1.y += self.ai1.speed

            # Handle ai2 UP and DOWN movement.
            # The ai2 paddle's y always follows the y position of the ball.
            if self.ai2.y + (self.ai2.height // 2) > self.ball.y:
                self.ai2.y -= self.ai2.speed
            elif self.ai2.y + (self.ai2.height // 2) < self.ball.y:
                self.ai2.y += self.ai2.speed

            # Handle ai1 ball collision.
            if self.ball.x <= self.ai1.x + self.ai1.width and \
                self.ball.x + self.ball.width >= self.ai1.x:
                if self.ball.y <= self.ai1.y + self.ai1.height and \
                    self.ball.y + self.ball.height >= self.ai1.y:
                    #print("AI1 hit the ball.")
                    self.ball.x = self.ai1.x + self.ball.width
                    self.ball.move_x = DIRECTION["RIGHT"]
                    # Reward the Q agent every time it hits the ball.
                    if qlearn_mode:
                        self.qagent.update_reward(1)
                    # Move the Q agent's paddle back to the center.
                    qagent_next_state = middle_state

            # Handle ai2 ball collision.
            if self.ball.x <= self.ai2.x + self.ai2.width and \
                self.ball.x + self.ball.width >= self.ai2.x:
                if self.ball.y <= self.ai2.y + self.ai2.height and \
                    self.ball.y + self.ball.height >= self.ai2.y:
                    # self.ball.x = self.ai2.x + self.ball.width
                    # self.ball.move_x = DIRECTION["LEFT"]
                    # Q agent learns or plays the game.
                    if self.qlearn_mode:
                        #print("AI2 hit the ball.")
                        #print("AI1 y: " + str(self.ai1.y))
                        self.ball.x = self.ai2.x - self.ball.width
                        self.ball.move_x = DIRECTION["LEFT"]
                        #print("Q learning method called.")
                        qagent_next_state = self.qagent.qlearn()
                    else:
                        rand_num = round(random.random(), 1)
                        if rand_num <= opponent_prob:
                            self.ball.x = self.ai2.x - self.ball.width
                            self.ball.move_x = DIRECTION["LEFT"]
                            qagent_next_state = self.qagent.play_game()
                        else:
                            self.ball.x += self.ai2.width * 2

        if qlearn_mode:
            self.qagent.write_q_table('pong-qtable.dat')
            print("Q learning finished!")
        else:
            print("Pong game finished!")
        if self.ai1.score == winning_score:
            print("AI1 is the winner!")
        elif self.ai2.score == winning_score:
            print("AI2 is the winner!")
        print("AI1 score (Q agent): " + str(self.ai1.score))
        print(f"AI2 score ({int(opponent_prob*100)}% perfect agent): " + str(self.ai2.score))


class Qagent():
    """The Q agent playing the Pong game."""

    def __init__(self, pong_game, paddle, opponent, ball):
        self.pong_game = pong_game
        self.paddle = paddle
        self.opponent = opponent
        self.ball = ball
        self.alpha = 0.1  # learning rate.
        self.gamma = 0.8  # discount factor.
        self.epsilon = 1  # randomness factor. e=0 makes the agent greedy.
        self.num_y_directions = 2
        self.num_paddle_states = math.ceil(pong_game.canvas_height / self.paddle.height)
        self.num_ball_states = math.ceil(pong_game.canvas_height / self.ball.height)
        self.reward = [
                [[0]*self.num_paddle_states] * self.num_ball_states
        ] * self.num_y_directions
        self.qtable = None
        if os.path.isfile('pong-qtable.dat'):
            self.read_q_table('pong-qtable.dat')
            print(self.qtable[0][0])
        else:
            self.qtable = [
                [[0]*self.num_paddle_states] * self.num_ball_states
            ] * self.num_y_directions
        self.prev_state = None
        self.prev_action = None

    def update_reward(self, val):
        """Update the reward table based on the outcome of the action taken."""
        self.reward[self.prev_state[0]][self.prev_state[1]][self.prev_action] = val

    def r(self, s, a):
        """
        A reward function R(s,a) that gives the agent a reward for taking
        action a in state s.
        """
        return self.reward[s[0]][s[1]][a]

    def q(self, s, a):
        """The Q function Q(s,a) gives the quality of taking action a in state s."""
        ball_direction = s[0]
        ball_x_state = s[1]
        next_state_q_values = self.qtable[ball_direction][ball_x_state]
        next_state_max_q = max(next_state_q_values)
        self.qtable[ball_direction][ball_x_state][a] = round(
            self.qtable[ball_direction][ball_x_state][a]
            + self.alpha * (self.r(s, a) + self.gamma * next_state_max_q
                - self.qtable[ball_direction][ball_x_state][a]
            )
        , 3)

        return self.qtable[ball_direction][ball_x_state][a]

    def get_curr_paddle_state(self):
        """Return the current state of the paddle as an integer."""
        return self.paddle.y // (self.pong_game.canvas_height // self.num_paddle_states)

    def get_curr_ball_state(self):
        return self.ball.y // (self.pong_game.canvas_height // self.num_ball_states)

    def qlearn(self):
        """Make the Q agent learn about its environment."""
        self.epsilon = max(0.1, round(1 - (self.opponent.score+1)/self.pong_game.winning_score, 2))
        ball_direction = self.ball.move_y - 1
        ball_x_state = self.get_curr_ball_state()
        curr_state = (ball_direction, ball_x_state)
        rand_num = round(random.random(), 2)

        # move is the next state the paddle will go to.
        if rand_num < self.epsilon:  # exploration
            #print("random move made.")
            move = random.randint(0, self.num_paddle_states-1)
        else:
            #print("exploitation move made.")                        # exploitation
            move = self.get_curr_paddle_state()
            next_state_q_values = self.qtable[ball_direction][ball_x_state]
            curr_q = next_state_q_values[move]
            for state in range(len(next_state_q_values)):
                state_q = next_state_q_values[state]
                if state_q > curr_q:
                    curr_q = state_q
                    move = state
        self.prev_state = curr_state
        self.prev_action = move
        self.q(curr_state, move)

        return move

    def play_game(self):
        """
        Make the Q agent play the Pong game after having
        learned all the Q values.
        """
        best_next_state = self.get_curr_paddle_state()
        ball_direction = self.ball.move_y - 1
        ball_x_state = self.get_curr_ball_state()
        next_state_q_values = self.qtable[ball_direction][ball_x_state]
        curr_q = next_state_q_values[best_next_state]
        for state in range(len(next_state_q_values)):
            state_q = next_state_q_values[state]
            if state_q > curr_q:
                curr_q = state_q
                best_next_state = state

        return best_next_state

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

    def get_qtable_str(self):
        """Represent the Q table as a readable string."""
        output = "[\n"
        for i in range(self.num_y_directions):
            output += "[\n"
            for j in range(self.num_ball_states):
                for row in self.qtable[i][j]:
                    output += "\t" + str([round(x,2) for x in row]) + ",\n"
            output += "],\n"
        output += "]\n"

        return output

if __name__ == '__main__':
    pong_game = PongGame()
    # If the Q table already exists, then load the table and make the Q agent play the game.
    # Else train the Q agent by playing n games.
    num_simulations = 1000  # 11
    if os.path.isfile('pong-qtable.dat'):
        print("Game started")
        pong_game.play()
    else:
        print("Q learning started")
        start_time = time.time()
        pong_game.play(winning_score=num_simulations, qlearn_mode=True)
        end_time = time.time()
        total_time_elapsed = end_time - start_time
        print(f"Total time elapsed for {num_simulations} simulations: %.2fs" % (total_time_elapsed))
        avg_time_per_simulation = round(total_time_elapsed / num_simulations, 2)
        print(f"Avg. time per simulation: {avg_time_per_simulation}s")
