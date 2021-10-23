// Global Variables
var DIRECTION = {
    IDLE: 0,
    UP: 1,
    DOWN: 2,
    LEFT: 3,
    RIGHT: 4
};

var winningScore = 11; // 250;
//var numRounds = 10;

// The ball object (The cube that bounces back and forth)
var Ball = {
    new: function () {
        return {
            width: 18,
            height: 18,
            x: (this.canvas.width / 2),
            y: (this.canvas.height / 2),
            moveX: DIRECTION.IDLE,
            moveY: DIRECTION.IDLE,
            speed: 9
        };
    }
};

// The paddle object (The two lines that move up and down)
var Paddle = {
    new: function (side) {
        return {
            width: 18,
            height: 70,
            x: side === 'left' ? 150 : this.canvas.width - 150,
            y: (this.canvas.height / 2) - 35,
            score: 0,
            move: DIRECTION.IDLE,
            speed: 6
        };
    }
};

// The Q agent used to play the Pong game.
var Qagent = {
    new: function () {
        return {
            alpha: 0.1,  // learning rate.
            gamma: 0.8,  // discount factor.
            epsilon: 1,  // randomness factor. e=0 makes the agent greedy.
            qtable: Array(2).fill(Array(56)).fill(Array(15)).fill(0)
            // ^^ There are 1,680 (2*56*15) unique states in the qtable.
            // 2 states for the direction of the ball,
            // 56 states for the location of the ball,
            // 15 states for the location of the agent's paddle.
        };
    },

    // Reward function
    r: function () {
        return 0;
    },

    // Q function
    q: function (s, a) {
        return 0;
    },

    // Used to make the Q agent learn about its Pong environment.
    qlearn: function () {
        winningScore = 11; //250;
        winningScore = 11; // reset winning score.
    },

    // https://stackoverflow.com/questions/56904136/how-to-save-an-array-to-file-and-later-read-it-into-an-array-variable-in-nodejs/56904711
    // Used to save the contents of the qtable to a file.
    writeQtable: function (qtable, path) {
        // fs.writeFileSync(path, JSON.stringify(qtable));
        return 0;
    },

    readQtable: function (path) {
        // fileContent = fs.readFileSync(path);
        // JSON.parse(fileContent);
        return 0;
    },

    // Used to represent the Q table as a readable string.
    get_qtable_str: function () {
        let output = "[\n";
        for (let i = 0; i < qtable.length; i++) {
            output += "[\n";
            for (let j = 0; j < qtable[0].length; j++) {
                output += "[\n";
                for (let k = 0; k < qtable[0][0].length; k++) {
                    output += qtable[i][j][k] + ", ";
                }
                output += "]\n";
            }
            output += "]\n";
        }
        output += "]\n";

        return output;
    }
};

var Game = {
    initialize: function () {
        this.canvas = document.querySelector('canvas');
        this.context = this.canvas.getContext('2d');

        this.canvas.width = 1400;
        this.canvas.height = 1000;

        this.canvas.style.width = (this.canvas.width / 2) + 'px';
        this.canvas.style.height = (this.canvas.height / 2) + 'px';

        this.ai = Paddle.new.call(this, 'left');
        this.qagent = Qagent.new.call(this);
        this.ai2 = Paddle.new.call(this, 'right');
        //this.ai2.width = 0;  // Use to remove opponent's paddle.
        //this.ai2.height = 0; // Use to remove opponent's paddle.
        this.ball = Ball.new.call(this);

        //this.ai2.speed = 8;      // Make ai2's paddle speed slower than ai.
        this.running = false;      // To check whether the game is running.
        this.over = false;         // To check if the game is over.
        this.turn = this.ai2;      // It's the ai2's turn first.
        this.timer = this.round = 0;
        this.color = '#000000';    // Color the game background black.

        Pong.menu();
        Pong.listen();
    },

    endGameMenu: function (text) {
        // Change the canvas font size and color
        Pong.context.font = '50px Courier New';
        Pong.context.fillStyle = this.color; // Color of background is black.

        // Draw the rectangle behind the 'Press any key to begin' text.
        Pong.context.fillRect(
            Pong.canvas.width / 2 - 350,
            Pong.canvas.height / 2 - 48,
            700,
            100
        );

        // Change the canvas color;
        Pong.context.fillStyle = '#ffffff'; // Color of the text is white.

        // Draw the end game menu text ('Game Over' and 'Winner')
        Pong.context.fillText(text,
            Pong.canvas.width / 2,
            Pong.canvas.height / 2 + 15
        );

        setTimeout(function () {
            Pong = Object.assign({}, Game);
            Pong.initialize();
        }, 3000);
    },

    menu: function () {
        // Draw all the Pong objects in their current state
        Pong.draw();

        // Change the canvas font size and color
        this.context.font = '50px Courier New';
        this.context.fillStyle = this.color;

        // Draw the rectangle behind the 'Press Enter to begin' text.
        this.context.fillRect(
            this.canvas.width / 2 - 350,
            this.canvas.height / 2 - 48,
            700,
            100
        );

        // Change the canvas color;
        this.context.fillStyle = '#ffffff';

        // Draw the 'press Enter to begin' text
        this.context.fillText('Press Enter to begin',
            this.canvas.width / 2,
            this.canvas.height / 2 + 15
        );
    },

    // Update all objects (move paddles, ball, increment score, etc.)
    update: function () {
        if (!this.over) {
            // If the ball makes it past either of the paddles,
            // add a point to the winner and reset the turn.
            // If the ball collides with the top and bottom bound limits, bounce it.
            if (this.ball.x <= 0)
                Pong._resetTurn.call(this, this.ai2, this.ai);
            if (this.ball.x >= this.canvas.width - this.ball.width)
                this.ball.moveX = DIRECTION.LEFT; // bounce off the right wall.
                //Pong._resetTurn.call(this, this.ai, this.ai2);
            if (this.ball.y <= 0)
                this.ball.moveY = DIRECTION.DOWN;
            if (this.ball.y >= this.canvas.height - this.ball.height)
                this.ball.moveY = DIRECTION.UP;

            // Move player if the player.move value
            // was updated by a keyboard event
            // if (this.player.move === DIRECTION.UP)
            //   this.player.y -= this.player.speed;
            // else if (this.player.move === DIRECTION.DOWN)
            //   this.player.y += this.player.speed;

            // On new serve (start of each turn),
            // reset the paddles to their center position,
            // move the ball to the correct side.
            if (Pong._turnDelayIsOver.call(this) && this.turn) {
                this.ai.y = (this.canvas.height / 2) - 35;
                //this.ai2.y = (this.canvas.height / 2) - 35;
                this.ball.moveX =
                    this.turn === this.ai ?
                        DIRECTION.LEFT : DIRECTION.RIGHT;
                this.ball.moveY = DIRECTION.UP; //DIRECTION.IDLE;
                this.ball.y = this.canvas.height - 150;
                this.turn = null;
            }

            // If the ai collides with the wall, update the x and y coords.
            if (this.ai.y <= 0) {
                this.ai.y = 0;
            }
            else if (this.ai.y >=
                (this.canvas.height - this.ai.height)) {
                this.ai.y = (this.canvas.height - this.ai.height);
            }

            // Handled ai2 wall collision
            if (this.ai2.y <= 0) {
                this.ai2.y = 0;
            }
            else if (this.ai2.y >=
                (this.canvas.height - this.ai2.height)) {
                this.ai2.y = (this.canvas.height - this.ai2.height);
            }

            // Move ball in intended direction based on moveY and moveX values
            if (this.ball.moveY === DIRECTION.UP) {
                this.ball.y -= (this.ball.speed / 1.5);
            }
            else if (this.ball.moveY === DIRECTION.DOWN) {
                this.ball.y += (this.ball.speed / 1.5);
            }
            if (this.ball.moveX === DIRECTION.LEFT) {
                this.ball.x -= this.ball.speed;
            }
            else if (this.ball.moveX === DIRECTION.RIGHT) {
                this.ball.x += this.ball.speed;
            }

            // Handle ai UP and DOWN movement
            if (this.ai.y > this.ball.y - (this.ai.height / 2)) {
                if (this.ball.moveX === DIRECTION.RIGHT)
                    this.ai.y -= this.ai.speed;
                else
                    this.ai.y -= this.ai.speed;
            }
            if (this.ai.y < this.ball.y - (this.ai.height / 2)) {
                if (this.ball.moveX === DIRECTION.RIGHT)
                    this.ai.y += this.ai.speed;
                else
                    this.ai.y += this.ai.speed;
            }

            // Handle ai2 UP and DOWN movement
            if (this.ai2.y > this.ball.y - (this.ai2.height / 2)) {
                if (this.ball.moveX === DIRECTION.RIGHT)
                    this.ai2.y -= this.ai2.speed;
                else
                    this.ai2.y -= this.ai2.speed;
            }
            if (this.ai2.y < this.ball.y - (this.ai2.height / 2)) {
                if (this.ball.moveX === DIRECTION.RIGHT)
                    this.ai2.y += this.ai2.speed;
                else
                    this.ai2.y += this.ai2.speed;
            }

            // Handle ai-ball collisions
            if (this.ball.x - this.ball.width <= this.ai.x &&
                this.ball.x >= this.ai.x - this.ai.width) {
                if (this.ball.y <= this.ai.y + this.ai.height &&
                    this.ball.y + this.ball.height >= this.ai.y) {
                    this.ball.x = this.ai.x + this.ball.width;
                    this.ball.moveX = DIRECTION.RIGHT;
                }
            }

            // Handle ai2-ball collision
            if (this.ball.x - this.ball.width <= this.ai2.x &&
                this.ball.x >= this.ai2.x - this.ai2.width) {
                if (this.ball.y <= this.ai2.y + this.ai2.height &&
                    this.ball.y + this.ball.height >= this.ai2.y) {
                    if (Math.random() <= 0.7) {
                        this.ball.x = this.ai2.x - this.ball.width;
                        this.ball.moveX = DIRECTION.LEFT;
                    } else {
                        this.ball.x += this.ball.width * 2;
                    }
                }
            }
        }

        // Handle the end of round transition
        // Check to see if the ai or the ai2 won the game.
        if (this.ai.score === winningScore) {
            this.over = true;
            setTimeout(function () { Pong.endGameMenu('AI1 is the Winner!'); }, 1000);
            // Check to see if there are any more rounds left.
        }
        else if (this.ai2.score === winningScore) {
            this.over = true;
            setTimeout(function () { Pong.endGameMenu('AI2 is the Winner!'); }, 1000);
        }
    },

    // Draw the objects to the canvas element
    draw: function () {
        // Clear the Canvas
        this.context.clearRect(
            0,
            0,
            this.canvas.width,
            this.canvas.height
        );

        // Set the fill style to black
        this.context.fillStyle = this.color;

        // Draw the background
        this.context.fillRect(
            0,
            0,
            this.canvas.width,
            this.canvas.height
        );

        // Set the fill style to white (For the paddles and the ball)
        this.context.fillStyle = '#ffffff';

        // Draw the Player
        this.context.fillRect(
            this.ai.x,
            this.ai.y,
            this.ai.width,
            this.ai.height
        );

        // Draw the Paddle
        this.context.fillRect(
            this.ai2.x,
            this.ai2.y,
            this.ai2.width,
            this.ai2.height
        );

        // Draw the Ball
        if (Pong._turnDelayIsOver.call(this)) {
            this.context.fillRect(
                this.ball.x,
                this.ball.y,
                this.ball.width,
                this.ball.height
            );
        }

        // Draw the net (Line in the middle)
        this.context.beginPath();
        this.context.setLineDash([7, 15]);
        this.context.moveTo((this.canvas.width / 2), this.canvas.height - 140);
        this.context.lineTo((this.canvas.width / 2), 140);
        this.context.lineWidth = 10;
        this.context.strokeStyle = '#ffffff';
        this.context.stroke();

        // Set the default canvas font and align it to the center
        this.context.font = '100px Courier New';
        this.context.textAlign = 'center';

        // Draw the players' score (left)
        this.context.fillText(
            this.ai.score.toString(),
            (this.canvas.width / 2) - 300,
            200
        );

        // Draw the ai2's score (right)
        this.context.fillText(
            this.ai2.score.toString(),
            (this.canvas.width / 2) + 300,
            200
        );

        // Change the font size for the center score text
        this.context.font = '30px Courier New';

        // Draw the round number (center)
        // this.context.fillText(
        // 	'Round ' + (Pong.round + 1),
        // 	(this.canvas.width / 2),
        // 	35
        // );

        // Change the font size for the center score value
        this.context.font = '40px Courier';
    },

    loop: function () {
        Pong.update();
        Pong.draw();

        // If the game is not over, draw the next frame.
        if (!Pong.over) requestAnimationFrame(Pong.loop);
    },

    listen: function () {
        document.addEventListener('keydown', function (key) {
            // Handle the 'Press Enter to begin' function and start the game.
            if (key.key == 'Enter' && Pong.running === false) {
                Pong.running = true;
                window.requestAnimationFrame(Pong.loop);
            }

            // Handle up arrow key event
            //if (key.key === 'ArrowUp') Pong.player.move = DIRECTION.UP;

            // Handle down arrow key event
            //if (key.key === 'ArrowDown') Pong.player.move = DIRECTION.DOWN;
        });

        // Stop the player from moving when there are no keys being pressed.
        // document.addEventListener('keyup', function (key) {
        //   Pong.player.move = DIRECTION.IDLE;
        // });

        document.addEventListener('click', function () {
            // If user clicks the page, start the game.
            if (Pong.running === false) {
                Pong.running = true;
                window.requestAnimationFrame(Pong.loop);
            }
        });
    },

    // Reset the ball location.
    // The ball starts at the loser's side.
    // Set a delay before the next round begins.
    _resetTurn: function (victor, loser) {
        this.ball = Ball.new.call(this, this.ball.speed);
        this.turn = loser;
        this.timer = (new Date()).getTime();

        victor.score++;
    },

    // Wait for a delay to have passed after each turn.
    _turnDelayIsOver: function () {
        return ((new Date()).getTime() - this.timer >= 1000);
    },
};

var Pong = Object.assign({}, Game);
Pong.initialize();