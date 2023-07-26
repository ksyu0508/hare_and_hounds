from collections import deque, namedtuple, Counter, defaultdict
import random
import math
import numpy as np
import functools 
cache = functools.lru_cache(10**6)

class Game:
    """A game is similar to a problem, but it has a terminal test instead of 
    a goal test, and a utility for each terminal state. To create a game, 
    subclass this class and implement `actions`, `result`, `is_terminal`, 
    and `utility`. You will also need to set the .initial attribute to the 
    initial state; this can be done in the constructor."""

    def actions(self, state):
        """Return a collection of the allowable moves from this state."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def is_terminal(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)
    
    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError
        

def play_game(game, strategies: dict, verbose=False):
    """Play a turn-taking game. `strategies` is a {player_name: function} dict,
    where function(state, game) is used to get the player's move."""
    state = game.initial
    while not game.is_terminal(state):
        player = state.to_move
        move = strategies[player](game, state)
        state = game.result(state, move)
        if verbose: 
            print('Player', player, 'move:', move)
            print(state)
    return state

def minimax_search(game, state):
    """Search game tree to determine best move; return (value, move) pair."""

    player = state.to_move

    def max_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a))
            if v2 > v:
                v, move = v2, a
        return v, move

    def min_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a))
            if v2 < v:
                v, move = v2, a
        return v, move

    return max_value(state)

infinity = math.inf

def alphabeta_search(game, state, max_depth=10):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move


    def max_value(state, alpha, beta, depth):
        # 같은 상황으로 돌아오는 수에 대해서 탐색 종료
        if depth >= 4:
            if state.prev[-1] == state.prev[-3][::-1] and state.prev[-2] == state.prev[-4][::-1]:
                return game.utility(state, player), None
        # 게임 종료 시점 혹은 max_depth 수 이후 시점에서 상태 평가

        if game.is_terminal(state) or depth >= max_depth:
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta, depth + 1)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move

        return v, move

    def min_value(state, alpha, beta, depth):
        if depth >= 4:
            if state.prev[-1] == state.prev[-3][::-1] and state.prev[-2] == state.prev[-4][::-1]:
                return game.utility(state, player), None  
        if game.is_terminal(state) or depth >= max_depth:
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta, depth + 1)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move

        return v, move

    return max_value(state, -infinity, +infinity, 0)


class HAH(Game):
    def __init__(self):
        self.moves = {(0, 1), (0, 2), (0, 3),
              (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), 
                      (2, 1), (2, 2), (2, 3)}
        self.initial = HAHBoard(to_move = 'H', utility=0, turn=0, prev=[])
        self.initial[0, 1] = 'H'
        self.initial[1, 0] = 'H'
        self.initial[2, 1] = 'H'
        self.initial[1, 4] = 'R'

    def actions(self, state): # return all available action from current state(input)
        moves = []

        if state.to_move == 'R':
            for curr in self.moves:
                if state[curr] == 'R':
                    for next in connections[curr[0]][curr[1]]:
                        if state[next] == state.empty:
                            moves.append([curr, next])

        elif state.to_move == 'H':
            for curr in self.moves:
                if state[curr] == 'H':
                    for next in connections[curr[0]][curr[1]]:
                        if curr[1] > next[1]: # hounds can't go back
                            continue
                        if state[next] == state.empty:
                            moves.append([curr, next])

        return moves

    def result(self, state, move): # return next game state
        player = state.to_move
        prev = state.prev[:]
        prev.append(move)
        state = state.new(move[0], {move[1]: player}, to_move=('H' if player == 'R' else 'R'), utility=0, turn=state.turn + 1, prev=prev)

        state.utility += heuristic(state)
        state.utility += - 100 * state.turn # 턴에 따른 가중치
        state.utility += hound_win(state) * 999999 # 같은 조건에서 빨리 끝나는 수 선택을 위해서 무한대 아닌 실수로 승리 조건

        return state
    
    def utility(self, state, player): # return value of state -> hueristic
        return state.utility if player == 'H' else -state.utility
    
    def is_terminal(self, state): # return whether game is over or not
        return hound_win(state)
    
    def display(self, board): print(board)

connections = [[[] for _ in range(5)] for _ in range(3)]
connections[0][1] = [(1, 2), (1, 1), (1, 0), (0, 2)]
connections[0][2] = [(0, 1), (1, 2), (0, 3)]
connections[0][3] = [(0, 2), (1, 2), (1, 3), (1, 4)]
connections[1][0] = [(0, 1), (1, 1), (2, 1)]
connections[1][1] = [(0, 1), (1, 0), (2, 1), (1, 2)]
connections[1][2] = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (2, 1), (2, 2), (2, 3)]
connections[1][3] = [(0, 3), (1, 2), (2, 3), (1, 4)]
connections[1][4] = [(0, 3), (1, 3), (2, 3)]
connections[2][1] = [(1, 0), (1, 1), (1, 2), (2, 2)]
connections[2][2] = [(2, 1), (1, 2), (2, 3)]
connections[2][3] = [(2, 2), (1, 2), (1, 3), (1, 4)]

def heuristic(state):
    total_h = 0
    # h func for hound
    # minus score for hare

    # 1. 토끼 왼쪽에 늑대 있을 수록 점수
    cnt = 0

    for x in range(3):
        for y in range(5):
            if state[x, y] == 'R':
                hare = (x, y)

    hounds = []

    for x in range(3):
        for y in range(5):
            if state[x, y] == 'H':
                hounds.append((x, y))
                if hare[1] > hounds[-1][1]:
                    cnt += 1
    
    total_h += 500 * cnt

    # 2. 맨 오른쪽 구석에 들어갈 시 페널티
    if state[1, 4] == 'H':
        total_h += -5000

    # 3. 늑대로 닫힌 공간을 만들 경우 점수, 공간이 좁아질 수록 점수 높아짐

    # BFS
    visited = [[False] * 5 for _ in range(3)]
    visited[0][0] = True
    visited[0][-1] = True
    visited[-1][0] = True
    visited[-1][-1] = True

    distance = [[0] * 5 for _ in range(3)]
    
    for x, y in hounds:
        visited[x][y] = True

    q = deque([hare])

    while q:
        current_node = q.popleft()
        x, y = current_node

        for next_x, next_y in connections[x][y]:
            if not visited[next_x][next_y]:
                visited[next_x][next_y] = True
                distance[next_x][next_y] = distance[x][y] + 1
                q.append((next_x, next_y))

    cnt = 0
    for x in range(3):
        for y in range(5):
            if not visited[x][y]:
                cnt += 1
    
    # 3-1. 토끼가 1,0에 도달하지 못하는 경우, 즉 늑대에 의해서 둘러 싸인 경우
    if not visited[1][0]: 
        total_h += 5000

    # 3-2. 토끼 기준 주변 공간 점수, 가까운 영역일수록 높은 점수(늑대 기준 음수)

    distance_weight = [0, 10, 5, 2, 1, 1, 1, 1]
    for x in range(3):
        for y in range(5):
            total_h += -100 * distance_weight[distance[x][y]]

    return total_h
    

class HAHBoard(defaultdict):
    empty = '○'
    off = '#'
    
    def __init__(self, to_move=None, **kwds):
        self.width = 3
        self.height = 5
        self.__dict__.update(to_move=to_move, **kwds)

    def new(self, before: tuple, after: dict, **kwds) -> 'HAHBoard':
        board = HAHBoard(**kwds)
        board.update(self)
        board.pop(before)
        board.update(after)
        return board

    def __missing__(self, loc):
        x, y = loc
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.empty
        else:
            return self.off

    def __hash__(self): 
        return hash(tuple(sorted(self.items()))) + hash(self.to_move)
    
    def __repr__(self):
        '''
            1---2---3 
          / | \ | / | \
        4---5---6---7---8
          \ | / | \ | /
            9---10--11
        '''
        n = f'    {self[0, 1]}---{self[0, 2]}---{self[0, 3]}\n' + \
            f'  / | \\ | / | \\\n' + \
            f'{self[1, 0]}---{self[1, 1]}---{self[1, 2]}---{self[1, 3]}---{self[1, 4]}\n' + \
            f'  \\ | / | \\ | /\n' + \
            f'    {self[2, 1]}---{self[2, 2]}---{self[2, 3]}'      
        
        return n

def hound_win(state):
    rabbit_win = True
    r_y = -1

    # hare's early win condition
    # hare wins if there is no hound in front of hare
    for x in range(3):
        for y in range(5):
            if state[x, y] == 'R':
                r_y = y
    for x in range(3):
        for y in range(5):
            if state[x, y] == 'H' and r_y > y:
                rabbit_win = False
    
    if rabbit_win:
        return -1

    # hound's win case
    if state[2, 1] == 'H' and state[1, 2] == 'H' and state[2, 3] == 'H' and state[2, 2] == 'R':
        return 1
    elif state[0, 1] == 'H' and state[1, 2] == 'H' and state[0, 3] == 'H' and state[0, 2] == 'R':
        return 1
    elif state[0, 3] == 'H' and state[1, 3] == 'H' and state[2, 3] == 'H' and state[1, 4] == 'R':
        return 1
    
    if state.turn > 40:
        return -1

    return 0

def random_player(game, state):
    return random.choice(list(game.actions(state)))

def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move

def human_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    actions = list(game.actions(state))

    print('available moves: ', end='')
    for i, action in enumerate(actions): 
        print(f'{i}: {action} ', end='')
    print('')
    if len(actions):
        while True:
            idx = input(f'Select Your move (0 ~ {len(actions)-1}): ')
            if str.isdigit(idx):
                idx = int(idx)
                if idx in range(0, len(actions)):
                    break
            else:
                print('Integer input only, Try again.')
    else:
        print('no legal moves: passing turn to next player')

    return actions[idx]

def player(search_algorithm, **kargs):
    """A game player who uses the specified search algorithm"""
    return lambda game, state: search_algorithm(game, state, **kargs)[1]


if __name__ == '__main__':
    while True:
        side = input(f'Select Your Sides (0: Hare, 1: Hound): ')
        if str.isdigit(side):
            side = int(side)
            if side in range(0, 2):
                break
        else:
            print('Integer input only, Try again.')

    if side == 0:
        play_game(HAH(), dict(R=human_player, H=player(alphabeta_search, max_depth=8)), verbose=True).utility
    else:
        play_game(HAH(), dict(H=human_player, R=player(alphabeta_search, max_depth=8)), verbose=True).utility