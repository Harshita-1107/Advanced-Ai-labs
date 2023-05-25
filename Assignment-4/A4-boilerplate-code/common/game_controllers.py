import numpy as np
import random
from collections import defaultdict
import common.game_constants as game_constants
import common.game_state as game_state
import pygame


class KeyboardController:
    def GetAction(self, state: game_state.GameState) -> game_state.GameActions:
        keys = pygame.key.get_pressed()
        action = game_state.GameActions.No_action
        if keys[pygame.K_LEFT]:
            action = game_state.GameActions.Left
        if keys[pygame.K_RIGHT]:
            action = game_state.GameActions.Right
        if keys[pygame.K_UP]:
            action = game_state.GameActions.Up
        if keys[pygame.K_DOWN]:
            action = game_state.GameActions.Down

        return action


class AIController:
    def __init__(self) -> None:
        self.q_table = defaultdict(lambda: np.zeros(len(game_state.GameActions)))

        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1

    def GetAction(self, state: game_state.GameState) -> game_state.GameActions:
        if np.random.uniform() < self.epsilon:
            # Random action
            action = np.random.choice(list(game_state.GameActions))
        else:
            # Greedy action
            state_array = np.array([1, 2, 3])
            state_tuple = tuple(state_array)
            q_values = self.q_table[state_tuple]
       
            action = np.argmax(q_values)

        return action
       
    def to_array(self, state) -> np.ndarray:
        player_x = state.PlayerEntity.entity.x
        player_y = state.PlayerEntity.entity.y
        goal_x = state.GoalLocation.x
        goal_y = state.GoalLocation.y
        enemy_x = [enemy.entity.x for enemy in state.EnemyCollection]
        enemy_y = [enemy.entity.y for enemy in state.EnemyCollection]

        state_array = np.array([player_x, player_y, goal_x, goal_y] + enemy_x + enemy_y)

        return state_array

    def TrainModel(self):
        epochs = 1000

        for epoch in range(epochs):
            state = game_state.GameState()
            done = False
            while not done:
                # Choose an action
                action = self.GetAction(state)

                # Update the state and observe the reward
                obs = state.Update(action)

                if obs == game_state.GameObservation.Enemy_Attacked:
                    reward = -10
                    done = True
                elif obs == game_state.GameObservation.Reached_Goal:
                    reward = 100
                    done = True
                else:
                    reward = -1

                next_state_array = self.to_array(state)
                #next_state_array = state.to_array()
                next_q_values = self.q_table[tuple(next_state_array)]

                # Update the Q-table
                target = reward + self.discount_factor * np.max(next_q_values)
                state_array = self.to_array(state)
                #state_array = state.to_array()
               
                q_values = self.q_table[tuple(state_array)]
                action_tuple = (action,)  # creates a tuple with one element
                action_to_int = {game_state.GameActions.No_action: 0, game_state.GameActions.Left: 1, game_state.GameActions.Right: 2, game_state.GameActions.Up: 3,
                                        game_state.GameActions.Down: 4}
                if action not in action_to_int:
                       action_to_int[action] = 0
                q_index = tuple([action_to_int[a] for a in action_tuple])
                q_values[q_index] += self.learning_rate * (target - q_values[q_index])

                self.q_table[tuple(state_array)] = q_values

        # Save the Q-table to a file
        np.save('q_table.npy', dict(self.q_table))

    def EvaluateModel(self):
        attacked = 0
        reached_goal = 0
        state = game_state.GameState()
        for _ in range(100000):
            action = self.GetAction(state)
            obs = state.Update(action)
            if obs == game_state.GameObservation.Enemy_Attacked:
                attacked += 1
            elif obs == game_state.GameObservation.Reached_Goal:
                reached_goal += 1
        return (attacked, reached_goal)
