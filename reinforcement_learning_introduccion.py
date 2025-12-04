import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from IPython import display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# creamos el entorno de OpenAI Gym que balancea un post sobre un carrito
env = gym.make("CartPole-v1")

# configuración matplotlib
try:
    is_ipython = True
except ImportError:
    is_ipython = False

plt.ion()

# usar la gpu con cuda
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# transición del agente
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# clase donde guardamos las experiencias del agente
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# contructor de la red neuronal con 128 neuronas
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # pasa los estados por las diferentes capas
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Hiperparámetros (tamaño lote, recompensa futura, política exploración, velocidad de actualización de la red y tasa de aprendizaje)
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

# obtenemos el tamaño de acciones y observaciones
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# creamos las redes de política y objetivo
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# optimizador y Adam para mejorar la regularización
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

# valor para calcular la política
steps_done = 0


# selección de acción
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


# duración episodios
episode_durations = []


# mostrar el rendimiento
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    # hasta que la memoria no contenga al menos el batch size no se empieza a entrenar
    if len(memory) < BATCH_SIZE:
        return
    # batch aleatorio
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # separamos estados finales de no finales
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # creamos batches tensores para cada variable
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # calcular el valor de Q_predicho
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # calcular valores de Q del siguiente estado
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # calcular la pérdida
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # calcular gradientes, actualizar pesos
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# si cuda está disponible hará 600 episodios sino 50
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

# bucle de episodios
for i_episode in range(num_episodes):
    # reiniciar el entorno y preparar el estado
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # bucle de los pasos
    for t in count():
        action = select_action(state)  # selecciona la acción
        # ejecuta la acción
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = (
            terminated or truncated
        )  # terminated si llega a un estado terminal o truncated si se interrumpe por límite de tiempo

        # creamos el siguiente paso
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # guardamos la transición
        memory.push(state, action, next_state, reward)

        # y avanzamos al siguiente estado
        state = next_state

        # entrenamos la red
        optimize_model()

        # Hacemos un soft update
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        # si el episodio ha terminado
        # guardamos duración
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

# y mostramos el gráfico
plot_durations(show_result=True)
plt.ioff()
plt.show()
