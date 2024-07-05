import numpy as np
import matplotlib.pyplot as plt


def polynomial_decay_schedule(initial_lr: float, power: float, max_epochs: int = 100) -> np.ndarray:
    """
    Generate a polynomial decay learning rate schedule.

    Args:
        initial_lr: The initial learning rate.
        power: The power of the polynomial.
        max_epochs: The maximum number of epochs.

    Returns:
        An array of learning rates for each epoch.
    """
    epochs = np.arange(max_epochs)
    lr = initial_lr * ((1 - (epochs / max_epochs)) ** power)
    return lr


def natural_exp_decay_schedule(initial_lr: float, decay_rate: float, max_epochs: int = 100) -> np.ndarray:
    """
    Generate a natural exponential decay learning rate schedule.

    Args:
        initial_lr: The initial learning rate.
        decay_rate: The decay rate.
        max_epochs: The maximum number of epochs.

    Returns:
        An array of learning rates for each epoch.
    """
    epochs = np.arange(max_epochs)
    lr = initial_lr * np.exp(-decay_rate * epochs)
    return lr


def staircase_exp_decay_schedule(initial_lr: float, decay_rate: float, step_size: int, max_epochs: int = 100) -> np.ndarray:
    """
    Generate a staircase exponential decay learning rate schedule.

    Args:
        initial_lr: The initial learning rate.
        decay_rate: The decay rate.
        step_size: The step size.
        max_epochs: The maximum number of epochs.

    Returns:
        An array of learning rates for each epoch.
    """
    epochs = np.arange(max_epochs)
    lr = initial_lr * np.exp(-decay_rate * np.floor((1 + epochs) / step_size))
    return lr


def step_decay_schedule(initial_lr: float, decay_factor: float, step_size: int, max_epochs: int = 100) -> np.ndarray:
    """
    Generate a step decay learning rate schedule.

    Args:
        initial_lr: The initial learning rate.
        decay_factor: The decay factor.
        step_size: The step size.
        max_epochs: The maximum number of epochs.

    Returns:
        An array of learning rates for each epoch.
    """
    epochs = np.arange(max_epochs)
    lr = initial_lr * (decay_factor ** np.floor((1 + epochs) / step_size))
    return lr


def cosine_annealing_schedule(lr_min: float, lr_max: float, max_epochs: int = 100) -> np.ndarray:
    """
    Generate a cosine annealing learning rate schedule.

    Args:
        lr_min: The minimum learning rate.
        lr_max: The maximum learning rate.
        max_epochs: The maximum number of epochs.

    Returns:
        An array of learning rates for each epoch.
    """
    epochs = np.arange(max_epochs)
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epochs / max_epochs * np.pi))
    return lr


def exponential_decay_schedule(initial_lr: float, decay_rate: float, max_epochs: int = 100) -> np.ndarray:
    """
    Generate an exponential decay learning rate schedule.

    Args:
        initial_lr: The initial learning rate.
        decay_rate: The decay rate.
        max_epochs: The maximum number of epochs.

    Returns:
        An array of learning rates for each epoch.
    """
    epochs = np.arange(max_epochs)
    lr = initial_lr * np.exp(-decay_rate * epochs)
    return lr


def linear_decay_schedule(initial_lr: float, max_epochs: int = 100) -> np.ndarray:
    epochs = np.arange(max_epochs)
    lr = initial_lr * (1 - (epochs / max_epochs))
    return lr


# Define the learning rate schedules
max_epochs = 4131
schedules = {
    "Step Decay": step_decay_schedule(initial_lr=1.0, decay_factor=0.5, step_size=10, max_epochs=max_epochs),
    "Exponential Decay": exponential_decay_schedule(initial_lr=1.0, decay_rate=0.05, max_epochs=max_epochs),
    "Cosine Annealing": cosine_annealing_schedule(lr_min=0.01, lr_max=1.0, max_epochs=max_epochs),
    "Polynomial_1 Decay": polynomial_decay_schedule(initial_lr=1.0, power=1, max_epochs=max_epochs),
    "Polynomial_2 Decay": polynomial_decay_schedule(initial_lr=1.0, power=2, max_epochs=max_epochs),
    "Polynomial_2_2 Decay": polynomial_decay_schedule(initial_lr=1.0, power=2, max_epochs=max_epochs*2),
    "Natural Exp. Decay": natural_exp_decay_schedule(initial_lr=1.0, decay_rate=0.005, max_epochs=max_epochs),
    "Staircase Exp. Decay": staircase_exp_decay_schedule(initial_lr=1.0, decay_rate=0.05, step_size=10, max_epochs=max_epochs),
    "linear_decay_schedule": linear_decay_schedule(initial_lr=1.0, max_epochs=max_epochs)
}

# Define a color palette

# Plot with defined colors
plt.figure(figsize=(15, 10))
for schedule_name, schedule in schedules.items():
    plt.plot(schedule, label=schedule_name)

plt.title('Learning Rate Schedules', fontsize=20)
plt.ylabel('Learning Rate', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.6)
plt.minorticks_on()
plt.legend(prop={'size': 12})
plt.show()
