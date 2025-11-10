import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


# Define the function
def f(x, y):
    return (x - 2) ** 2 + (y - 3) ** 2 + np.sin(3 * x) + np.cos(3 * y)


def compute_gradient_finite_diff(func, x, y, epsilon=1e-5):
    """
    Compute gradient using finite differences
    """
    # Partial derivative with respect to x
    grad_x = (func(x + epsilon, y) - func(x - epsilon, y)) / (2 * epsilon)

    # Partial derivative with respect to y
    grad_y = (func(x, y + epsilon) - func(x, y - epsilon)) / (2 * epsilon)

    return grad_x, grad_y


def sgd_optimize(func, initial_point, learning_rate=0.1, n_iterations=100, epsilon=1e-5):
    """
    Stochastic Gradient Descent with finite difference gradient approximation

    Parameters:
    -----------
    func : function
        The objective function to minimize
    initial_point : tuple
        Starting point (x0, y0)
    learning_rate : float
        Step size for gradient descent
    n_iterations : int
        Number of iterations
    epsilon : float
        Small value for finite difference approximation

    Returns:
    --------
    history : list
        List of (x, y, loss) tuples for each iteration
    """
    x, y = initial_point
    history = [(x, y, func(x, y))]

    for i in range(n_iterations):
        # Step 2: Compute gradient using finite differences
        grad_x, grad_y = compute_gradient_finite_diff(func, x, y, epsilon)

        # Step 3: Take a small step in the negative gradient direction
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

        # Record the new position and loss
        loss = func(x, y)
        history.append((x, y, loss))

        # Optional: Print progress every 20 iterations
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1}: x={x:.4f}, y={y:.4f}, loss={loss:.4f}")

    return history


def visualize_optimization(func, history, title="SGD Optimization Path", save_path="sgd_optimization.png"):
    """
    Visualize the optimization path on the loss surface and save as PNG
    """
    # Create a grid for the contour plot
    x_range = np.linspace(-1, 5, 200)
    y_range = np.linspace(0, 6, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 6))

    # Subplot 1: Contour plot with optimization path
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)

    # Extract path coordinates
    path_x = [h[0] for h in history]
    path_y = [h[1] for h in history]

    # Plot the optimization path
    ax1.plot(path_x, path_y, 'r.-', linewidth=2, markersize=5, label='Optimization Path')
    ax1.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
    ax1.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='End')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{title} - Contour View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Loss over iterations
    ax2 = fig.add_subplot(122)
    losses = [h[2] for h in history]
    ax2.plot(losses, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Convergence')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.show()

    return fig


def visualize_3d_optimization(func, history, save_path="sgd_3d.png"):
    """
    3D visualization of the optimization path on the loss surface
    """
    # Create a grid for the surface
    x_range = np.linspace(-1, 5, 100)
    y_range = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')

    # Extract path coordinates
    path_x = [h[0] for h in history]
    path_y = [h[1] for h in history]
    path_z = [h[2] for h in history]

    # Plot the optimization path
    ax.plot(path_x, path_y, path_z, 'r.-', linewidth=3, markersize=8, label='SGD Path')
    ax.plot([path_x[0]], [path_y[0]], [path_z[0]], 'go', markersize=12, label='Start')
    ax.plot([path_x[-1]], [path_y[-1]], [path_z[-1]], 'ro', markersize=12, label='End')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y) - Loss')
    ax.set_title('SGD Optimization on Loss Surface')
    ax.legend()

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3D visualization to {save_path}")
    plt.show()

    return fig


def create_animation_gif(func, history, save_path="sgd_animation.gif"):
    """
    Create an animated GIF showing the SGD optimization process
    """
    # Create a grid for the contour plot
    x_range = np.linspace(-1, 5, 200)
    y_range = np.linspace(0, 6, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Setup contour plot
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('SGD Optimization Path')
    ax1.grid(True, alpha=0.3)

    # Setup loss plot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Convergence')
    ax2.grid(True, alpha=0.3)

    # Extract data
    path_x = [h[0] for h in history]
    path_y = [h[1] for h in history]
    losses = [h[2] for h in history]

    # Initialize plots
    path_line, = ax1.plot([], [], 'r-', linewidth=2, label='Path')
    current_point, = ax1.plot([], [], 'ro', markersize=10)
    start_point, = ax1.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')

    loss_line, = ax2.plot([], [], 'b-', linewidth=2)

    # Set limits for loss plot
    ax2.set_xlim(0, len(history))
    ax2.set_ylim(min(losses) * 0.9, max(losses) * 1.1)

    ax1.legend()

    def init():
        path_line.set_data([], [])
        current_point.set_data([], [])
        loss_line.set_data([], [])
        return path_line, current_point, loss_line

    def animate(frame):
        # Update path
        path_line.set_data(path_x[:frame+1], path_y[:frame+1])
        current_point.set_data([path_x[frame]], [path_y[frame]])

        # Update loss plot
        loss_line.set_data(range(frame+1), losses[:frame+1])

        # Update title with current iteration info
        if frame < len(history):
            ax1.set_title(f'SGD Optimization - Iteration {frame}, Loss: {losses[frame]:.4f}')

        return path_line, current_point, loss_line

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(history),
                        interval=50, blit=True, repeat=True)

    # Save as GIF
    writer = animation.PillowWriter(fps=20)
    anim.save(save_path, writer=writer)
    print(f"Saved animation to {save_path}")

    plt.close()
    return anim


def create_3d_animation_gif(func, history, save_path="sgd_3d_animation.gif"):
    """
    Create a 3D animated GIF showing the SGD optimization on the surface
    """
    # Create a grid for the surface
    x_range = np.linspace(-1, 5, 50)
    y_range = np.linspace(0, 6, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    # Extract path data
    path_x = [h[0] for h in history]
    path_y = [h[1] for h in history]
    path_z = [h[2] for h in history]

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')

    # Initialize path line and point
    path_line, = ax.plot([], [], [], 'r-', linewidth=3, label='SGD Path')
    current_point, = ax.plot([], [], [], 'ro', markersize=10)
    start_point, = ax.plot([path_x[0]], [path_y[0]], [path_z[0]], 'go', markersize=10, label='Start')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y) - Loss')
    ax.set_title('SGD Optimization on Loss Surface')
    ax.legend()

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

    def init():
        path_line.set_data([], [])
        path_line.set_3d_properties([])
        current_point.set_data([], [])
        current_point.set_3d_properties([])
        return path_line, current_point

    def animate(frame):
        # Update path
        path_line.set_data(path_x[:frame+1], path_y[:frame+1])
        path_line.set_3d_properties(path_z[:frame+1])

        # Update current point
        current_point.set_data([path_x[frame]], [path_y[frame]])
        current_point.set_3d_properties([path_z[frame]])

        # Update title
        ax.set_title(f'SGD - Iteration {frame}, Loss: {path_z[frame]:.4f}')

        return path_line, current_point

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(history),
                        interval=50, blit=False, repeat=True)

    # Save as GIF
    writer = animation.PillowWriter(fps=20)
    anim.save(save_path, writer=writer)
    print(f"Saved 3D animation to {save_path}")

    plt.close()
    return anim


def run_multiple_sgd(func, n_runs=5, learning_rate=0.1, n_iterations=100, save_path="sgd_multiple_runs.png"):
    """
    Run SGD from multiple random starting points
    """
    fig = plt.figure(figsize=(12, 8))

    # Create contour plot
    x_range = np.linspace(-1, 5, 200)
    y_range = np.linspace(0, 6, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    ax = fig.add_subplot(111)
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)

    colors = plt.cm.rainbow(np.linspace(0, 1, n_runs))

    all_histories = []
    for i in range(n_runs):
        # Step 1: Sample random starting point
        x0 = np.random.uniform(-1, 5)
        y0 = np.random.uniform(0, 6)

        print(f"\nRun {i+1}: Starting from ({x0:.4f}, {y0:.4f})")

        # Run SGD
        history = sgd_optimize(func, (x0, y0), learning_rate, n_iterations)
        all_histories.append(history)

        # Plot path
        path_x = [h[0] for h in history]
        path_y = [h[1] for h in history]
        ax.plot(path_x, path_y, '.-', color=colors[i], linewidth=1.5,
                markersize=3, alpha=0.7, label=f'Run {i+1}')
        ax.plot(path_x[0], path_y[0], 'o', color=colors[i], markersize=8)
        ax.plot(path_x[-1], path_y[-1], 's', color=colors[i], markersize=8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Multiple SGD Runs from Random Starting Points')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved multiple runs visualization to {save_path}")
    plt.show()

    return all_histories


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("STOCHASTIC GRADIENT DESCENT WITH FINITE DIFFERENCES")
    print("=" * 60)

    # Single run example
    print("\n1. Single SGD Run")
    print("-" * 40)

    # Step 1: Sample a random starting point
    x0 = np.random.uniform(-1, 5)
    y0 = np.random.uniform(0, 6)
    print(f"Random starting point: ({x0:.4f}, {y0:.4f})")
    print(f"Initial loss: {f(x0, y0):.4f}")

    # Run SGD
    history = sgd_optimize(
        func=f,
        initial_point=(x0, y0),
        learning_rate=0.1,
        n_iterations=100,
        epsilon=1e-5
    )

    # Print final results
    final_x, final_y, final_loss = history[-1]
    print(f"\nFinal point: ({final_x:.4f}, {final_y:.4f})")
    print(f"Final loss: {final_loss:.4f}")
    print(f"True minimum is approximately at (2, 3)")

    # Visualize and save the optimization
    print("\n2. Visualizing and Saving Optimization Path")
    print("-" * 40)
    visualize_optimization(f, history)

    # 3D visualization
    print("\n3. 3D Visualization")
    print("-" * 40)
    visualize_3d_optimization(f, history)

    # Create animated GIFs
    print("\n4. Creating Animated GIFs")
    print("-" * 40)
    create_animation_gif(f, history)
    create_3d_animation_gif(f, history)

    # Multiple runs from different starting points
    print("\n5. Multiple SGD Runs from Random Points")
    print("-" * 40)
    all_histories = run_multiple_sgd(f, n_runs=5, learning_rate=0.1, n_iterations=100)

    print("\n" + "=" * 60)
    print("SGD IMPLEMENTATION COMPLETE")
    print("All visualizations and animations have been saved!")
    print("Files created:")
    print("  - sgd_optimization.png (2D optimization path)")
    print("  - sgd_3d.png (3D surface with path)")
    print("  - sgd_animation.gif (2D animated optimization)")
    print("  - sgd_3d_animation.gif (3D animated optimization)")
    print("  - sgd_multiple_runs.png (multiple random starts)")
    print("=" * 60)