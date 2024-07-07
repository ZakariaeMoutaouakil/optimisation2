from pickle import load

from matplotlib.pyplot import tight_layout, show, subplots


def plot_results() -> None:
    dp = False
    with open('results.pkl', 'rb') as f:
        results = load(f)

    fig, ax = subplots(figsize=(12, 6))

    if dp:
        # Plot the times taken for each computation
        for result in results:
            n = result['n']
            ax.plot(n, result['time_simulation'], 'ro', label='Simulation Time' if result == results[0] else "")
            ax.plot(n, result['time_analytical'], 'go', label='Combinatorial Time' if result == results[0] else "")
            ax.plot(n, result['time_my_max_cdf'], 'bo', label='Poissonization Time' if result == results[0] else "")
            ax.plot(n, result['time_dynamic_programming'], 'yo',
                    label='Dynamic Programming Time' if result == results[0] else "")
    else:
        # Plot the times taken for each computation
        for result in results:
            n = result['n']
            ax.plot(n, result['time_simulation'], 'ro', label='Simulation Time' if result == results[0] else "")
            ax.plot(n, result['time_analytical'], 'go', label='Combinatorial Time' if result == results[0] else "")
            ax.plot(n, result['time_my_max_cdf'], 'bo', label='Poissonization Time' if result == results[0] else "")
    ax.set_xlabel('n')
    ax.set_ylabel('Time (seconds)')
    ax.legend()
    ax.set_title('Computation Time Comparison')

    tight_layout()
    show()


if __name__ == "__main__":
    plot_results()
