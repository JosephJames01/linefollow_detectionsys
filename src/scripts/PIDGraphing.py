import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the data from the file
with open('data.pkl', 'rb') as f:
    errors, times = pickle.load(f)

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')  # Red line for plotting error
ax.set_xlim(0, max(times) + 1)  # Set x-axis limit based on the maximum time
ax.set_ylim(-500, 500)  # Set y-axis limit, adjust as necessary

# Update function for the animation
def update_plot(frame):
    line.set_data(times, errors)
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=True)
plt.title('Error Over Time (pixels)')
plt.xlabel('Time')
plt.ylabel('Error')

plt.grid()
# Display the plot
plt.show()
