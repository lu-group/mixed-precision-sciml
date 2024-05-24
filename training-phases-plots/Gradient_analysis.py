#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by  
@author Jacob Goldman-Wetzler

Modified by 
@author Joel Hayford
"""

# ================  Load libraries and dependencies  =========================
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.size'] = 18
SEED = 0xdde
dde.config.set_random_seed(SEED)

# ============================================================================
# =======================  Define functions  =================================
# ============================================================================

def f1(x):
    return x * np.sin(5 * x)
def get_gradients_of_weights(model16):
    x_train = model16.data.train_x
    y_train = model16.data.train_y
    with tf.GradientTape() as tape:
        y_pred = model16.net.call(x_train)
        loss_fn = dde.losses.get("MSE")
        loss = loss_fn(y_train, y_pred)
    gradients16 = tape.gradient(loss,model16.net.trainable_weights)
    gradients161d = np.concatenate([gradient.numpy().ravel() for gradient in gradients16])
    return gradients161d

def get_weights(model16):
    return np.concatenate([weight.flatten() for weight in model16.net.get_weights()])

def cos_sim_and_dist_of_vectors(g16, g32):
    def cosine_similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        return dot_product / (norm_vector1 * norm_vector2)
    csim = cosine_similarity(g16, g32)
    dist = np.linalg.norm(g16 - g32)

    return csim, dist

class SaveGradientsCallback(dde.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.list_of_weights = []
    def on_epoch_begin(self):
        weights = get_weights(self.model)
        grads = get_gradients_of_weights(self.model)
#         print(np.array(weights).shape, np.array(grads).shape)
        self.list_of_weights.append((weights, grads))

# ============================================================================
# ===============  Create the float32 and float16 models  ====================
# ============================================================================

dde.config.set_default_float('float32')

geom = dde.geometry.Interval(-1, 1)
data = dde.data.Function(geom, f1, 16, 100)

net = dde.nn.FNN([1] + [10] * 2 + [1], "tanh", "Glorot uniform")
model32 = dde.Model(data, net)
model32.compile("adam", lr=0.001, metrics=["l2 relative error"])
model32.train(iterations=0)

dde.config.set_default_float('float16')

geom = dde.geometry.Interval(-1, 1)
data = dde.data.Function(geom, f1, 16, 100)

net = dde.nn.FNN([1] + [10] * 2 + [1], "tanh", "Glorot uniform")
model16 = dde.Model(data, net)
model16.compile("adam", lr=0.001, metrics=["l2 relative error"])
model16.train(iterations=0)

# ================= copy the weights from float32 to float16 ==================

print(model32.net.__dict__)
for i, layer in enumerate(model32.net.denses):
    model16.net.denses[i].set_weights(
        [tf.cast(w, dtype=tf.float16) for w in layer.get_weights()]
    )
    
# ============================================================================
# ===============  Train the float32 and float16 models  =====================
# ============================================================================

dde.config.set_default_float('float32')
cback32 = SaveGradientsCallback()
losshistory, train_state = model32.train(iterations=10_000,
                                         callbacks=[cback32]
)
dde.config.set_default_float('float16')
cback16 = SaveGradientsCallback()
losshistory, train_state = model16.train(iterations=10_000,callbacks=[cback16])

# ============================================================================
# ==========================  Process the data  ==============================
# ============================================================================

cos_similarity_grad = []
grad_mags = []
dist_grad = []

cos_similarity_weights = []
mags_weights = []
dist_weights = []

def adjust_gradients_for_float16(gradients, learning_rate):
    adjusted_gradients = gradients * learning_rate  
    return adjusted_gradients

def adjust_gradients_for_float32(gradients, learning_rate):
    adjusted_gradients = gradients * learning_rate
    
    return adjusted_gradients

adjusted_gradients_16_all = []
adjusted_gradients_32_all = []
weights_16_all = []
weights_32_all = []


for (weights16, grads16), (weights32, grads32) in zip(cback16.list_of_weights,cback32.list_of_weights):
    # calculate the metrics for the gradients
    csim, dist = cos_sim_and_dist_of_vectors(grads16, grads32)
    grad_mags.append([np.linalg.norm(grads16), np.linalg.norm(grads32)])
    cos_similarity_grad.append(csim)
    dist_grad.append(dist)
    # calculate the metrics for the weights
    csim, dist = cos_sim_and_dist_of_vectors(weights16, weights32)
    mags_weights.append([np.linalg.norm(weights16), np.linalg.norm(weights32)])
    cos_similarity_weights.append(csim)
    dist_weights.append(dist)
    
    weights_16_all.append(weights16)
    weights_32_all.append(weights32)
    
    

    adjusted_gradients_16 = adjust_gradients_for_float16(grads16, learning_rate= 0.001)
    adjusted_gradients_16_all.append(adjusted_gradients_16)

    zero_count_16 = sum(np.count_nonzero(adjusted == 0) for adjusted in adjusted_gradients_16)
    nan_count_16 = sum(np.count_nonzero(np.isnan(adjusted)) for adjusted in adjusted_gradients_16)
    
    adjusted_gradients_32 = adjust_gradients_for_float32(grads32, learning_rate= 0.001)
    adjusted_gradients_32_all.append(adjusted_gradients_32)

    zero_count_32 = sum(np.count_nonzero(adjusted == 0) for adjusted in adjusted_gradients_32)
    nan_count_32 = sum(np.count_nonzero(np.isnan(adjusted)) for adjusted in adjusted_gradients_32)


print(f"Gradients_16 adjusted to zero: {zero_count_16}")
print(f"Gradients_16 adjusted to NaN: {nan_count_16}")


# Convert the lists of NumPy arrays into lists of lists, replacing NaNs with a string for clarity
grads_16_lists = [list(np.where(np.isnan(grad), 'NaN', grad)) for grad in adjusted_gradients_16_all]
grads_32_lists = [list(np.where(np.isnan(grad), 'NaN', grad)) for grad in adjusted_gradients_32_all]

# Convert lists of lists into Pandas DataFrames
df_grads_16 = pd.DataFrame(grads_16_lists)
df_grads_32 = pd.DataFrame(grads_32_lists)

# Save to CSV
df_grads_16.to_csv('adjusted_gradients_float16.csv', index=False, header=False)
df_grads_32.to_csv('adjusted_gradients_float32.csv', index=False, header=False)

print(f"Gradients_32 adjusted to zero: {zero_count_32}")
print(f"Gradients_32 adjusted to NaN: {nan_count_32}")

# ============================================================================
# =======================  Visualize the results  ============================
# ============================================================================

grad_mags = np.array(grad_mags)
mags_weights = np.array(mags_weights)
epochaxis = np.linspace(0, 10_000, 10_000)
fig, ax = plt.subplots()
ax.plot(epochaxis, grad_mags[:,1],'r', label="Float32")
ax.plot(epochaxis, grad_mags[:,0],'b', label="Float16")
ax.set_yscale('log')
plt.xlabel('No. of iterations')
plt.ylabel('$L^2$ norm of gradients')
plt.xlim(left=0)
plt.legend(loc=(0.5, 0.8), frameon=False)
# Set the linewidth of the figure border to 1.5
for axis in ['top', 'bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(1.5)


plt.savefig("mags1632grads.pdf", format='pdf', bbox_inches='tight')
plt.show()

iterations = range(len(adjusted_gradients_16_all))

# Calculate mean absolute gradient at each iteration
mean_abs_grads_16 = [np.mean(np.abs(grads)) for grads in adjusted_gradients_16_all]
mean_abs_grads_32 = [np.mean(np.abs(grads)) for grads in adjusted_gradients_32_all]

plt.figure(figsize=(12, 6))

plt.plot(iterations, mean_abs_grads_16, label='Mean Abs Gradient Float16', color='red')
plt.plot(iterations, mean_abs_grads_32, label='Mean Abs Gradient Float32', color='blue')

plt.xlabel('Iterations')
plt.ylabel('Mean Absolute Gradient')
plt.title('Mean Absolute Adjusted Gradients Over Iterations')
plt.legend()
plt.grid(True)
plt.savefig("zero_grads.pdf", format='pdf', bbox_inches='tight')
plt.show()
plt.show()



# Function to create and plot binary heatmap
def plot_binary_heatmap(adjusted_gradients_all, filename):
    # Normalize gradient magnitudes for visualization
    max_len = max(len(grad) for grad in adjusted_gradients_all)
    binary_grads = np.zeros((max_len, len(adjusted_gradients_all)))

    for i, grad in enumerate(adjusted_gradients_all):
        # Set to 1 if gradient component is not zero, otherwise leave as 0
        binary_grads[:len(grad), i] = np.where(grad != 0, 1, 0)

    cmap = ListedColormap(['blue', 'red'])
    # Plotting
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(10, 8))
    plt.imshow(binary_grads, aspect='auto', cmap=cmap, interpolation='nearest')
    # Create custom legends
    red_patch = mpatches.Patch(color='red', label='Non-zero derivative')
    blue_patch = mpatches.Patch(color='blue', label='Zero derivative')
    plt.legend(handles=[red_patch, blue_patch], loc='upper center', bbox_to_anchor=(0.5,1.09), ncol=2, frameon=False, fontsize=20)
    
    
    plt.ylabel('Network parameter index')
    plt.xlabel('No. of iterations')
    plt.xlim(0,10000)
    # Start y-axis from 1
    plt.yticks(ticks=np.arange(0, max_len, step=20), labels=np.arange(1, max_len + 1, step=20))
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()




def plot_weight_updates_binary_heatmap(weights_all, filename):
    # Assuming weights_all is a list of arrays representing weights at each iteration
    num_iterations = len(weights_all)
    max_len = max(len(weights) for weights in weights_all)
    
    # Initialize the binary matrix
    binary_updates = np.zeros((max_len, num_iterations-1))
    
    for i in range(1, num_iterations):
        # Calculate the difference between consecutive weight arrays
        diff = weights_all[i][:len(weights_all[i-1])] - weights_all[i-1]
        # Mark as 1 (update occurred) if difference is non-zero
        binary_updates[:len(diff), i-1] = np.where(diff != 0, 1, 0)
        
    cmap = ListedColormap(['blue', 'red'])
    # Plotting
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(10, 8))
    plt.imshow(binary_updates, aspect='auto', cmap=cmap, interpolation='nearest')
    # Create custom legends
    red_patch = mpatches.Patch(color='red', label='Updated')
    blue_patch = mpatches.Patch(color='blue', label='Not updated')
    plt.legend(handles=[red_patch, blue_patch], loc='upper center', bbox_to_anchor=(0.5,1.09), ncol=2, frameon=False, fontsize=20)

    plt.ylabel('Network parameter index')
    plt.xlabel('No. of iterations')
    plt.xlim(0,10000)
    plt.yticks(ticks=np.arange(0, max_len, step=20), labels=np.arange(1, max_len + 1, step=20))
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()
    
print(type(weights_16_all))  # Should be <class 'list'>
if len(weights_16_all) > 0:
    print(type(weights_16_all[0]))  # Should be <class 'numpy.ndarray'> or similar array type
    print(weights_16_all[0].shape)  # Should show the shape of the array, e.g., (100,) for 100 weights
    
# Plot binary heatmap for Float16 gradients
plot_binary_heatmap(adjusted_gradients_16_all, 'binary_float16_grads_heatmap.pdf')

# Plot binary heatmap for Float32 gradients
plot_binary_heatmap(adjusted_gradients_32_all,  'binary_float32_grads_heatmap.pdf')

# Plot binary heatmap for Float32 gradients
plot_weight_updates_binary_heatmap(weights_16_all, 'binary_float16_weights_heatmap.pdf')


# Assuming weights_16_all is a list of weight arrays at each iteration
mean_weights_16 = []
prev_weight = None
# num_const = 0 
for i in range(1, len(weights_16_all)):
    unchanged = np.isclose(weights_16_all[i], weights_16_all[i-1], atol=1e-10)
    percent_unchanged = np.mean(unchanged) * 100
    mean_weights_16.append(percent_unchanged)
    
# Assuming weights_16_all is a list of weight arrays at each iteration
mean_weights_32 = []
for i in range(1, len(weights_32_all)):
    unchanged = np.isclose(weights_32_all[i], weights_32_all[i-1], atol=1e-10)
    percent_unchanged = np.mean(unchanged) * 100
    mean_weights_32.append(percent_unchanged)

# Calculate percentage of zero gradients for adjusted_gradients_16_all
mean_grads = [(np.mean(grad == 0) * 100) for grad in adjusted_gradients_16_all]
mean_grads_32 = [(np.mean(grad == 0) * 100) for grad in adjusted_gradients_32_all]


# Assuming equal number of iterations across weights and gradients
iterations1 = list(range(1, len(weights_16_all)))
iterations = list(range(len(mean_grads)))

# Plotting
plt.rcParams['font.size'] = 24
plt.figure(figsize=(10, 8))
plt.plot(iterations1, mean_weights_16, label = "Not updated", color='red')
plt.plot(iterations, mean_grads, label = "Zero derivative", color='blue')
plt.ylabel('Network parameters (%)')
plt.xlabel('No. of iterations')
# Set the x-axis to start at 0
plt.xlim(left=0)
plt.ylim(0,100)
plt.text(0.04, 0.95, 'float16', transform=plt.gca().transAxes, fontsize=24, verticalalignment='top', 
         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
leg1 = plt.legend(loc = 'right', frameon=False)
plt.savefig('weight percentages', format='pdf', bbox_inches='tight')
plt.show()

# Plotting
plt.rcParams['font.size'] = 24
plt.figure(figsize=(10, 8))
plt.plot(iterations1, mean_weights_32, label = "Not updated", color='red')
plt.plot(iterations, mean_grads_32, label = "Zero derivative", color='blue')
plt.ylabel('Network parameters (%)')
plt.xlabel('No. of iterations')
# Set the x-axis to start at 0
plt.xlim(left=0)
plt.ylim(-2,100) 
plt.text(0.04, 0.95, 'float32', transform=plt.gca().transAxes, fontsize=24, verticalalignment='top', 
         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
leg1 = plt.legend(loc = 'upper right', frameon=False)
plt.savefig('weight percentages_float32', format='pdf', bbox_inches='tight')
plt.show()


