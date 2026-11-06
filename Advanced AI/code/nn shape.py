import matplotlib.pyplot as plt
import networkx as nx

# ==============================================
# معماری بهینه برای ۵ ورودی و ۳ خروجی
# ==============================================
layer_sizes = [5, 16, 16, 8, 3]  # فقط ۲ لایه پنهان اصلی + ۱ لایه اختیاری
layer_names = ['Input', 'H1', 'H2', 'H3', 'Output']

# ==============================================
# ساختن و رسم گراف (کد مشابه قبل، با اندازه‌های کوچکتر)
# ==============================================
G = nx.DiGraph()
pos = {}
node_id = 0

for layer_idx, num_nodes in enumerate(layer_sizes):
    x = layer_idx
    for j in range(num_nodes):
        y = -j + (num_nodes - 1) / 2
        G.add_node(node_id, layer=layer_idx)
        pos[node_id] = (x, y)
        node_id += 1

node_counter = 0
for layer_idx in range(len(layer_sizes) - 1):
    start_id = node_counter
    node_counter += layer_sizes[layer_idx]
    end_id = node_counter
    for i in range(start_id, start_id + layer_sizes[layer_idx]):
        for j in range(end_id, end_id + layer_sizes[layer_idx + 1]):
            G.add_edge(i, j)

plt.figure(figsize=(12, 6))

node_colors = []
for node in G.nodes(data=True):
    layer = node[1]['layer']
    if layer == 0:
        node_colors.append('#2ECC71')
    elif layer == len(layer_sizes) - 1:
        node_colors.append('#E74C3C')
    else:
        node_colors.append('#3498DB')

nx.draw(G, pos, node_size=120, node_color=node_colors, edge_color='#BDC3C7', width=0.5, alpha=0.8, arrows=False, with_labels=False)

for layer_idx, name in enumerate(layer_names):
    x = layer_idx
    y = (layer_sizes[layer_idx] - 1) / 2 + 0.8
    plt.text(x, y, f'{name}\n{layer_sizes[layer_idx]}',
             fontsize=10, fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#2C3E50'))

plt.title('Optimal Neural Network: 5 Inputs → 2 Hidden Layers → 3 Outputs', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('Optimal_Neural_Network.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()