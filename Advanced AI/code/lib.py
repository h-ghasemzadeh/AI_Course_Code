"""
PyTorch Infographic Generator - Exact Match to Your Text File
دقیقاً مطابق با ساختار فایل متنی
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines
import os
import numpy as np

# ======================== CONFIGURATION ========================
CONFIG = {
    'dpi': 300,
    'bg_color': '#FFFFFF',
    'figsize': (16, 9),
    'colors': {
        'pytorch': '#EE4C2C',
        'blue': '#2E86C1',
        'orange': '#E67E22',
        'green': '#27AE60',
        'purple': '#8E44AD',
        'red': '#E74C3C',
        'teal': '#1ABC9C',
        'gray': '#95A5A6',
        'dark': '#2C3E50',
        'white': '#FFFFFF',
        'light_gray': '#F5F5F5'
    }
}

# ======================== HELPER FUNCTIONS ========================
def rounded_rect(ax, x, y, w, h, color=None, alpha=0.08, linewidth=1):
    """Draw a rounded rectangle"""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=color if color else 'white',
        edgecolor=color if color else '#E0E0E0',
        linewidth=linewidth if color else 0.5,
        alpha=alpha if color else 1,
        zorder=1
    )
    ax.add_patch(rect)
    return rect

def section_badge(ax, x, y, num, color):
    """Draw section number badge"""
    circle = Circle((x, y), 0.016, facecolor=color, edgecolor='white', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, str(num), fontsize=6, color='white', fontweight='bold',
            ha='center', va='center')

def draw_arrow(ax, start, end, color='#95A5A6', style='->'):
    """Draw an arrow"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        alpha=0.4,
        linewidth=1,
        connectionstyle='arc3,rad=0.05'
    )
    ax.add_patch(arrow)

# ======================== SECTION 1: LIBRARY OVERVIEW ========================
def section1(ax):
    """Section 1: Library Overview - Top bar"""
    x, y, w, h = 0.02, 0.82, 0.96, 0.15
    
    # Main card
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['pytorch'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 1, CONFIG['colors']['pytorch'])
    ax.text(x+0.045, y+h-0.018, 'LIBRARY OVERVIEW', fontsize=9, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    # PyTorch Logo
    logo_x, logo_y = x+0.035, y+0.065
    logo = Circle((logo_x, logo_y), 0.035, facecolor=CONFIG['colors']['pytorch'],
                  edgecolor='white', linewidth=2)
    ax.add_patch(logo)
    ax.text(logo_x, logo_y, 'PyTorch', fontsize=7, color='white',
            fontweight='bold', ha='center', va='center')
    
    # Info items
    items = [
        ('Developer', 'Meta AI', 0.14),
        ('First Release', '2016', 0.26),
        ('Language', 'Python, C++', 0.38),
        ('License', 'BSD-3', 0.50),
        ('Current Version', '2.0.1', 0.62),
        ('Website', 'pytorch.org', 0.74)
    ]
    
    for label, value, xpos in items:
        ax.text(xpos, y+0.085, label, fontsize=5.5, color=CONFIG['colors']['gray'],
                ha='center', va='center')
        ax.text(xpos, y+0.065, value, fontsize=7.5, color=CONFIG['colors']['dark'],
                fontweight='bold', ha='center', va='center')
    
    # Description - wrapped
    desc = "PyTorch is an open-source deep learning framework designed for research, "
    desc += "production AI, and large-scale neural network development."
    ax.text(x+0.03, y+0.035, desc, fontsize=6.5, color=CONFIG['colors']['dark'])

# ======================== SECTION 2: AI ECOSYSTEM ========================
def section2(ax):
    """Section 2: PyTorch in the AI Ecosystem"""
    x, y, w, h = 0.02, 0.59, 0.30, 0.21
    
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['blue'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 2, CONFIG['colors']['blue'])
    ax.text(x+0.045, y+h-0.018, 'PYTORCH IN THE AI ECOSYSTEM', fontsize=7.5, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    # PyTorch center
    cx, cy = x+0.15, y+0.08
    center = Circle((cx, cy), 0.04, facecolor=CONFIG['colors']['pytorch'],
                    edgecolor='white', linewidth=2)
    ax.add_patch(center)
    ax.text(cx, cy, 'PyTorch', fontsize=6.5, color='white', fontweight='bold',
            ha='center', va='center')
    
    # Nodes
    nodes = [
        ('Deep Learning', x+0.28, y+0.16, CONFIG['colors']['blue']),
        ('Data Processing', x+0.28, y+0.08, CONFIG['colors']['teal']),
        ('Machine Learning', x+0.28, y+0.00, CONFIG['colors']['green']),
        ('MLOps', x+0.20, y-0.04, CONFIG['colors']['purple']),
        ('Deployment', x+0.10, y-0.04, CONFIG['colors']['orange']),
        ('Computer Vision', x+0.02, y+0.00, CONFIG['colors']['red']),
        ('NLP', x+0.02, y+0.08, CONFIG['colors']['purple']),
        ('LLM', x+0.02, y+0.16, CONFIG['colors']['orange'])
    ]
    
    for name, nx, ny, color in nodes:
        line = mlines.Line2D([cx, nx], [cy, ny], color=color, alpha=0.2, linewidth=0.8)
        ax.add_line(line)
        node = Circle((nx, ny), 0.015, facecolor=color, edgecolor='white', 
                      linewidth=0.8, alpha=0.8)
        ax.add_patch(node)
        ax.text(nx, ny-0.022, name, fontsize=4.5, color=CONFIG['colors']['dark'],
                ha='center', va='center', fontweight='bold')

# ======================== SECTION 3: CORE ARCHITECTURE ========================
def section3(ax):
    """Section 3: Core Architecture"""
    x, y, w, h = 0.34, 0.59, 0.32, 0.21
    
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['orange'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 3, CONFIG['colors']['orange'])
    ax.text(x+0.045, y+h-0.018, 'CORE ARCHITECTURE', fontsize=7.5, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    # Vertical flow
    comps = [
        ('Tensor', x+0.16, y+0.16, CONFIG['colors']['blue']),
        ('Autograd', x+0.16, y+0.12, CONFIG['colors']['orange']),
        ('Dynamic Graph', x+0.16, y+0.08, CONFIG['colors']['green']),
        ('Neural Network', x+0.16, y+0.04, CONFIG['colors']['purple']),
        ('Loss Function', x+0.16, y+0.00, CONFIG['colors']['red']),
        ('Optimizer', x+0.16, y-0.04, CONFIG['colors']['teal']),
        ('CUDA Engine', x+0.16, y-0.08, CONFIG['colors']['pytorch']),
        ('Model', x+0.16, y-0.12, CONFIG['colors']['dark'])
    ]
    
    for name, cx, cy, color in comps:
        box = FancyBboxPatch(
            (cx-0.055, cy-0.011), 0.11, 0.022,
            boxstyle="round,pad=0.002",
            facecolor=color, alpha=0.10,
            edgecolor=color, linewidth=0.6
        )
        ax.add_patch(box)
        ax.text(cx, cy, name, fontsize=5.5, color=CONFIG['colors']['dark'],
                fontweight='bold', ha='center', va='center')
        
        if cy > comps[-1][2]:
            draw_arrow(ax, (cx, cy-0.011), (cx, cy-0.022), color)

# ======================== SECTION 4: TRAINING WORKFLOW ========================
def section4(ax):
    """Section 4: Neural Network Training Workflow"""
    x, y, w, h = 0.68, 0.59, 0.30, 0.21
    
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['green'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 4, CONFIG['colors']['green'])
    ax.text(x+0.045, y+h-0.018, 'TRAINING WORKFLOW', fontsize=7.5, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    # Two columns
    left = ['Dataset', 'DataLoader', 'Tensor', 'Forward Pass', 
            'Loss Calculation', 'Backpropagation']
    right = ['Optimizer', 'Weight Update', 'Validation', 
             'Testing', 'Saved Model', 'Deployment']
    
    # Left column
    for i, step in enumerate(left):
        cy = y + 0.17 - (i * 0.025)
        box = FancyBboxPatch(
            (x+0.04, cy-0.009), 0.10, 0.018,
            boxstyle="round,pad=0.002",
            facecolor=CONFIG['colors']['blue'], alpha=0.08,
            edgecolor=CONFIG['colors']['blue'], linewidth=0.5
        )
        ax.add_patch(box)
        ax.text(x+0.09, cy, step, fontsize=4.5, color=CONFIG['colors']['dark'],
                ha='center', va='center')
        if i < len(left)-1:
            draw_arrow(ax, (x+0.09, cy-0.009), (x+0.09, cy-0.019), CONFIG['colors']['blue'])
    
    # Right column
    for i, step in enumerate(right):
        cy = y + 0.17 - (i * 0.025)
        box = FancyBboxPatch(
            (x+0.48, cy-0.009), 0.10, 0.018,
            boxstyle="round,pad=0.002",
            facecolor=CONFIG['colors']['orange'], alpha=0.08,
            edgecolor=CONFIG['colors']['orange'], linewidth=0.5
        )
        ax.add_patch(box)
        ax.text(x+0.53, cy, step, fontsize=4.5, color=CONFIG['colors']['dark'],
                ha='center', va='center')
        if i < len(right)-1:
            draw_arrow(ax, (x+0.53, cy-0.009), (x+0.53, cy-0.019), CONFIG['colors']['orange'])
    
    # Middle arrow
    draw_arrow(ax, (x+0.15, y+0.16), (x+0.46, y+0.16), CONFIG['colors']['gray'])

# ======================== SECTION 5: MAJOR MODULES ========================
def section5(ax):
    """Section 5: Major Modules"""
    x, y, w, h = 0.02, 0.36, 0.96, 0.21
    
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['purple'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 5, CONFIG['colors']['purple'])
    ax.text(x+0.045, y+h-0.018, 'MAJOR MODULES', fontsize=7.5, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    modules = [
        ('torch', 'Core tensor ops', 0.08, CONFIG['colors']['blue']),
        ('torch.nn', 'Neural layers', 0.20, CONFIG['colors']['blue']),
        ('torch.optim', 'Optimizers', 0.32, CONFIG['colors']['orange']),
        ('torch.cuda', 'GPU support', 0.44, CONFIG['colors']['red']),
        ('torch.data', 'Data loading', 0.56, CONFIG['colors']['green']),
        ('torch.autograd', 'Auto-diff', 0.68, CONFIG['colors']['purple']),
        ('torch.distributed', 'Distributed', 0.80, CONFIG['colors']['teal']),
        ('torch.compile', 'JIT compile', 0.92, CONFIG['colors']['pytorch'])
    ]
    
    for name, desc, xpos, color in modules:
        box = FancyBboxPatch(
            (xpos-0.04, y+0.04), 0.08, 0.05,
            boxstyle="round,pad=0.002",
            facecolor=color, alpha=0.08,
            edgecolor=color, linewidth=0.6
        )
        ax.add_patch(box)
        ax.text(xpos, y+0.058, name, fontsize=6.5, color=color,
                fontweight='bold', ha='center', va='center')
        ax.text(xpos, y+0.04, desc, fontsize=4.5, color=CONFIG['colors']['gray'],
                ha='center', va='center')

# ======================== SECTION 6: INPUTS AND OUTPUTS ========================
def section6(ax):
    """Section 6: Inputs and Outputs"""
    x, y, w, h = 0.02, 0.13, 0.48, 0.21
    
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['teal'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 6, CONFIG['colors']['teal'])
    ax.text(x+0.045, y+h-0.018, 'INPUTS AND OUTPUTS', fontsize=7.5, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    # Inputs
    inputs = ['Images', 'Text', 'Audio', 'Video', 'Time Series', 'Tabular Data', 'Point Clouds']
    for i, inp in enumerate(inputs):
        cy = y + 0.15 - (i * 0.022)
        box = FancyBboxPatch(
            (x+0.05, cy-0.008), 0.10, 0.016,
            boxstyle="round,pad=0.002",
            facecolor=CONFIG['colors']['teal'], alpha=0.08,
            edgecolor=CONFIG['colors']['teal'], linewidth=0.4
        )
        ax.add_patch(box)
        ax.text(x+0.10, cy, inp, fontsize=4.5, color=CONFIG['colors']['dark'],
                ha='center', va='center')
    
    # Outputs
    outputs = ['Classification', 'Regression', 'Segmentation', 'Object Detection',
               'LLM Text Generation', 'Speech Recognition', 'Forecasting', 'Embeddings']
    for i, out in enumerate(outputs):
        cy = y + 0.15 - (i * 0.022)
        box = FancyBboxPatch(
            (x+0.34, cy-0.008), 0.12, 0.016,
            boxstyle="round,pad=0.002",
            facecolor=CONFIG['colors']['purple'], alpha=0.08,
            edgecolor=CONFIG['colors']['purple'], linewidth=0.4
        )
        ax.add_patch(box)
        ax.text(x+0.40, cy, out, fontsize=4.5, color=CONFIG['colors']['dark'],
                ha='center', va='center')
    
    # Center arrow
    draw_arrow(ax, (x+0.17, y+0.12), (x+0.32, y+0.12), CONFIG['colors']['gray'], '<->')
    ax.text(x+0.245, y+0.10, 'PyTorch', fontsize=6, color=CONFIG['colors']['pytorch'],
            fontweight='bold', ha='center', va='center')

# ======================== SECTION 7: ECOSYSTEM ========================
def section7(ax):
    """Section 7: PyTorch Ecosystem"""
    x, y, w, h = 0.52, 0.13, 0.46, 0.21
    
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['purple'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 7, CONFIG['colors']['purple'])
    ax.text(x+0.045, y+h-0.018, 'PYTORCH ECOSYSTEM', fontsize=7.5, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    # PyTorch center
    cx, cy = x+0.23, y+0.08
    center = Circle((cx, cy), 0.03, facecolor=CONFIG['colors']['pytorch'],
                    edgecolor='white', linewidth=2)
    ax.add_patch(center)
    ax.text(cx, cy, 'PyTorch', fontsize=5.5, color='white', fontweight='bold',
            ha='center', va='center')
    
    # Libraries
    libs = [
        ('TorchVision', x+0.44, y+0.20, CONFIG['colors']['blue']),
        ('TorchAudio', x+0.48, y+0.14, CONFIG['colors']['teal']),
        ('TorchText', x+0.48, y+0.06, CONFIG['colors']['green']),
        ('Lightning', x+0.44, y+0.00, CONFIG['colors']['orange']),
        ('FastAI', x+0.36, y-0.04, CONFIG['colors']['purple']),
        ('Captum', x+0.28, y-0.06, CONFIG['colors']['red']),
        ('ONNX', x+0.18, y-0.04, CONFIG['colors']['orange']),
        ('Transformers', x+0.10, y+0.00, CONFIG['colors']['blue']),
        ('Accelerate', x+0.06, y+0.06, CONFIG['colors']['teal']),
        ('PEFT', x+0.08, y+0.14, CONFIG['colors']['green']),
        ('vLLM', x+0.14, y+0.20, CONFIG['colors']['purple']),
        ('TensorRT', x+0.22, y+0.24, CONFIG['colors']['orange'])
    ]
    
    for name, lx, ly, color in libs:
        line = mlines.Line2D([cx, lx], [cy, ly], color=CONFIG['colors']['gray'],
                            alpha=0.12, linewidth=0.5)
        ax.add_line(line)
        node = Circle((lx, ly), 0.013, facecolor=color, edgecolor='white',
                      linewidth=0.5, alpha=0.7)
        ax.add_patch(node)
        ax.text(lx, ly-0.018, name, fontsize=4, color=CONFIG['colors']['dark'],
                ha='center', va='center')

# ======================== SECTION 8: APPLICATIONS ========================
def section8(ax):
    """Section 8: Applications"""
    x, y, w, h = 0.02, -0.09, 0.96, 0.20
    
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['green'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 8, CONFIG['colors']['green'])
    ax.text(x+0.045, y+h-0.018, 'APPLICATIONS', fontsize=7.5, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    apps = [
        ('Computer Vision', 0.06, CONFIG['colors']['blue']),
        ('Data Science', 0.13, CONFIG['colors']['teal']),
        ('Machine Learning', 0.20, CONFIG['colors']['green']),
        ('Deep Learning', 0.28, CONFIG['colors']['purple']),
        ('AI', 0.36, CONFIG['colors']['red']),
        ('Robotics', 0.43, CONFIG['colors']['orange']),
        ('Generative AI', 0.50, CONFIG['colors']['pytorch']),
        ('Diffusion Models', 0.58, CONFIG['colors']['purple']),
        ('Scientific Computing', 0.66, CONFIG['colors']['teal']),
        ('Time Series', 0.74, CONFIG['colors']['green']),
        ('Production AI', 0.82, CONFIG['colors']['orange']),
        ('MLOps', 0.90, CONFIG['colors']['blue'])
    ]
    
    for name, xpos, color in apps:
        box = FancyBboxPatch(
            (xpos-0.03, y+0.04), 0.06, 0.026,
            boxstyle="round,pad=0.002",
            facecolor=color, alpha=0.08,
            edgecolor=color, linewidth=0.4
        )
        ax.add_patch(box)
        ax.text(xpos, y+0.053, name, fontsize=4.5, color=CONFIG['colors']['dark'],
                fontweight='bold', ha='center', va='center')

# ======================== SECTION 9: ADVANTAGES & LIMITATIONS ========================
def section9(ax):
    """Section 9: Advantages and Limitations"""
    x, y, w, h = 0.02, -0.31, 0.96, 0.20
    
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['gray'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 9, CONFIG['colors']['gray'])
    ax.text(x+0.045, y+h-0.018, 'ADVANTAGES & LIMITATIONS', fontsize=7.5, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    # Advantages
    adv = ['Dynamic Computation Graph', 'Easy Debugging', 'Efficient GPU Support',
           'Large Community', 'Research Friendly', 'Open Source',
           'Distributed Training', 'Hugging Face Compatible', 'Rich Ecosystem', 'Production Ready']
    
    ax.text(x+0.04, y+0.12, 'ADVANTAGES', fontsize=7, fontweight='bold',
            color=CONFIG['colors']['green'])
    
    for i, a in enumerate(adv):
        row = i // 2
        col = i % 2
        px = x + 0.06 + (col * 0.28)
        py = y + 0.095 - (row * 0.022)
        ax.text(px, py, f'✓ {a}', fontsize=5, color=CONFIG['colors']['green'],
                ha='left', va='center')
    
    # Limitations
    lim = ['Memory Intensive', 'Large models require high-end GPUs']
    
    ax.text(x+0.70, y+0.12, 'LIMITATIONS', fontsize=7, fontweight='bold',
            color=CONFIG['colors']['red'])
    
    for i, l in enumerate(lim):
        px = x + 0.72
        py = y + 0.095 - (i * 0.022)
        ax.text(px, py, f'✗ {l}', fontsize=5, color=CONFIG['colors']['red'],
                ha='left', va='center')

# ======================== SECTION 10: ROADMAP ========================
def section10(ax):
    """Section 10: Learning Roadmap"""
    x, y, w, h = 0.02, -0.53, 0.96, 0.20
    
    rounded_rect(ax, x, y, w, h, CONFIG['colors']['orange'], 0.06)
    section_badge(ax, x+0.025, y+h-0.018, 10, CONFIG['colors']['orange'])
    ax.text(x+0.045, y+h-0.018, 'LEARNING ROADMAP', fontsize=7.5, 
            fontweight='bold', color=CONFIG['colors']['dark'], va='center')
    
    roadmap = ['Python', 'NumPy', 'Pandas', 'Matplotlib', 'Scikit-learn',
               'PyTorch', 'Transformers', 'LLMs', 'Agents', 'RAG', 'MLOps', 'Production AI']
    
    for i, step in enumerate(roadmap):
        px = 0.035 + (i * 0.078)
        py = y + 0.05
        
        color = CONFIG['colors']['pytorch'] if step == 'PyTorch' else CONFIG['colors']['blue']
        box = FancyBboxPatch(
            (px-0.025, py-0.011), 0.05, 0.022,
            boxstyle="round,pad=0.002",
            facecolor=color, alpha=0.15 if step != 'PyTorch' else 0.3,
            edgecolor=color, linewidth=0.6 if step != 'PyTorch' else 2
        )
        ax.add_patch(box)
        ax.text(px, py, step, fontsize=4.5, color=CONFIG['colors']['dark'],
                ha='center', va='center', fontweight='bold' if step == 'PyTorch' else 'normal')
        
        if i < len(roadmap)-1:
            draw_arrow(ax, (px+0.025, py), (px+0.05, py), CONFIG['colors']['gray'])

# ======================== MAIN FUNCTION ========================
def generate():
    """Generate the complete infographic"""
    
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.75, 1)
    ax.axis('off')
    
    # Generate all sections
    section1(ax)
    section2(ax)
    section3(ax)
    section4(ax)
    section5(ax)
    section6(ax)
    section7(ax)
    section8(ax)
    section9(ax)
    section10(ax)
    
    # Main Title
    ax.text(0.5, 0.99, 'PyTorch Deep Learning Library', fontsize=22,
            fontweight='bold', color=CONFIG['colors']['pytorch'],
            ha='center', va='center')
    ax.text(0.5, 0.972, 'Professional Vector Infographic for AI Textbooks', fontsize=10,
            color=CONFIG['colors']['gray'], ha='center', va='center')
    
    # Color Legend - مطابق فایل
    legend = [
        ('Blue: Core Framework', CONFIG['colors']['blue']),
        ('Orange: Training Pipeline', CONFIG['colors']['orange']),
        ('Green: Applications', CONFIG['colors']['green']),
        ('Purple: Ecosystem', CONFIG['colors']['purple']),
        ('Red: Hardware', CONFIG['colors']['red']),
        ('Teal: Input/Output', CONFIG['colors']['teal']),
        ('Gray: Utilities', CONFIG['colors']['gray'])
    ]
    
    lx = 0.22
    for i, (label, color) in enumerate(legend):
        pos = lx + (i * 0.09)
        rect = Rectangle((pos, -0.72), 0.012, 0.012, facecolor=color, edgecolor='none')
        ax.add_patch(rect)
        ax.text(pos+0.015, -0.715, label, fontsize=4.5, color=CONFIG['colors']['dark'], va='center')
    
    # Footer
    ax.text(0.5, -0.74, 'Designed for IEEE/Nature/ACM Style | Fully Editable | Vector Graphics | 4K Resolution',
            fontsize=5.5, color=CONFIG['colors']['gray'], ha='center', va='center', style='italic')
    
    # Save
    out = "pytorch_infographic_v3"
    os.makedirs(out, exist_ok=True)
    
    print("📁 Saving files...")
    for fmt, ext in [('svg', 'svg'), ('pdf', 'pdf'), ('png', 'png')]:
        path = f"{out}/pytorch_infographic.{ext}"
        plt.savefig(path, format=fmt, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ {ext.upper()}: {path}")
    
    plt.close()
    print(f"\n✅ Done! All files in '{out}'")
    print("\n📋 Structure matches your text file exactly:")
    print("   • 10 sections with numbering")
    print("   • Same content organization")
    print("   • Color legend as specified")

# ======================== RUN ========================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Generating PyTorch Infographic - Exact Match")
    print("=" * 60)
    try:
        generate()
        print("\n" + "=" * 60)
        print("✅ Complete!")
        print("💡 For PPTX: Insert SVG → Convert to Shape")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()