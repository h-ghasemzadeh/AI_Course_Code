from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# Initialize presentation
prs = Presentation()

# Slide 1: Title Slide
slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "Gradient Descent Optimization for Linear Regression"
subtitle.text = "A Step-by-Step Explanation\nYour Name/Institution\nDate"

# Slide 2: Introduction to Linear Regression
slide_layout = prs.slide_layouts[1]
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Introduction to Linear Regression"
content.text = (
    "- Supervised learning algorithm for predicting continuous outputs.\n"
    "- Model: y = θ₀ + θ₁x₁ + ... + θₙxₙ\n"
    "- Goal: Find the best-fit line minimizing prediction errors."
)

# Slide 3: Cost Function (MSE)
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Cost Function (MSE)"
content.text = (
    "Mean Squared Error (MSE):\n"
    "J(θ) = (1/2m) Σ (hθ(xⁱ) - yⁱ)²\n\n"
    "Why MSE?\n"
    "- Differentiable, convex, penalizes large errors."
)

# Slide 4: What is Gradient Descent?
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "What is Gradient Descent?"
content.text = (
    "- Iterative optimization algorithm to minimize J(θ).\n"
    "- Analogy: Walking downhill in small steps.\n"
    "- Key Idea: Update parameters in the direction of steepest descent."
)

# Slide 5: Gradient Descent Algorithm
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Gradient Descent Algorithm"
content.text = (
    "Update Rule:\n"
    "θⱼ := θⱼ - α ∂J(θ)/∂θⱼ\n\n"
    "- α: Learning rate (step size)\n"
    "- ∂J(θ)/∂θⱼ: Gradient (slope)"
)

# Slide 6: Computing the Gradient
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Computing the Gradient"
content.text = (
    "Partial Derivative of MSE:\n"
    "∂J(θ)/∂θⱼ = (1/m) Σ (hθ(xⁱ) - yⁱ) xⱼⁱ\n\n"
    "Intuition: Adjust θⱼ based on error and feature xⱼ."
)

# Slide 7: Learning Rate (α)
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Learning Rate (α)"
content.text = (
    "- Too Small: Slow convergence\n"
    "- Too Large: Overshooting (divergence)\n"
    "- Goldilocks Principle: Choose α carefully."
)

# Slide 8: Types of Gradient Descent
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Types of Gradient Descent"
content.text = (
    "1. Batch GD: Uses all training data per step (slow, precise)\n"
    "2. Stochastic GD: Uses one random sample per step (fast, noisy)\n"
    "3. Mini-Batch GD: Compromise (small batches)"
)

# Slide 9: Convergence Criteria
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Convergence Criteria"
content.text = (
    "When to Stop?\n"
    "- Fixed iterations\n"
    "- Minimal change in J(θ)\n"
    "- Gradient magnitude threshold"
)

# Slide 10: Practical Tips
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Practical Tips"
content.text = (
    "- Feature Scaling: Normalize features for faster convergence\n"
    "- Debugging: Plot J(θ) to check convergence\n"
    "- Regularization: Avoid overfitting (e.g., Ridge/Lasso)"
)

# Slide 11: Demo (Optional)
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Demo (Python Implementation)"
content.text = (
    "Code Snippet:\n"
    "def gradient_descent(X, y, θ, α, iterations):\n"
    "    m = len(y)\n"
    "    for _ in range(iterations):\n"
    "        gradient = (1/m) * X.T @ (X @ θ - y)\n"
    "        θ -= α * gradient\n"
    "    return θ"
)

# Slide 12: Summary
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Summary"
content.text = (
    "Key Takeaways:\n"
    "1. Gradient descent minimizes MSE by iteratively updating θ.\n"
    "2. Learning rate and feature scaling are critical.\n"
    "3. Choose Batch/Stochastic/Mini-Batch GD based on needs."
)

# Slide 13: Q&A
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]
title.text = "Q&A"
content.text = "Open Floor: Questions?\n\nContact: your.email@example.com"

# Save the presentation
prs.save("Gradient_Descent_Linear_Regression.pptx")
print("PowerPoint file generated successfully!")