# EC2 ↔ EBS (single AZ) architecture diagram
# Requires: pip install matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch

# Canvas
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Availability Zone container
az = FancyBboxPatch(
    (0.7, 0.7), 8.6, 8.6,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    linewidth=2, edgecolor="black", facecolor="white"
)
ax.add_patch(az)
ax.text(5, 9.9, "Availability Zone", ha="center", va="top", fontsize=16, weight="bold")

# EC2 node (rounded rectangle with label)
ec2 = FancyBboxPatch(
    (3.5, 6.5), 3.0, 2.0,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    linewidth=2, edgecolor="black", facecolor="white"
)
ax.add_patch(ec2)
ax.text(5.0, 7.5, "EC2 Instance", ha="center", va="center", fontsize=13, weight="bold")

# Optional small overlay to suggest the EC2 "layers" icon look
overlay = Rectangle((5.9, 7.7), 0.9, 0.6, linewidth=1.2, edgecolor="black", facecolor="white")
ax.add_patch(overlay)
overlay2 = Rectangle((5.6, 7.4), 0.9, 0.6, linewidth=1.2, edgecolor="black", facecolor="white")
ax.add_patch(overlay2)

# EBS cylinder (ellipse + rect + ellipse outline)
cyl_width = 3.2
cyl_height = 1.2
cyl_x = 3.4
cyl_y = 3.0

# Top ellipse (filled)
top_ellipse = Ellipse(
    (cyl_x + cyl_width/2, cyl_y + 2.2 + cyl_height/2),
    cyl_width, cyl_height, linewidth=2, edgecolor="black", facecolor="white"
)
ax.add_patch(top_ellipse)

# Body rectangle
body_rect = Rectangle((cyl_x, cyl_y), cyl_width, 2.2, linewidth=2, edgecolor="black", facecolor="white")
ax.add_patch(body_rect)

# Bottom ellipse outline
bottom_ellipse = Ellipse(
    (cyl_x + cyl_width/2, cyl_y), cyl_width, cyl_height, linewidth=2, edgecolor="black", facecolor="white"
)
ax.add_patch(bottom_ellipse)

ax.text(5.0, 3.9, "EBS Volume", ha="center", va="center", fontsize=13, weight="bold")

# Connector (line) EC2 -> EBS
arrow = FancyArrowPatch((5.0, 6.5), (5.0, 4.2), arrowstyle="-", linewidth=2)
ax.add_patch(arrow)

# Caption
ax.text(
    5.0, 1.2,
    "Single-AZ attachment: one EC2 instance connected to one EBS volume\n"
    "(Volume and instance must be in the same Availability Zone)",
    ha="center", va="center", fontsize=10
)

# Save
plt.savefig("ec2_ebs_single_az.png", dpi=240, bbox_inches="tight")
plt.savefig("ec2_ebs_single_az.svg", bbox_inches="tight")
plt.show()
